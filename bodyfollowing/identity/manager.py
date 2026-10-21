"""Identity ledger: binds body tracks to identities and keeps them honest.

Responsibilities
----------------
1. Apply the exponential decay to every identity at the current timestamp.
2. Refresh confidence only from **trustworthy** face evidence.
3. Re-identify a returning person from their face appearance descriptor.
4. Detect appearance conflicts (the body tracker swapped two people during a
   crossing) and refuse to confirm a bogus link.
5. Retire identities that have been unobservable for too long.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..association.face_body import AssociationResult, FaceBodyMatch
from ..config import AssociationConfig, IdentityConfig
from ..types import BodyTrack, FaceTrack, Identity, IdentityClaim, IdentityState
from ..utils.geometry import cosine_similarity, ema
from .enrollment import EnrollmentBook
from .states import decay_confidence, describe, state_for


@dataclass
class IdentityManager:
    """Maintains the identity ledger across frames."""

    config: IdentityConfig = field(default_factory=IdentityConfig)
    association_config: AssociationConfig = field(default_factory=AssociationConfig)

    identities: dict[int, Identity] = field(default_factory=dict)
    target_identity_id: int | None = None
    auto_lock: bool = True
    #: Optional registry of named people.  When present, a face matching an
    #: enrolled descriptor labels its identity, and a recognized person is
    #: locked on sight.
    enrollment: EnrollmentBook | None = None
    #: Set when the operator enrolled someone this session, so the demo can
    #: report it without polling the ledger.
    last_enrollment: str | None = None

    _next_id: int = 1
    _last_now: float = 0.0
    events: list[tuple[float, int, str, str]] = field(default_factory=list)

    # ---------------------------------------------------------------- helpers
    def _log(self, now: float, identity_id: int, kind: str, detail: str = "") -> None:
        self.events.append((now, identity_id, kind, detail))

    def get(self, identity_id: int | None) -> Identity | None:
        if identity_id is None:
            return None
        return self.identities.get(identity_id)

    def identity_for_body(self, body_id: int | None) -> Identity | None:
        if body_id is None:
            return None
        for ident in self.identities.values():
            if ident.body_track_id == body_id:
                return ident
        return None

    @property
    def target(self) -> Identity | None:
        return self.get(self.target_identity_id)

    def lock(self, identity_id: int | None) -> None:
        """Manually select the identity to follow."""
        self.target_identity_id = identity_id
        if identity_id is not None:
            self._log(self._last_now, identity_id, "lock")

    def lock_by_body(self, body_id: int) -> Identity | None:
        ident = self.identity_for_body(body_id)
        if ident is not None:
            self.lock(ident.identity_id)
        return ident

    def lock_at_point(self, x: float, y: float, bodies: list[BodyTrack]) -> Identity | None:
        """Pick the body under a click and lock onto its identity."""
        for body in bodies:
            if body.box.contains(x, y):
                return self.lock_by_body(body.track_id)
        return None

    # ------------------------------------------------------------------ decay
    def _decay_all(self, now: float) -> None:
        elapsed = max(0.0, now - self._last_now)
        for ident in self.identities.values():
            if elapsed > 0:
                ident.confidence = decay_confidence(
                    ident.confidence, elapsed, self.config.decay_tau
                )
            ident.last_update = now
        self._last_now = now

    # ----------------------------------------------------------------- update
    def update(
        self,
        bodies: list[BodyTrack],
        faces: list[FaceTrack],
        association: AssociationResult,
        now: float,
    ) -> list[Identity]:
        """Advance the ledger by one frame."""
        cfg = self.config
        self._decay_all(now)

        body_by_id = {b.track_id: b for b in bodies}
        face_by_id = {f.track_id: f for f in faces}

        # 1. Bodies that vanished lose their binding (the identity survives).
        for ident in self.identities.values():
            if ident.body_track_id is not None and ident.body_track_id not in body_by_id:
                ident.body_lost_time = now
                ident.body_track_id = None
                ident.face_track_id = None
                self._log(now, ident.identity_id, "body_lost")

        # 2. Collect trustworthy face evidence per body.
        evidence: dict[int, FaceBodyMatch] = {}
        for match in association.matches:
            if not match.is_trustworthy:
                continue
            if match.score < cfg.min_evidence_score:
                continue
            current = evidence.get(match.body_id)
            if current is None or match.score > current.score:
                evidence[match.body_id] = match

        # 3. Refresh bound identities from evidence, or note the face is gone.
        for ident in self.identities.values():
            if ident.body_track_id is None:
                continue
            match = evidence.get(ident.body_track_id)
            if match is None:
                ident.face_track_id = None
                continue
            face = face_by_id.get(match.face_id)
            self._apply_evidence(ident, match, face, now, reacquired=False)

        # 4. Bodies with evidence but no identity yet: re-identify or create.
        for body_id, match in evidence.items():
            if self.identity_for_body(body_id) is not None:
                continue
            face = face_by_id.get(match.face_id)
            existing = self._try_reidentify(face, now)
            if existing is not None:
                self._rebind(existing, body_id, match, face, now)
            else:
                self._create(body_id, match, face, now)

        # 5. Face-only re-identification for an identity whose body is missing.
        self._face_only_reacquire(faces, association, now)

        # 6. Recompute states, update timers, auto-lock, retire.
        for ident in self.identities.values():
            self._refresh_state(ident, now, body_by_id)
        self._maybe_auto_lock(now)
        self._retire(now)
        return list(self.identities.values())

    # ---------------------------------------------------------------- evidence
    def _apply_evidence(
        self,
        ident: Identity,
        match: FaceBodyMatch,
        face: FaceTrack | None,
        now: float,
        reacquired: bool,
    ) -> None:
        cfg = self.config
        score = float(match.score)

        # Appearance guard: a strong, stable face signature that suddenly
        # changes means the body tracker probably swapped people.
        if face is not None and face.descriptor is not None and ident.descriptor is not None:
            if ident.face_observations >= cfg.min_appearance_observations:
                sim = cosine_similarity(face.descriptor, ident.descriptor)
                if sim < cfg.appearance_conflict_threshold:
                    ident.confidence = min(ident.confidence, cfg.unknown_threshold * 0.9)
                    ident.face_track_id = None
                    ident.body_track_id = None
                    self._log(now, ident.identity_id, "appearance_conflict", f"sim={sim:.2f}")
                    return

        # A face that reappears after a gap is a *re-acquisition*, not a
        # continuation: the belief should snap to the quality of the new
        # observation instead of crawling back up through the EMA.  That is
        # what makes recovery from a head turn feel instant.
        gap = (now - ident.last_face_time) if ident.last_face_time >= 0 else 0.0
        is_reacquisition = reacquired or gap > cfg.freshness_window

        if reacquired:
            # Weight the evidence by how well the returning appearance matches
            # the one we recorded before the gap.
            sim = 1.0
            if face is not None and face.descriptor is not None and ident.descriptor is not None:
                sim = max(0.0, cosine_similarity(face.descriptor, ident.descriptor))
            target_conf = score * sim
        else:
            target_conf = score

        if ident.face_observations == 0:
            # First evidence sets the belief directly; smoothing it up from
            # zero would delay CONFIRMED by several frames for no reason.
            ident.confidence = target_conf
        elif is_reacquisition:
            ident.confidence = max(ident.confidence, target_conf)
        else:
            ident.confidence = ema(ident.confidence, target_conf, cfg.evidence_smoothing)

        ident.face_observations += 1
        ident.last_face_time = now
        ident.association_score = score
        ident.face_track_id = match.face_id if face is not None else None

        if face is not None and face.descriptor is not None:
            if ident.descriptor is None or ident.descriptor.size != face.descriptor.size:
                ident.descriptor = face.descriptor.copy()
            else:
                merged = ema(ident.descriptor, face.descriptor, 0.3)
                norm = float(np.linalg.norm(merged))
                ident.descriptor = merged / norm if norm > 0 else merged

        self._recognize(ident, face, now)

    def _recognize(self, ident: Identity, face: FaceTrack | None, now: float) -> None:
        """Label ``ident`` from the enrollment book, if the face matches.

        A name is only ever *set* here, never overwritten by a different one:
        once we have committed to "this is Alice" we do not silently relabel
        the same identity as somebody else because a later frame looked
        marginal.  A genuine conflict is left to the appearance guard in
        ``_apply_evidence``, which detaches the link instead.
        """
        if self.enrollment is None or face is None or face.descriptor is None:
            return
        match = self.enrollment.match(face.descriptor)
        if match is None:
            return
        if ident.name is None:
            ident.name = match.name
            self._log(now, ident.identity_id, "recognized", f"{match.name} sim={match.similarity:.2f}")
        elif ident.name != match.name:
            self._log(
                now,
                ident.identity_id,
                "recognition_conflict",
                f"had={ident.name} saw={match.name}",
            )

    def _try_reidentify(self, face: FaceTrack | None, now: float) -> Identity | None:
        """Match a returning face to an identity that currently has no body."""
        if face is None or face.descriptor is None:
            return None
        cfg = self.config
        scored: list[tuple[float, Identity]] = []
        for ident in self.identities.values():
            if ident.body_track_id is not None or ident.descriptor is None:
                continue
            if ident.face_observations < 1:
                continue
            if now - ident.last_face_time > self.association_config.reid_max_gap_seconds:
                continue
            sim = cosine_similarity(face.descriptor, ident.descriptor)
            scored.append((sim, ident))

        if not scored:
            return None
        scored.sort(key=lambda item: item[0], reverse=True)
        best_sim, best = scored[0]
        if best_sim < cfg.appearance_reid_similarity:
            return None
        # Two look-alikes must not be resolved by a coin flip.
        if len(scored) > 1 and (best_sim - scored[1][0]) < cfg.appearance_reid_margin:
            return None
        return best

    def _face_only_reacquire(
        self,
        faces: list[FaceTrack],
        association: AssociationResult,
        now: float,
    ) -> None:
        """Keep an identity alive from a matching face even without its body.

        This covers a person walking out from behind an occluder with their
        body detector still suppressed.  It may raise confidence, but it can
        never produce a CONFIRMED state (that needs a body).
        """
        orphan_ids = set(association.unmatched_faces)
        if not orphan_ids:
            return
        for ident in self.identities.values():
            if ident.body_track_id is not None or ident.descriptor is None:
                continue
            if now - ident.last_face_time > self.config.freshness_window:
                continue
            for face in faces:
                if face.track_id not in orphan_ids or face.descriptor is None:
                    continue
                sim = cosine_similarity(face.descriptor, ident.descriptor)
                if sim >= self.association_config.reid_similarity:
                    ident.confidence = max(ident.confidence, sim * self.config.min_evidence_score)
                    ident.last_face_time = now
                    ident.face_track_id = face.track_id
                    self._log(now, ident.identity_id, "face_only", f"sim={sim:.2f}")
                    break

    def _rebind(
        self,
        ident: Identity,
        body_id: int,
        match: FaceBodyMatch,
        face: FaceTrack | None,
        now: float,
    ) -> None:
        ident.body_track_id = body_id
        ident.body_lost_time = -1.0
        self._log(now, ident.identity_id, "reidentified", f"body={body_id}")
        self._apply_evidence(ident, match, face, now, reacquired=True)

    def _create(
        self,
        body_id: int,
        match: FaceBodyMatch,
        face: FaceTrack | None,
        now: float,
    ) -> Identity:
        ident = Identity(
            identity_id=self._next_id,
            body_track_id=body_id,
            state=IdentityState.UNKNOWN,
            confidence=0.0,
            last_update=now,
            descriptor=face.descriptor.copy() if face is not None and face.descriptor is not None else None,
        )
        self._next_id += 1
        self.identities[ident.identity_id] = ident
        self._apply_evidence(ident, match, face, now, reacquired=False)
        self._log(now, ident.identity_id, "created", f"body={body_id}")
        return ident

    # ------------------------------------------------------------------ state
    def _refresh_state(
        self,
        ident: Identity,
        now: float,
        body_by_id: dict[int, BodyTrack],
    ) -> None:
        cfg = self.config
        has_body = ident.body_track_id in body_by_id
        since_face = now - ident.last_face_time if ident.last_face_time >= 0 else float("inf")
        since_body = now - ident.body_lost_time if ident.body_lost_time >= 0 else 0.0

        new_state = state_for(
            confidence=ident.confidence,
            has_live_body=has_body,
            seconds_since_face=since_face,
            seconds_since_body=since_body,
            confirm_threshold=cfg.confirm_threshold,
            unknown_threshold=cfg.unknown_threshold,
            freshness_window=cfg.freshness_window,
            body_grace_seconds=cfg.body_grace_seconds,
            enough_observations=ident.face_observations >= cfg.min_confirmations,
        )
        if new_state is not ident.state:
            ident.state_changes.append((now, new_state))
            self._log(now, ident.identity_id, "state", new_state.value)
            ident.state = new_state

    def _maybe_auto_lock(self, now: float) -> None:
        if not self.auto_lock:
            return
        confirmed = [
            i for i in self.identities.values() if i.state is IdentityState.CONFIRMED
        ]
        if not confirmed:
            return
        current = self.target
        # A person we can name outranks a stranger we merely saw a face on.
        named = [i for i in confirmed if i.name is not None]
        # Once an enrollment registry exists, never auto-lock an unidentified
        # visitor. Wait until ArcFace recognizes a registered person.
        if self.enrollment is not None and len(self.enrollment) > 0 and not named:
            return
        # If an unknown person was auto-locked during camera warm-up, switch
        # to the enrolled person as soon as their face is recognized. Keep a
        # named target stable once selected.
        if current is not None and current.name is not None:
            return
        if not named and current is not None:
            return
        pool = named or confirmed
        best = max(pool, key=lambda i: i.confidence)
        if current is not None and current.identity_id == best.identity_id:
            return
        self.target_identity_id = best.identity_id
        detail = f"{best.name} (recognized)" if best.name else "first confirmed person"
        self._log(now, best.identity_id, "auto_lock", detail)

    # ------------------------------------------------------------- enrollment
    def enroll_target(self, name: str, now: float | None = None) -> str | None:
        """Teach the book who the locked target is.

        Returns the enrolled name, or ``None`` when there is no target or no
        usable face signature yet (nothing is recorded in that case, so a
        premature enroll cannot corrupt the record).
        """
        if self.enrollment is None:
            return None
        ident = self.target
        if ident is None or ident.descriptor is None:
            return None
        stamp = self._last_now if now is None else now
        if not self.enrollment.enroll(name, ident.descriptor, now=stamp):
            return None
        ident.name = (name or "").strip() or None
        self.last_enrollment = ident.name
        self._log(stamp, ident.identity_id, "enrolled", ident.name or "")
        return ident.name

    def _retire(self, now: float) -> None:
        cfg = self.config
        dead: list[int] = []
        for ident in self.identities.values():
            if ident.state is not IdentityState.UNKNOWN:
                continue
            if ident.body_track_id is not None:
                continue
            missing_for = now - max(ident.last_face_time, ident.last_update)
            if ident.confidence <= cfg.retire_threshold or (
                ident.body_lost_time >= 0 and now - ident.body_lost_time >= cfg.retire_after
            ):
                dead.append(ident.identity_id)
        for ident_id in dead:
            self.identities.pop(ident_id, None)
            self._log(now, ident_id, "retired")
            if self.target_identity_id == ident_id:
                self.target_identity_id = None

    # ------------------------------------------------------------------ claim
    def claim(self, now: float | None = None) -> IdentityClaim:
        """Return the honest, current identity claim for the locked target."""
        ident = self.target
        if ident is None:
            return IdentityClaim(
                identity_id=None,
                state=IdentityState.UNKNOWN,
                confidence=0.0,
                reason="no target locked",
            )

        reason = describe(ident.state, ident.confidence)
        if ident.state is IdentityState.TEMPORARY and ident.face_track_id is None:
            reason = (
                f"face not visible for {ident.seconds_since_face:.1f}s, "
                f"confidence decaying ({ident.confidence * 100:.0f}%)"
            )
        elif ident.state is IdentityState.UNKNOWN:
            reason = "confidence floor reached - no identity claim"
        if ident.name:
            reason = f"{ident.name}: {reason}"

        return IdentityClaim(
            identity_id=ident.identity_id,
            state=ident.state,
            confidence=ident.confidence,
            name=ident.name,
            body_track_id=ident.body_track_id,
            face_track_id=ident.face_track_id,
            reason=reason,
        )
