"""Ground-truth evaluation of identity continuity.

The evaluator drives the *real* pipeline over a simulated scene and scores
what the module claimed against what was actually true.  Three numbers matter
most:

``confirmed_precision``
    Of the frames where the module said CONFIRMED, how many were the right
    person?  A dishonest model can score 100% continuity here only by never
    confirming anything -- which the coverage metrics then expose.

``coverage``
    Fraction of frames where the module made a *correct* positive claim while
    the target was actually visible.  This is the cost side of honesty.

``recovery_time``
    Seconds from the face becoming visible again to CONFIRMED being restored.
    This is the actual product requirement: identity continuity across
    temporary facial loss.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

import numpy as np

from ..config import PipelineConfig
from ..pipeline import BodyFollowingPipeline
from ..types import BodyTrack

STATES = ("confirmed", "temporary", "unknown")


class SceneLike(Protocol):
    """What the evaluator needs from a scene (simulated or composited)."""

    name: str
    fps: float
    width: int
    height: int

    def frames(self):  # pragma: no cover - protocol declaration
        ...


# --------------------------------------------------------------------------- record
@dataclass
class FrameRecord:
    index: int
    timestamp: float
    state: str
    confidence: float
    identity_id: int | None
    body_track_id: int | None
    claimed_gt: int | None
    correct: bool | None
    target_body_visible: bool
    target_face_visible: bool
    centring_active: bool
    error_x: float
    error_y: float


@dataclass
class ScenarioResult:
    name: str
    target_gt_id: int
    records: list[FrameRecord] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    identity_ids: list[int] = field(default_factory=list)

    # ------------------------------------------------------------------ access
    def states(self) -> list[str]:
        return [r.state for r in self.records]

    def timeline(self) -> list[dict[str, Any]]:
        """Compress the per-frame states into labelled segments."""
        if not self.records:
            return []
        segments: list[dict[str, Any]] = []
        current = self.records[0]
        start = current.timestamp
        confs = [current.confidence]
        for rec in self.records[1:]:
            if rec.state != current.state:
                segments.append(
                    {
                        "state": current.state,
                        "start": round(start, 2),
                        "end": round(rec.timestamp, 2),
                        "confidence": round(float(np.mean(confs)), 3),
                    }
                )
                current, start, confs = rec, rec.timestamp, [rec.confidence]
            else:
                confs.append(rec.confidence)
        segments.append(
            {
                "state": current.state,
                "start": round(start, 2),
                "end": round(self.records[-1].timestamp, 2),
                "confidence": round(float(np.mean(confs)), 3),
            }
        )
        return segments


# --------------------------------------------------------------------------- helpers
def map_bodies_to_ground_truth(
    bodies: list[BodyTrack],
    truth: dict,
    iou_threshold: float = 0.35,
) -> dict[int, int]:
    """Map each body track id to the ground-truth id it overlaps most."""
    mapping: dict[int, int] = {}
    for body in bodies:
        best_id, best_iou = None, iou_threshold
        for gt_id, actor in truth.items():
            iou = body.box.iou(actor.body_box)
            if iou >= best_iou:
                best_iou, best_id = iou, gt_id
        if best_id is not None:
            mapping[body.track_id] = best_id
    return mapping


class StableTrackLabeler:
    """Assigns each tracker id a ground-truth person *once*, then keeps it.

    Grading identity with a per-frame IoU match is unsound: while two people
    are on top of each other their ground-truth boxes are interchangeable, so
    the *metric* flips even when the tracker is doing the right thing.  A
    tracker id is therefore labelled at a moment when the answer is
    unambiguous, and that label sticks for the track's whole life.  If a track
    is born mid-crossing the label is deferred until the ambiguity clears.
    """

    def __init__(self, iou_threshold: float = 0.5, margin: float = 0.05) -> None:
        self.iou_threshold = iou_threshold
        self.margin = margin
        self.labels: dict[int, int] = {}
        self.unresolved: set[int] = set()

    def update(self, bodies: list[BodyTrack], truth: dict) -> None:
        for body in bodies:
            if body.track_id in self.labels:
                continue
            scored = sorted(
                ((body.box.iou(actor.body_box), gt_id) for gt_id, actor in truth.items()),
                reverse=True,
            )
            if not scored or scored[0][0] < self.iou_threshold:
                continue
            if len(scored) > 1 and (scored[0][0] - scored[1][0]) < self.margin:
                self.unresolved.add(body.track_id)
                continue
            self.labels[body.track_id] = scored[0][1]
            self.unresolved.discard(body.track_id)

    def label_for(self, track_id: int | None) -> int | None:
        if track_id is None:
            return None
        return self.labels.get(track_id)


def _recovery_times(records: list[FrameRecord]) -> tuple[list[float], int]:
    """Seconds from face re-appearing to the next CONFIRMED frame.

    Only counts genuine *re*-acquisitions: an episode must follow a period in
    which the identity had already been confirmed at least once.  The initial
    acquisition (after the user selects a target) is not a recovery and is
    reported separately as ``time_to_first_confirm_s``.
    """
    recoveries: list[float] = []
    episodes = 0
    pending_since: float | None = None
    was_visible = False
    ever_confirmed = False
    for rec in records:
        appeared = rec.target_face_visible and not was_visible
        if appeared and ever_confirmed:
            pending_since = rec.timestamp
            episodes += 1
        if rec.state == "confirmed":
            if pending_since is not None:
                recoveries.append(rec.timestamp - pending_since)
                pending_since = None
            ever_confirmed = True
        if not rec.target_face_visible:
            pending_since = None
        was_visible = rec.target_face_visible
    return recoveries, episodes


def _confirmed_hold_times(records: list[FrameRecord]) -> list[float]:
    """Seconds a CONFIRMED claim survived after the face disappeared.

    Quantifies the honesty cost of temporal smoothing: how long the module
    keeps asserting an identity once the supporting face evidence is gone.
    """
    holds: list[float] = []
    lost_at: float | None = None
    was_visible = True
    for rec in records:
        if was_visible and not rec.target_face_visible:
            lost_at = rec.timestamp
        if lost_at is not None and rec.state != "confirmed":
            holds.append(rec.timestamp - lost_at)
            lost_at = None
        was_visible = rec.target_face_visible
    return holds


def _count_switches(records: list[FrameRecord], target_gt_id: int) -> int:
    """How many times the claimed ground-truth identity changed."""
    switches = 0
    previous: int | None = None
    for rec in records:
        if rec.claimed_gt is None:
            continue
        if previous is not None and rec.claimed_gt != previous:
            switches += 1
        previous = rec.claimed_gt
    return switches


# --------------------------------------------------------------------------- driver
def evaluate_scene(
    scene,
    config: PipelineConfig | None = None,
    target_gt_id: int = 1,
    lock_time: float = 1.0,
    auto_lock: bool = False,
    detector=None,
) -> ScenarioResult:
    """Run the pipeline over a scene and score it against ground truth.

    Works for both scene kinds:

    * simulated scenes supply ``detections`` directly (no pixels needed);
    * composited scenes supply real frames and ``detections=None``, so the
      configured ``detector`` runs on real pixels.

    The target is selected the way a user would: by clicking on the person at
    ``lock_time``.  ``auto_lock`` is off by default so the evaluation measures
    continuity of a *chosen* target rather than whichever person the module
    happened to like first.
    """
    cfg = copy.deepcopy(config) if config is not None else PipelineConfig()
    cfg.auto_lock = auto_lock

    pipeline = BodyFollowingPipeline(cfg, detector=detector)
    result = ScenarioResult(name=scene.name, target_gt_id=target_gt_id)
    seen_identity_ids: set[int] = set()
    labeler = StableTrackLabeler()
    locked = False
    locked_at: float | None = None

    for sim in scene.frames():
        detections = getattr(sim, "detections", None)
        if detections is None:
            frame_result = pipeline.step(sim.frame_array(), now=sim.timestamp)
        else:
            frame_result = pipeline.step(
                sim.frame_array(), now=sim.timestamp, detections=detections
            )
        truth = sim.ground_truth
        labeler.update(frame_result.bodies, truth)  # noqa: E501

        # "User click" on the target body once the module has an identity for it.
        if not locked and sim.timestamp >= lock_time:
            target_truth = truth.get(target_gt_id)
            if target_truth is not None and target_truth.body_visible:
                ident = pipeline.identity.lock_at_point(
                    target_truth.body_box.cx, target_truth.body_box.cy, frame_result.bodies
                )
                if ident is not None:
                    locked = True
                    locked_at = sim.timestamp

        claim = frame_result.claim
        if claim.identity_id is not None:
            seen_identity_ids.add(claim.identity_id)

        claimed_gt = labeler.label_for(claim.body_track_id)
        correct = None if claimed_gt is None else (claimed_gt == target_gt_id)
        target_truth = truth.get(target_gt_id)

        result.records.append(
            FrameRecord(
                index=sim.index,
                timestamp=sim.timestamp,
                state=claim.state.value,
                confidence=claim.confidence,
                identity_id=claim.identity_id,
                body_track_id=claim.body_track_id,
                claimed_gt=claimed_gt,
                correct=correct,
                target_body_visible=bool(target_truth.body_visible),
                target_face_visible=bool(target_truth.face_visible),
                centring_active=frame_result.command.active,
                error_x=frame_result.command.error_x,
                error_y=frame_result.command.error_y,
            )
        )

    result.identity_ids = sorted(seen_identity_ids)
    result.metrics = _compute_metrics(result, scene, pipeline, locked, locked_at)
    return result


def _compute_metrics(
    result: ScenarioResult,
    scene: SceneLike,
    pipeline: BodyFollowingPipeline,
    locked: bool,
    locked_at: float | None,
) -> dict[str, Any]:
    records = result.records
    n = len(records)
    if n == 0:
        return {"frames": 0}

    state_counts = {s: 0 for s in STATES}
    for rec in records:
        state_counts[rec.state] = state_counts.get(rec.state, 0) + 1

    confirmed = [r for r in records if r.state == "confirmed"]
    confirmed_correct = sum(1 for r in confirmed if r.correct is True)
    confirmed_wrong = sum(1 for r in confirmed if r.correct is False)
    positive = confirmed_correct + confirmed_wrong

    claim_frames = [r for r in records if r.claimed_gt is not None]
    correct_claims = sum(1 for r in claim_frames if r.correct)
    wrong_claims = sum(1 for r in claim_frames if r.correct is False)

    face_visible_records = [r for r in records if r.target_face_visible]
    unknown_while_visible = sum(
        1 for r in face_visible_records if r.state == "unknown"
    )

    recoveries, face_episodes = _recovery_times(records)
    holds = _confirmed_hold_times(records)
    first_confirm = next(
        (r.timestamp for r in records if r.state == "confirmed"), None
    )

    # Post-lock statistics: the pre-lock warm-up is not a tracking failure.
    post_lock = [r for r in records if locked_at is None or r.timestamp >= locked_at]
    post_counts = {s: 0 for s in STATES}
    for rec in post_lock:
        post_counts[rec.state] = post_counts.get(rec.state, 0) + 1
    post_n = max(len(post_lock), 1)

    visible_body_frames = sum(1 for r in records if r.target_body_visible)
    active = [r for r in records if r.centring_active]
    deadzone_w, deadzone_h = (
        pipeline.config.centring.deadzone_x,
        pipeline.config.centring.deadzone_y,
    )

    # Temporary->Confirmed episode count (identity carried through a gap).
    temporary_episodes = 0
    previous = None
    for rec in records:
        if rec.state == "temporary" and previous == "confirmed":
            temporary_episodes += 1
        previous = rec.state

    durations = {
        s: sum(1 for r in records if r.state == s) / max(scene.fps, 1e-9) for s in STATES
    }

    return {
        "frames": n,
        "duration_s": round(n / scene.fps, 2),
        "locked": locked,
        "locked_at": round(locked_at, 2) if locked_at is not None else None,
        "identities_claimed": len(result.identity_ids),
        "identities_created": pipeline.identity._next_id - 1,
        "state_frames": state_counts,
        "state_duration_s": {k: round(v, 2) for k, v in durations.items()},
        "state_fraction": {k: round(v / n, 3) for k, v in state_counts.items()},
        "confirmed_frames": len(confirmed),
        "confirmed_correct": confirmed_correct,
        "confirmed_wrong": confirmed_wrong,
        "confirmed_precision": round(confirmed_correct / positive, 3) if positive else None,
        "wrong_person_frames": wrong_claims,
        "correct_person_frames": correct_claims,
        "claim_accuracy": round(correct_claims / len(claim_frames), 3) if claim_frames else None,
        "identity_switches": _count_switches(records, result.target_gt_id),
        "temporary_episodes": temporary_episodes,
        "face_visible_frames": len(face_visible_records),
        "unknown_while_face_visible": unknown_while_visible,
        "unknown_while_face_visible_rate": (
            round(unknown_while_visible / len(face_visible_records), 3)
            if face_visible_records
            else 0.0
        ),
        "recovery_times_s": [round(x, 2) for x in recoveries],
        "recovery_mean_s": round(float(np.mean(recoveries)), 2) if recoveries else None,
        "face_episodes": face_episodes,
        "time_to_first_confirm_s": round(first_confirm, 2) if first_confirm else None,
        "confirmed_hold_after_face_loss_s": [round(h, 2) for h in holds],
        "confirmed_hold_mean_s": round(float(np.mean(holds)), 2) if holds else None,
        "state_fraction_post_lock": {
            k: round(v / post_n, 3) for k, v in post_counts.items()
        },
        "centring_active_rate": round(len(active) / n, 3),
        "centring_error_mean_px": (
            round(float(np.mean([abs(r.error_x) for r in active])), 1) if active else None
        ),
        "centring_in_deadzone_rate": (
            round(
                sum(
                    1
                    for r in active
                    if abs(r.error_x) / (scene.width / 2.0) <= deadzone_w
                    and abs(r.error_y) / (scene.height / 2.0) <= deadzone_h
                )
                / len(active),
                3,
            )
            if active
            else None
        ),
        "body_visible_frames": visible_body_frames,
    }


# --------------------------------------------------------------------------- checks
@dataclass
class Expectation:
    """What a scenario is expected to demonstrate."""

    description: str = ""
    require_confirmed: bool = True
    require_temporary_during_face_loss: bool = True
    require_recovery: bool = True
    require_unknown_reached: bool = False
    max_wrong_person_frames: int = 0
    max_identity_switches: int = 0
    max_identities_created: int = 1
    min_confirmed_precision: float = 1.0
    min_centring_active_rate: float = 0.8


@dataclass
class Check:
    name: str
    ok: bool
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def check_expectation(result: ScenarioResult, exp: Expectation) -> list[Check]:
    m = result.metrics
    checks: list[Check] = []
    if m.get("frames", 0) == 0:
        return [Check("frames", False, "scenario produced no frames")]

    checks.append(
        Check(
            "confirmed observed",
            (m["confirmed_frames"] > 0) if exp.require_confirmed else True,
            f"{m['confirmed_frames']} confirmed frames",
        )
    )
    checks.append(
        Check(
            "temporary during face loss",
            (m["state_frames"].get("temporary", 0) > 0)
            if exp.require_temporary_during_face_loss
            else True,
            f"{m['state_frames'].get('temporary', 0)} temporary frames",
        )
    )
    checks.append(
        Check(
            "recovered after face loss",
            (m["recovery_mean_s"] is not None) if exp.require_recovery else True,
            f"recovery_mean={m['recovery_mean_s']}s over {m['face_episodes']} episodes",
        )
    )
    if exp.require_unknown_reached:
        checks.append(
            Check(
                "unknown reached",
                m["state_frames"].get("unknown", 0) > 0,
                f"{m['state_frames'].get('unknown', 0)} unknown frames",
            )
        )
    if exp.min_confirmed_precision is not None and m["confirmed_precision"] is not None:
        checks.append(
            Check(
                "confirmed precision",
                m["confirmed_precision"] >= exp.min_confirmed_precision,
                f"{m['confirmed_precision']:.2%} of confirmations correct",
            )
        )
    checks.append(
        Check(
            "no wrong-person claims",
            m["wrong_person_frames"] <= exp.max_wrong_person_frames,
            f"{m['wrong_person_frames']} frames claimed a different person",
        )
    )
    checks.append(
        Check(
            "identity continuity",
            m["identity_switches"] <= exp.max_identity_switches,
            f"{m['identity_switches']} identity switches",
        )
    )
    checks.append(
        Check(
            "single identity",
            m["identities_claimed"] <= exp.max_identities_created,
            f"{m['identities_claimed']} identity id(s) claimed for the target "
            f"({m['identities_created']} people tracked in total)",
        )
    )
    checks.append(
        Check(
            "centring active",
            m["centring_active_rate"] >= exp.min_centring_active_rate,
            f"active in {m['centring_active_rate']:.0%} of frames",
        )
    )
    return checks


EXPECTATIONS: dict[str, Expectation] = {
    "head_turn": Expectation(
        description="Face turns away for 2.2 s; body never leaves the frame.",
        require_temporary_during_face_loss=True,
        require_recovery=True,
        max_identity_switches=0,
        max_identities_created=1,
    ),
    "occlusion": Expectation(
        description="A passer-by crosses in front and hides the target's face.",
        require_temporary_during_face_loss=True,
        require_recovery=True,
        max_wrong_person_frames=0,
        max_identity_switches=0,
        max_identities_created=1,
    ),
    "path_crossing": Expectation(
        description="Two people cross at the same depth; identity must not swap.",
        require_confirmed=True,
        require_temporary_during_face_loss=False,
        require_recovery=False,
        max_wrong_person_frames=0,
        max_identity_switches=0,
        max_identities_created=1,
    ),
    "long_absence": Expectation(
        description="Face hidden 6 s; confidence must decay to UNKNOWN and recover.",
        require_unknown_reached=True,
        require_recovery=True,
        max_identity_switches=0,
        max_identities_created=1,
        # Centring is *deliberately* inactive while the identity is UNKNOWN,
        # so a long absence necessarily lowers the active rate.
        min_centring_active_rate=0.35,
    ),
    "combined": Expectation(
        description="Head turn + occlusion + a crossing person, all at once.",
        require_temporary_during_face_loss=True,
        require_recovery=True,
        max_wrong_person_frames=0,
        max_identity_switches=0,
        max_identities_created=1,
        min_centring_active_rate=0.6,
    ),
}
