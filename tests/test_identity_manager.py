"""Tests for the identity ledger: evidence, decay, re-ID and honesty guards."""

from __future__ import annotations

import numpy as np
import pytest

from bodyfollowing.association.face_body import AssociationResult, FaceBodyMatch
from bodyfollowing.config import AssociationConfig, IdentityConfig
from bodyfollowing.identity.manager import IdentityManager
from bodyfollowing.types import BodyTrack, Box, FaceTrack, IdentityState


def make_body(track_id: int = 1, x: float = 300.0) -> BodyTrack:
    return BodyTrack(track_id=track_id, box=Box.from_cxcyw(x, 300, 100, 250), confirmed=True)


def make_face(track_id: int, body: BodyTrack, descriptor=None, tsu: int = 0) -> FaceTrack:
    fh = 0.17 * body.box.height
    fw = 0.78 * fh
    return FaceTrack(
        track_id=track_id,
        box=Box.from_cxcyw(body.box.cx, body.box.y1 + 0.075 * body.box.height, fw, fh),
        score=0.95,
        time_since_update=tsu,
        descriptor=descriptor,
    )


def make_assoc(face: FaceTrack | None, body: BodyTrack, score: float = 0.9, **kw):
    if face is None:
        return AssociationResult()
    return AssociationResult(
        matches=[
            FaceBodyMatch(face_id=face.track_id, body_id=body.track_id, score=score, **kw)
        ]
    )


def descriptor(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vec = rng.normal(size=40)
    return vec / np.linalg.norm(vec)


@pytest.fixture
def manager() -> IdentityManager:
    return IdentityManager(IdentityConfig(), AssociationConfig(), auto_lock=False)


def test_face_evidence_confirms_identity(manager):
    body = make_body()
    face = make_face(11, body, descriptor(1))
    identities = manager.update([body], [face], make_assoc(face, body), now=0.0)
    assert len(identities) == 1
    ident = identities[0]
    assert ident.state is IdentityState.CONFIRMED
    assert ident.confidence == pytest.approx(0.9, abs=1e-6)
    assert ident.body_track_id == body.track_id


def test_confidence_decays_exponentially_without_face(manager):
    body = make_body()
    face = make_face(11, body, descriptor(1))
    manager.update([body], [face], make_assoc(face, body), now=0.0)
    start = manager.identities[1].confidence

    manager.update([body], [], AssociationResult(), now=1.0)
    expected = start * np.exp(-1.0 / manager.config.decay_tau)
    assert manager.identities[1].confidence == pytest.approx(expected, rel=1e-6)
    assert manager.identities[1].state is not IdentityState.CONFIRMED


def test_state_walks_confirmed_temporary_unknown(manager):
    body = make_body()
    face = make_face(11, body, descriptor(1))
    manager.update([body], [face], make_assoc(face, body), now=0.0)
    seen = [manager.identities[1].state]

    for now in np.arange(0.1, 12.0, 0.1):
        manager.update([body], [], AssociationResult(), now=float(now))
        state = manager.identities[1].state
        if state is not seen[-1]:
            seen.append(state)
    assert seen == [
        IdentityState.CONFIRMED,
        IdentityState.TEMPORARY,
        IdentityState.UNKNOWN,
    ]


def test_face_return_restores_confirmed_with_same_identity(manager):
    body = make_body()
    face = make_face(11, body, descriptor(1))
    manager.update([body], [face], make_assoc(face, body), now=0.0)
    identity_id = manager.identities[1].identity_id

    for now in np.arange(0.1, 4.0, 0.1):
        manager.update([body], [], AssociationResult(), now=float(now))
    assert manager.identities[identity_id].state is not IdentityState.CONFIRMED

    face2 = make_face(12, body, descriptor(1))
    manager.update([body], [face2], make_assoc(face2, body), now=4.0)
    ident = manager.identities[identity_id]
    assert ident.state is IdentityState.CONFIRMED
    assert ident.identity_id == identity_id
    assert len(manager.identities) == 1


def test_ambiguous_match_is_not_evidence(manager):
    body = make_body()
    face = make_face(11, body, descriptor(1))
    assoc = make_assoc(face, body, ambiguous=True)
    manager.update([body], [face], assoc, now=0.0)
    assert manager.identities == {}


def test_contaminated_match_is_not_evidence(manager):
    body = make_body()
    face = make_face(11, body, descriptor(1))
    assoc = make_assoc(face, body, contamination=0.8)
    manager.update([body], [face], assoc, now=0.0)
    assert manager.identities == {}


def test_weak_match_is_not_evidence(manager):
    body = make_body()
    face = make_face(11, body, descriptor(1))
    assoc = make_assoc(face, body, score=0.3)
    manager.update([body], [face], assoc, now=0.0)
    assert manager.identities == {}


def test_appearance_conflict_breaks_a_swapped_body(manager):
    """A sudden descriptor change means the tracker swapped people."""
    body = make_body()
    desc_a = descriptor(1)
    face = make_face(11, body, desc_a)
    for step in range(5):
        face.track_id = 11
        manager.update([body], [face], make_assoc(face, body), now=float(step))
    assert manager.identities[1].body_track_id == body.track_id

    # Same body id, completely different appearance -> refuse the link.
    face_b = make_face(12, body, descriptor(2))
    manager.update([body], [face_b], make_assoc(face_b, body), now=5.0)
    ident = manager.identities[1]
    assert ident.body_track_id is None
    assert ident.state is IdentityState.UNKNOWN


def test_reidentification_after_body_loss(manager):
    body = make_body(1)
    desc = descriptor(3)
    face = make_face(11, body, desc)
    manager.update([body], [face], make_assoc(face, body), now=0.0)
    identity_id = manager.identities[1].identity_id

    # Body disappears for a while (no detections at all).
    for now in np.arange(0.1, 3.0, 0.1):
        manager.update([], [], AssociationResult(), now=float(now))
    assert manager.identities[identity_id].body_track_id is None

    # A *new* body track appears elsewhere with the same face appearance.
    body2 = make_body(7, x=420)
    face2 = make_face(21, body2, desc)
    manager.update([body2], [face2], make_assoc(face2, body2), now=3.0)
    ident = manager.identities[identity_id]
    assert ident.body_track_id == 7
    assert ident.state is IdentityState.CONFIRMED
    assert len(manager.identities) == 1


def test_retirement_clears_the_target(manager):
    body = make_body()
    face = make_face(11, body, descriptor(1))
    manager.update([body], [face], make_assoc(face, body), now=0.0)
    manager.lock(1)
    assert manager.claim().identity_id == 1

    for now in np.arange(0.1, 20.0, 0.1):
        manager.update([], [], AssociationResult(), now=float(now))
    assert manager.identities == {}
    assert manager.claim().identity_id is None
    assert manager.claim().state is IdentityState.UNKNOWN


def test_claim_without_target_is_unknown(manager):
    claim = manager.claim()
    assert claim.identity_id is None
    assert claim.state is IdentityState.UNKNOWN
    assert not claim.asserts_identity


def test_lock_at_point_selects_the_body_under_the_cursor(manager):
    body = make_body(5, x=200)
    face = make_face(11, body, descriptor(1))
    manager.update([body], [face], make_assoc(face, body), now=0.0)
    ident = manager.lock_at_point(200, 300, [body])
    assert ident is not None and ident.identity_id == 1
    assert manager.target_identity_id == 1


def test_auto_lock_picks_first_confirmed_identity():
    mgr = IdentityManager(IdentityConfig(), AssociationConfig(), auto_lock=True)
    body = make_body()
    face = make_face(11, body, descriptor(1))
    mgr.update([body], [face], make_assoc(face, body), now=0.0)
    assert mgr.target_identity_id == 1
