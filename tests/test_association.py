"""Tests for spatial face-to-body mapping."""

from __future__ import annotations

import pytest

from bodyfollowing.association.face_body import (
    associate,
    contamination_for,
    pair_score,
)
from bodyfollowing.config import AssociationConfig
from bodyfollowing.types import BodyTrack, Box, FaceTrack

CFG = AssociationConfig()


def body(track_id: int, x: float = 300.0, y: float = 300.0, h: float = 250.0, w: float = 100.0):
    return BodyTrack(track_id=track_id, box=Box.from_cxcyw(x, y, w, h), confirmed=True)


def face_in_head(body_track: BodyTrack, track_id: int = 1, scale: float = 1.0, dx: float = 0.0):
    """A face plausibly belonging to ``body_track``'s head."""
    fh = 0.17 * body_track.box.height * scale
    fw = 0.78 * fh
    cy = body_track.box.y1 + 0.075 * body_track.box.height
    return FaceTrack(
        track_id=track_id,
        box=Box.from_cxcyw(body_track.box.cx + dx, cy, fw, fh),
        score=0.95,
    )


# --------------------------------------------------------------------- pair_score
def test_face_in_head_band_scores_high():
    b = body(1)
    f = face_in_head(b)
    score, parts = pair_score(b.box, f.box, CFG)
    assert score > 0.85
    assert parts["vertical"] == pytest.approx(1.0)
    assert parts["overlap"] > 0.9


def test_face_at_feet_scores_low():
    b = body(1)
    fh = 0.17 * b.box.height
    feet = FaceTrack(
        track_id=1,
        box=Box.from_cxcyw(b.box.cx, b.box.y2 - fh, 0.78 * fh, fh),
    )
    score, _ = pair_score(b.box, feet.box, CFG)
    assert score < 0.35


def test_face_far_to_the_side_scores_low():
    b = body(1)
    f = face_in_head(b, dx=1.9 * b.box.width)
    score, parts = pair_score(b.box, f.box, CFG)
    assert parts["horizontal"] < 0.2
    assert score < 0.45


def test_scale_mismatch_is_penalised():
    b = body(1)
    normal, _ = pair_score(b.box, face_in_head(b).box, CFG)
    huge, _ = pair_score(b.box, face_in_head(b, scale=4.0).box, CFG)
    tiny, _ = pair_score(b.box, face_in_head(b, scale=0.25).box, CFG)
    assert huge < normal and tiny < normal


def test_face_outside_body_is_killed():
    b = body(1, x=100)
    far = face_in_head(body(2, x=600))
    score, _ = pair_score(b.box, far.box, CFG)
    assert score < 0.2


# ----------------------------------------------------------------------- associate
def test_associate_assigns_each_face_to_its_own_body():
    b1, b2 = body(1, x=150), body(2, x=500)
    f1, f2 = face_in_head(b1, 11), face_in_head(b2, 22)
    result = associate([b1, b2], [f1, f2], CFG)
    pairs = {(m.face_id, m.body_id) for m in result.matches}
    assert pairs == {(11, 1), (22, 2)}
    assert result.unmatched_bodies == []
    assert result.unmatched_faces == []


def test_associate_is_one_to_one_even_with_duplicate_faces():
    b1 = body(1)
    f1, f2 = face_in_head(b1, 11), face_in_head(b1, 22)
    result = associate([b1], [f1, f2], CFG)
    assert len(result.matches) == 1
    assert len(result.unmatched_faces) == 1


def test_associate_without_faces_reports_unmatched_bodies():
    result = associate([body(1)], [], CFG)
    assert result.matches == []
    assert result.unmatched_bodies == [1]


def test_contamination_detects_an_occluding_body():
    b1 = body(1, x=200)
    f = face_in_head(b1)
    occluder = Box.from_cxcyw(f.box.cx, f.box.cy, f.box.width * 1.4, f.box.height * 4)
    # Face fully inside the other body -> contamination 1.0.
    assert contamination_for(f.box, [b1.box, occluder], 0) == pytest.approx(1.0)
    assert contamination_for(f.box, [b1.box], 0) == 0.0


def test_occluded_match_is_not_trustworthy():
    b1 = body(1, x=200)
    f = face_in_head(b1)
    occluder = body(2, x=f.box.cx, y=f.box.cy + 60, h=300, w=140)
    result = associate([b1, occluder], [f], CFG)
    match = result.for_face(f.track_id)
    assert match is not None
    assert match.contamination > 0.5
    assert not match.is_trustworthy


def test_ambiguous_crossing_is_flagged():
    """Two bodies giving a face nearly the same score must be flagged."""
    b1 = body(1, x=300, h=250)
    b2 = body(2, x=306, h=250)
    f = face_in_head(b1, 11)
    result = associate([b1, b2], [f], CFG)
    match = result.for_face(11)
    assert match is not None and match.ambiguous
    assert not match.is_trustworthy


def test_stale_face_tracks_are_not_evidence():
    b1 = body(1)
    observed = face_in_head(b1, 11)
    stale = face_in_head(b1, 22)
    stale.time_since_update = 3
    result = associate([b1], [observed, stale], CFG)
    assert [m.face_id for m in result.matches] == [11]
