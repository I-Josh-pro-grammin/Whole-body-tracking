"""Tests for the stable track->ground-truth labeler.

The labeler exists so that identity is graded against a *fixed* ground-truth
person per tracker id.  Grading against a per-frame IoU match would flip while
two people overlap and blame the tracker for a measurement artefact.
"""

from __future__ import annotations

from dataclasses import dataclass

from bodyfollowing.evaluation.metrics import StableTrackLabeler
from bodyfollowing.types import BodyTrack, Box


@dataclass
class FakeActor:
    body_box: Box


def body(track_id: int, cx: float) -> BodyTrack:
    return BodyTrack(track_id=track_id, box=Box.from_cxcyw(cx, 300, 100, 250))


def truth(positions: dict[int, float]) -> dict[int, FakeActor]:
    return {
        gt: FakeActor(body_box=Box.from_cxcyw(cx, 300, 100, 250))
        for gt, cx in positions.items()
    }


def test_label_is_assigned_once_and_kept():
    labeler = StableTrackLabeler()
    labeler.update([body(1, 200)], truth({1: 200, 2: 600}))
    assert labeler.label_for(1) == 1

    # Later the track drifts next to the other person; the label must stick.
    labeler.update([body(1, 600)], truth({1: 600, 2: 200}))
    assert labeler.label_for(1) == 1


def test_ambiguous_birth_is_deferred():
    labeler = StableTrackLabeler(margin=0.05)
    labeler.update([body(1, 300)], truth({1: 300, 2: 302}))
    assert labeler.label_for(1) is None
    assert 1 in labeler.unresolved

    # Once the two people separate, the label resolves.
    labeler.update([body(1, 300)], truth({1: 300, 2: 460}))
    assert labeler.label_for(1) == 1
    assert 1 not in labeler.unresolved


def test_low_overlap_is_not_labelled():
    labeler = StableTrackLabeler(iou_threshold=0.5)
    labeler.update([body(1, 100)], truth({1: 600}))
    assert labeler.label_for(1) is None


def test_unknown_track_has_no_label():
    labeler = StableTrackLabeler()
    assert labeler.label_for(None) is None
    assert labeler.label_for(99) is None


def test_two_tracks_get_two_labels():
    labeler = StableTrackLabeler()
    labeler.update([body(1, 200), body(2, 600)], truth({1: 200, 2: 600}))
    assert labeler.label_for(1) == 1
    assert labeler.label_for(2) == 2
