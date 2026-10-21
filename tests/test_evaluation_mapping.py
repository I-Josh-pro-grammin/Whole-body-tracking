"""Tests for the diagnostic per-frame body -> ground-truth mapping."""

from __future__ import annotations

from dataclasses import dataclass

from bodyfollowing.evaluation.metrics import map_bodies_to_ground_truth
from bodyfollowing.types import BodyTrack, Box


@dataclass
class Actor:
    body_box: Box


def body(track_id: int, cx: float) -> BodyTrack:
    return BodyTrack(track_id=track_id, box=Box.from_cxcyw(cx, 300, 100, 250))


def actor(cx: float) -> Actor:
    return Actor(body_box=Box.from_cxcyw(cx, 300, 100, 250))


def test_maps_each_track_to_the_overlapping_actor():
    mapping = map_bodies_to_ground_truth(
        [body(10, 200), body(11, 600)], {1: actor(202), 2: actor(598)}
    )
    assert mapping == {10: 1, 11: 2}


def test_non_overlapping_boxes_are_not_mapped():
    mapping = map_bodies_to_ground_truth([body(10, 100)], {1: actor(600)})
    assert mapping == {}


def test_threshold_is_respected():
    # Shifted by 40 px on a 100 px wide box -> IoU ~ 0.43.
    loose = map_bodies_to_ground_truth([body(10, 100)], {1: actor(140)}, iou_threshold=0.2)
    strict = map_bodies_to_ground_truth([body(10, 100)], {1: actor(140)}, iou_threshold=0.6)
    assert loose == {10: 1}
    assert strict == {}
