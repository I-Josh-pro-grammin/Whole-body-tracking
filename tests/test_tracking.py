"""Tests for the Kalman filter and the trackers."""

from __future__ import annotations

import numpy as np
import pytest

from bodyfollowing.config import FaceTrackerConfig, TrackerConfig
from bodyfollowing.tracking.body_tracker import BodyTracker
from bodyfollowing.tracking.face_tracker import FaceTracker
from bodyfollowing.tracking.kalman import BoxKalman
from bodyfollowing.types import Box, Detection


def det(x: float, y: float = 300.0, w: float = 90.0, h: float = 240.0, score: float = 0.9):
    return Detection(box=Box.from_cxcyw(x, y, w, h), score=score)


# --------------------------------------------------------------------- Kalman
def test_kalman_predict_advances_with_velocity():
    kf = BoxKalman(Box.from_cxcyw(100, 300, 90, 240))
    start = kf.box.cx
    for _ in range(5):
        kf.predict()
    assert kf.box.cx == pytest.approx(start, abs=1e-6)  # no velocity yet


def test_kalman_learns_constant_velocity():
    kf = BoxKalman(Box.from_cxcyw(100, 300, 90, 240))
    for step in range(25):
        kf.predict()
        kf.update(Box.from_cxcyw(100 + 10 * step, 300, 90, 240))
    # After learning the velocity, prediction should track the next position.
    kf.predict()
    assert kf.box.cx == pytest.approx(100 + 10 * 25, abs=12)


def test_kalman_box_never_inverts():
    kf = BoxKalman(Box.from_cxcyw(100, 300, 90, 240))
    for _ in range(50):
        kf.predict()
    assert kf.box.width > 0 and kf.box.height > 0
    assert np.all(np.isfinite(kf.x))


def test_kalman_converges_on_a_step_change():
    """A sudden 100 px jump is absorbed (with overshoot) and then settled."""
    kf = BoxKalman(Box.from_cxcyw(100, 300, 90, 240))
    for _ in range(10):
        kf.predict()
        kf.update(Box.from_cxcyw(200, 300, 90, 240))
    assert kf.box.cx > 150  # moved decisively toward the measurement
    for _ in range(30):
        kf.predict()
        kf.update(Box.from_cxcyw(200, 300, 90, 240))
    assert kf.box.cx == pytest.approx(200, abs=3)


# ---------------------------------------------------------------- body tracker
def test_body_track_keeps_id_across_frames():
    tracker = BodyTracker(TrackerConfig(min_hits=1))
    ids = set()
    for step in range(8):
        tracks = tracker.update([det(100 + 5 * step)])
        ids.update(t.track_id for t in tracks)
    assert len(ids) == 1


def test_body_track_survives_short_gap():
    tracker = BodyTracker(TrackerConfig(min_hits=1, max_age=20))
    for step in range(5):
        tracks = tracker.update([det(100 + 5 * step)])
    first_id = tracks[0].track_id
    # Ten frames with no detection: the track coasts on the motion model.
    for _ in range(10):
        tracks = tracker.update([])
    assert tracks and tracks[0].track_id == first_id
    assert tracks[0].time_since_update == 10


def test_body_track_dies_after_max_age():
    tracker = BodyTracker(TrackerConfig(min_hits=1, max_age=3))
    tracker.update([det(100)])
    for _ in range(5):
        tracker.update([])
    assert tracker.update([]) == []


def test_two_people_keep_distinct_ids():
    tracker = BodyTracker(TrackerConfig(min_hits=1))
    for step in range(6):
        tracks = tracker.update([det(100 + 3 * step), det(500 - 3 * step)])
    assert len(tracks) == 2
    assert len({t.track_id for t in tracks}) == 2


def test_min_hits_gates_confirmation():
    tracker = BodyTracker(TrackerConfig(min_hits=3))
    assert tracker.update([det(100)]) == []
    assert tracker.update([det(105)]) == []
    tracks = tracker.update([det(110)])
    assert len(tracks) == 1 and tracks[0].confirmed


# ---------------------------------------------------------------- face tracker
def test_face_tracker_keeps_id_for_moving_face():
    tracker = FaceTracker(FaceTrackerConfig(iou_threshold=0.1))
    ids = set()
    for step in range(10):
        tracks = tracker.update([det(100 + 8 * step, 120, 30, 38)])
        ids.update(t.track_id for t in tracks)
    assert len(ids) == 1


def test_face_tracker_uses_velocity_prediction():
    tracker = FaceTracker(FaceTrackerConfig(iou_threshold=0.05))
    for step in range(6):
        tracker.update([det(100 + 25 * step, 120, 26, 32)])
    # A large jump consistent with the velocity should still associate.
    tracks = tracker.update([det(100 + 25 * 6, 120, 26, 32)])
    assert len(tracks) == 1


def test_face_tracker_merges_descriptors():
    tracker = FaceTracker(FaceTrackerConfig(iou_threshold=0.1))
    d1 = np.array([1.0, 0.0, 0.0, 0.0])
    for _ in range(3):
        tracks = tracker.update([det(100, 120, 30, 38)], descriptors=[d1])
    assert np.allclose(tracks[0].descriptor, d1, atol=1e-6)


def test_face_tracker_separates_distant_faces():
    tracker = FaceTracker(FaceTrackerConfig(iou_threshold=0.1))
    tracks = tracker.update([det(100, 120, 30, 38), det(400, 120, 30, 38)])
    assert len({t.track_id for t in tracks}) == 2
