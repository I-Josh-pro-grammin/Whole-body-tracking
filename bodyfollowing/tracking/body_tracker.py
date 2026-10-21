"""SORT-style multi-object tracker for person bodies.

Detections -> Kalman prediction -> IoU association (Hungarian) -> track
lifecycle.  A track keeps its id while it is re-detected; when it is missed it
is carried by the motion model for ``max_age`` frames, which is exactly what
lets the identity layer survive an occlusion.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import TrackerConfig
from ..types import BodyTrack, Box, Detection
from ..utils.assignment import match_scores, unmatched
from ..utils.geometry import iou_matrix
from .kalman import BoxKalman
from .motion import motion_gate_matrix


@dataclass
class _BodyTrackState:
    track_id: int
    kf: BoxKalman
    score: float
    hits: int = 1
    age: int = 1
    time_since_update: int = 0
    confirmed: bool = False
    history: list[Box] = field(default_factory=list)

    def to_public(self) -> BodyTrack:
        return BodyTrack(
            track_id=self.track_id,
            box=self.kf.box,
            velocity=self.kf.velocity,
            score=self.score,
            hits=self.hits,
            age=self.age,
            time_since_update=self.time_since_update,
            confirmed=self.confirmed,
        )


class BodyTracker:
    """Multi-person tracker with stable ids."""

    def __init__(self, config: TrackerConfig | None = None) -> None:
        self.config = config or TrackerConfig()
        self.tracks: list[_BodyTrackState] = []
        self._next_id = 1

    # ------------------------------------------------------------------ update
    def update(self, detections: list[Detection]) -> list[BodyTrack]:
        """Consume one frame of person detections, return the live tracks."""
        cfg = self.config

        for track in self.tracks:
            track.kf.predict()
            track.age += 1
            track.time_since_update += 1

        predicted = [t.kf.box for t in self.tracks]
        det_boxes = [d.box for d in detections]

        matches: list[tuple[int, int, float]] = []
        if self.tracks and detections:
            scores = iou_matrix(predicted, det_boxes)
            if cfg.motion_gate:
                gate = motion_gate_matrix(
                    predicted,
                    det_boxes,
                    [t.kf.velocity for t in self.tracks],
                    min_speed=cfg.motion_min_speed,
                    pause=cfg.motion_pause,
                    floor=cfg.motion_penalty_floor,
                    jump_ratio=cfg.motion_jump_ratio,
                )
                scores = scores * gate
            # The threshold is applied inside match_scores.
            matches = match_scores(scores, cfg.iou_threshold)

        matched_tracks = {m[0] for m in matches}
        matched_dets = {m[1] for m in matches}

        for t_idx, d_idx, _score in matches:
            track = self.tracks[t_idx]
            det = detections[d_idx]
            track.kf.update(det.box)
            track.score = det.score
            track.hits += 1
            track.time_since_update = 0
            track.history.append(det.box)
            if track.hits >= cfg.min_hits:
                track.confirmed = True

        for d_idx in unmatched(len(detections), matched_dets):
            det = detections[d_idx]
            self.tracks.append(
                _BodyTrackState(
                    track_id=self._next_id,
                    kf=BoxKalman(det.box),
                    score=det.score,
                    hits=1,
                    age=1,
                    time_since_update=0,
                    confirmed=cfg.min_hits <= 1,
                    history=[det.box],
                )
            )
            self._next_id += 1

        # Retire tracks that have gone too long without a detection.
        self.tracks = [
            t for t in self.tracks if t.time_since_update <= cfg.max_age
        ]
        _ = matched_tracks  # kept for readability of the matching block

        return [t.to_public() for t in self.tracks if t.confirmed]

    # -------------------------------------------------------------- accessors
    def get(self, track_id: int) -> _BodyTrackState | None:
        for track in self.tracks:
            if track.track_id == track_id:
                return track
        return None

    @property
    def next_id(self) -> int:
        return self._next_id
