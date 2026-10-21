"""Short-horizon tracker for face detections.

Faces are small and move quickly relative to their size, so plain IoU is a
poor association metric.  This tracker predicts the next box with a constant
velocity estimate and scores pairs with a blend of IoU and normalised centre
distance.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import FaceTrackerConfig
from ..types import Box, Detection, FaceTrack
from ..utils.assignment import match_scores, unmatched
from ..utils.geometry import ema, iou_matrix


@dataclass
class _FaceTrackState:
    track_id: int
    box: Box
    velocity: tuple[float, float] = (0.0, 0.0)
    score: float = 0.0
    hits: int = 1
    age: int = 1
    time_since_update: int = 0
    yaw: float = 0.0
    descriptor: np.ndarray | None = None
    descriptor_alpha: float = 0.4
    history: list[Box] = field(default_factory=list)

    def predicted_box(self) -> Box:
        dx, dy = self.velocity
        return Box(
            self.box.x1 + dx,
            self.box.y1 + dy,
            self.box.x2 + dx,
            self.box.y2 + dy,
        )

    def to_public(self) -> FaceTrack:
        return FaceTrack(
            track_id=self.track_id,
            box=self.box,
            score=self.score,
            hits=self.hits,
            age=self.age,
            time_since_update=self.time_since_update,
            yaw=self.yaw,
            descriptor=self.descriptor,
        )


class FaceTracker:
    """Maintains stable ids for faces across frames."""

    def __init__(self, config: FaceTrackerConfig | None = None) -> None:
        self.config = config or FaceTrackerConfig()
        self.tracks: list[_FaceTrackState] = []
        self._next_id = 1

    # ---------------------------------------------------------------- similarity
    @staticmethod
    def _similarity(predicted: list[Box], detected: list[Box]) -> np.ndarray:
        iou = iou_matrix(predicted, detected)
        sim = np.zeros_like(iou)
        for i, pb in enumerate(predicted):
            scale = max(pb.width, pb.height, 1.0)
            for j, db in enumerate(detected):
                dist = np.hypot(pb.cx - db.cx, pb.cy - db.cy)
                dist_score = float(np.exp(-0.5 * (dist / (0.75 * scale)) ** 2))
                sim[i, j] = 0.6 * iou[i, j] + 0.4 * dist_score
        return sim

    # ------------------------------------------------------------------ update
    def update(
        self,
        detections: list[Detection],
        descriptors: list[np.ndarray] | None = None,
        yaws: list[float] | None = None,
    ) -> list[FaceTrack]:
        cfg = self.config
        for track in self.tracks:
            track.age += 1
            track.time_since_update += 1

        predicted = [t.predicted_box() for t in self.tracks]
        det_boxes = [d.box for d in detections]

        matches = []
        if self.tracks and detections:
            sim = self._similarity(predicted, det_boxes)
            matches = match_scores(sim, cfg.iou_threshold)

        matched_dets = {m[1] for m in matches}
        for t_idx, d_idx, _score in matches:
            track = self.tracks[t_idx]
            det = detections[d_idx]
            new_box = det.box
            dx = new_box.cx - track.box.cx
            dy = new_box.cy - track.box.cy
            track.velocity = (
                ema(track.velocity[0], dx, 0.5),
                ema(track.velocity[1], dy, 0.5),
            )
            track.box = new_box
            track.score = det.score
            track.hits += 1
            track.time_since_update = 0
            track.history.append(new_box)
            if descriptors is not None and d_idx < len(descriptors):
                desc = descriptors[d_idx]
                if desc is not None and desc.size:
                    if track.descriptor is None or track.descriptor.size != desc.size:
                        track.descriptor = desc.copy()
                    else:
                        merged = ema(track.descriptor, desc, track.descriptor_alpha)
                        norm = np.linalg.norm(merged)
                        track.descriptor = merged / norm if norm > 0 else merged
            if yaws is not None and d_idx < len(yaws):
                track.yaw = yaws[d_idx]

        for d_idx in unmatched(len(detections), matched_dets):
            det = detections[d_idx]
            desc = None
            if descriptors is not None and d_idx < len(descriptors):
                desc = descriptors[d_idx]
            yaw = yaws[d_idx] if yaws is not None and d_idx < len(yaws) else 0.0
            self.tracks.append(
                _FaceTrackState(
                    track_id=self._next_id,
                    box=det.box,
                    score=det.score,
                    yaw=yaw,
                    descriptor=desc,
                    history=[det.box],
                )
            )
            self._next_id += 1

        self.tracks = [t for t in self.tracks if t.time_since_update <= cfg.max_age]
        return [t.to_public() for t in self.tracks]

    # --------------------------------------------------------------- accessors
    def get(self, track_id: int) -> _FaceTrackState | None:
        for track in self.tracks:
            if track.track_id == track_id:
                return track
        return None
