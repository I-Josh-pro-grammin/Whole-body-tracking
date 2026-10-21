"""End-to-end pipeline: frame in, honest identity claim + centring command out.

The pipeline is deliberately source-agnostic: it consumes BGR numpy frames
(webcam, video file, or a rendered synthetic scene) so exactly the same code
path is exercised in every evaluation mode.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from .association.face_body import AssociationResult, associate
from .config import PipelineConfig
from .control.centering import CentringController
from .identity.enrollment import EnrollmentBook
from .identity.manager import IdentityManager
from .types import (
    BodyTrack,
    CentringCommand,
    Detection,
    FaceTrack,
    FrameResult,
    IdentityClaim,
    IdentityState,
)
from .tracking.body_tracker import BodyTracker
from .tracking.face_tracker import FaceTracker


class BodyFollowingPipeline:
    """Whole-person tracking + face association + identity continuity."""

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        detector=None,
        enrollment: Optional["EnrollmentBook"] = None,
    ) -> None:
        self.config = config or PipelineConfig()
        self.detector = detector  # any object with .detect(frame) -> DetectionBatch
        self.enrollment = enrollment
        self.body_tracker = BodyTracker(self.config.tracker)
        self.face_tracker = FaceTracker(self.config.face_tracker)
        self.identity = IdentityManager(
            config=self.config.identity,
            association_config=self.config.association,
            auto_lock=self.config.auto_lock,
            enrollment=enrollment,
        )
        self.centring = CentringController(self.config.centring)

        self._frame_index = 0
        self._start_time: float | None = None
        self._last_now: float = 0.0
        self._last_bodies: list[BodyTrack] = []
        self._last_faces: list[FaceTrack] = []

    # ------------------------------------------------------------------ helpers
    @property
    def frame_index(self) -> int:
        return self._frame_index

    def resize_frame_size(self, frame: np.ndarray) -> tuple[int, int]:
        h, w = frame.shape[:2]
        return int(w), int(h)

    # --------------------------------------------------------------------- step
    def step(
        self,
        frame: np.ndarray,
        now: Optional[float] = None,
        detections=None,
    ) -> FrameResult:
        """Process one frame.

        ``detections`` may be supplied directly (used by the kinematic
        simulator and unit tests); otherwise the configured detector runs on
        the frame.
        """
        if now is None:
            if self._start_time is None:
                self._start_time = time.monotonic()
            now = time.monotonic() - self._start_time
        self._last_now = now

        frame_size = self.resize_frame_size(frame) if frame is not None else (0, 0)

        if detections is None:
            if self.detector is None:
                raise RuntimeError(
                    "no detector configured and no detections supplied"
                )
            detections = self.detector.detect(frame)

        persons: list[Detection] = list(detections.persons)
        faces: list[Detection] = list(detections.faces)

        # ---- tracking -------------------------------------------------
        bodies = self.body_tracker.update(persons)
        faces = self.face_tracker.update(
            faces,
            descriptors=getattr(detections, "face_descriptors", None),
            yaws=getattr(detections, "face_yaws", None),
        )
        self._last_bodies = bodies
        self._last_faces = faces

        # ---- spatial association --------------------------------------
        association: AssociationResult = associate(
            bodies, faces, self.config.association
        )

        # ---- identity ledger ------------------------------------------
        identities = self.identity.update(bodies, faces, association, now)
        claim: IdentityClaim = self.identity.claim(now)

        # ---- target centring ------------------------------------------
        body = self._find_body(bodies, claim.body_track_id)
        face = self._find_face(faces, claim.face_track_id)
        command: CentringCommand = self.centring.update(body, face, claim, frame_size)

        result = FrameResult(
            frame_index=self._frame_index,
            timestamp=now,
            frame_size=frame_size,
            bodies=bodies,
            faces=faces,
            identities=identities,
            claim=claim,
            command=command,
            face_body_pairs=association.pairs,
            debug={
                "detections": {
                    "persons": len(persons),
                    "faces": len(faces),
                },
                "association": [m.to_dict() for m in association.matches],
            },
        )
        self._frame_index += 1
        return result

    # ------------------------------------------------------------------ lookup
    @staticmethod
    def _find_body(bodies: list[BodyTrack], track_id: int | None) -> BodyTrack | None:
        if track_id is None:
            return None
        for body in bodies:
            if body.track_id == track_id:
                return body
        return None

    @staticmethod
    def _find_face(faces: list[FaceTrack], track_id: int | None) -> FaceTrack | None:
        if track_id is None:
            return None
        for face in faces:
            if face.track_id == track_id:
                return face
        return None

    # -------------------------------------------------------------------- lock
    def lock_at_point(self, x: float, y: float, bodies: list[BodyTrack] | None = None):
        """Lock the followed identity to whoever is under the given point."""
        return self.identity.lock_at_point(x, y, bodies if bodies is not None else self._last_bodies)

    def unlock(self) -> None:
        self.identity.lock(None)

    def enroll_target(self, name: str) -> str | None:
        """Teach the enrolled-person registry who the locked target is."""
        return self.identity.enroll_target(name, now=self._last_now)

    def enrollments(self) -> EnrollmentBook | None:
        return self.enrollment

    @property
    def state(self) -> IdentityState:
        claim = self.identity.claim(self._last_now)
        return claim.state
