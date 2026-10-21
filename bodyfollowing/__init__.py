"""Whole-person tracking with honest face-association identity continuity.

A CPU-only module that keeps a *temporal identity* for a person across
temporary facial loss (head turns, occlusions, detection misses) using an
explicit three-state model -- CONFIRMED / TEMPORARY / UNKNOWN -- backed by
exponential confidence decay.

Quick start::

    from bodyfollowing import BodyFollowingPipeline, PipelineConfig

    pipe = BodyFollowingPipeline(PipelineConfig())
    result = pipe.step(frame, now=0.0)
    print(result.claim.state, result.claim.confidence, result.command.pan)
"""

from __future__ import annotations

from .config import (
    AssociationConfig,
    CentringConfig,
    DetectorConfig,
    FaceTrackerConfig,
    IdentityConfig,
    PipelineConfig,
    TrackerConfig,
)
from .identity.manager import IdentityManager
from .identity.states import (
    decay_confidence,
    confidence_after,
    state_for,
)
from .pipeline import BodyFollowingPipeline
from .types import (
    BodyTrack,
    Box,
    CentringCommand,
    Detection,
    FaceTrack,
    FrameResult,
    Identity,
    IdentityClaim,
    IdentityState,
)

__all__ = [
    "AssociationConfig",
    "BodyFollowingPipeline",
    "BodyTrack",
    "Box",
    "CentringCommand",
    "CentringConfig",
    "Detection",
    "DetectorConfig",
    "FaceTrack",
    "FaceTrackerConfig",
    "FrameResult",
    "Identity",
    "IdentityClaim",
    "IdentityConfig",
    "IdentityManager",
    "IdentityState",
    "PipelineConfig",
    "TrackerConfig",
    "confidence_after",
    "decay_confidence",
    "state_for",
]

__version__ = "1.0.0"
