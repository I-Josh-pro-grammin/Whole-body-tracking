"""Frame sources: webcam, video files and hardware-free simulated scenes."""
from .simulated import (
    SCENARIOS,
    ActorSpec,
    KinematicScene,
    NoiseConfig,
    SimFrame,
    build_scenario,
)
from .video import Frame, VideoSource, list_cameras

__all__ = [
    "SCENARIOS",
    "ActorSpec",
    "Frame",
    "KinematicScene",
    "NoiseConfig",
    "SimFrame",
    "VideoSource",
    "build_scenario",
    "list_cameras",
]
