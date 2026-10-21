"""Tunable configuration for the body-following module.

Every magic number used by the tracker, the association layer, the identity
state machine and the centring controller lives here so that behaviour can be
reproduced and swept by the evaluation harness.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class TrackerConfig:
    """Multi-object body tracker (SORT-style: Kalman + IoU + Hungarian)."""

    iou_threshold: float = 0.25
    max_age: int = 30          # frames a track survives without a detection
    min_hits: int = 3          # consecutive hits before a track is confirmed
    # Reject associations that would make the track teleport or reverse.
    # This is what stops two people walking toward each other from swapping
    # ids when their boxes overlap.
    motion_gate: bool = True
    motion_min_speed: float = 1.5   # px/frame before the gate engages
    motion_pause: float = 0.2       # cos-similarity below which a turn is penalised
    motion_penalty_floor: float = 0.12
    motion_jump_ratio: float = 3.0  # implied speed / track speed before penalising


@dataclass
class FaceTrackerConfig:
    """Face track association (faces move faster than bodies)."""

    iou_threshold: float = 0.20
    max_age: int = 12


@dataclass
class AssociationConfig:
    """Spatial face-to-body mapping.

    A face belongs to a body when it sits inside the body's head band, aligned
    with the body centre line, and is the right size relative to the body.
    """

    # Face centre must be inside the top fraction of the body box.
    head_band_top: float = -0.10   # allow a little above the body box
    head_band_bottom: float = 0.32  # face centre below this fraction of body height
    # Expected face height as a fraction of body height (full-body person).
    expected_face_ratio: float = 0.17
    face_ratio_sigma: float = 0.55  # log-space tolerance
    # Horizontal alignment tolerance, as a fraction of body width.
    horizontal_tolerance: float = 0.60
    # Weighting of the individual terms in the final association score.
    w_vertical: float = 0.34
    w_horizontal: float = 0.22
    w_scale: float = 0.20
    w_overlap: float = 0.24
    # Minimum combined score to accept a face->body match.
    min_score: float = 0.45
    # Ambiguity guard: if the two best bodies for one face score closer than
    # this, the match is refused (keeps the model honest during crossings).
    ambiguity_margin: float = 0.08
    # Re-identification margin used when scoring geometric evidence.
    reid_similarity: float = 0.72
    reid_max_gap_seconds: float = 12.0


@dataclass
class IdentityConfig:
    """The honest three-state model with exponential confidence decay.

    Confidence decays as ``c(t) = c0 * exp(-(t - t0) / tau)`` while no fresh
    face observation supports the link.  ``tau`` is the e-folding time; the
    half-life is ``tau * ln(2)``.
    """

    # Decay time-constant in seconds.  2.0 s => half-life 1.39 s.
    decay_tau: float = 2.0
    # Confidence at or above this (with a fresh face) => CONFIRMED.
    confirm_threshold: float = 0.70
    # A face observation is "fresh" for this many seconds.  This bounds how
    # long CONFIRMED survives a face miss; it is deliberately short because
    # continuity across a *longer* gap is the TEMPORARY state's job.
    freshness_window: float = 0.5
    # Below this the identity becomes UNKNOWN (no claim is made).
    unknown_threshold: float = 0.25
    # Below this the identity is retired entirely.
    retire_threshold: float = 0.02
    # A body can be missing this long and still keep the identity alive.
    body_grace_seconds: float = 1.5
    # Retire an identity that has been UNKNOWN / bodiless for this long.
    retire_after: float = 8.0
    # Minimum association score that counts as positive face evidence.
    min_evidence_score: float = 0.55
    # Temporal smoothing of confidence when fresh evidence arrives (0..1).
    evidence_smoothing: float = 0.35
    # Re-acquire a returning person by face appearance (cosine similarity).
    appearance_reid_similarity: float = 0.70
    # If two candidate identities are this close in appearance, refuse to pick
    # (two look-alikes must not silently swap).
    appearance_reid_margin: float = 0.06
    # Sudden appearance change on a bound body => the tracker swapped people.
    appearance_conflict_threshold: float = 0.35
    # Only judge appearance after this many face observations.
    min_appearance_observations: int = 3
    # Number of face observations required before a CONFIRMED claim is allowed.
    # Raise this on a noisy detector to make the first assertion more cautious.
    min_confirmations: int = 1


@dataclass
class EnrollmentConfig:
    """Persistent recognition of named people across runs.

    An enrolled person is identified by a stored appearance descriptor.  The
    thresholds are deliberately strict: this is a colour-layout signature,
    not a face embedding, so a false name is far worse than no name.
    """

    # JSON registry of enrolled people (None = keep it in memory only).
    path: str | None = "reports/enrollments.json"
    # Minimum cosine similarity to accept a recognition.
    similarity_threshold: float = 0.80
    # Best must beat the runner-up by this much, or the match is refused.
    margin: float = 0.05
    # Observations kept per person (oldest dropped beyond this).
    max_samples: int = 12
    # A recognized person is locked on sight without waiting for a click.
    auto_lock_known: bool = True
    # Observations of a *new* face needed before it is offered for enrollment,
    # so a single badly-lit frame cannot poison a person's record.
    min_samples_to_enroll: int = 3


@dataclass
class CentringConfig:
    """Target centring controller (pixel errors -> normalised pan/tilt)."""

    # Dead-zone as a fraction of the half-frame dimension (kills jitter).
    deadzone_x: float = 0.06
    deadzone_y: float = 0.08
    # Proportional gains (command units per normalised error unit).
    gain_x: float = 1.0
    gain_y: float = 0.7
    # Exponential smoothing of the measured error (0..1, higher = smoother).
    error_smoothing: float = 0.45
    # Command slew-rate limit per frame (keeps a servo from jerking).
    max_slew: float = 0.25
    # Fraction of the body height below the top used as the aim point when the
    # face is not available (aim at the chest/head, not the feet).
    fallback_aim_fraction: float = 0.18


@dataclass
class DetectorConfig:
    """Perception settings.  CPU-only by construction."""

    person_model: str = "models/yolov8n.pt"
    face_model: str = "models/face_detection_yunet_2023mar.onnx"
    person_conf: float = 0.30
    face_conf: float = 0.60
    # Inference size (smaller = faster on CPU).  Measured on the reference
    # CPU: 312 ms/frame at 640, 212 ms at 480, 172 ms at 320.
    img_size: int = 480
    device: str = "cpu"
    # Detect faces on every Nth frame (cost control); 1 = every frame.
    face_every: int = 1
    # Minimum face pixel height to bother with.
    min_face_px: int = 24


@dataclass
class PipelineConfig:
    """Top-level configuration bundle."""

    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    face_tracker: FaceTrackerConfig = field(default_factory=FaceTrackerConfig)
    association: AssociationConfig = field(default_factory=AssociationConfig)
    identity: IdentityConfig = field(default_factory=IdentityConfig)
    enrollment: EnrollmentConfig = field(default_factory=EnrollmentConfig)
    centring: CentringConfig = field(default_factory=CentringConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    # Lock onto the first person whose face is confirmed (None = manual lock).
    auto_lock: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def half_life(self) -> float:
        """Half-life of the exponential confidence decay, in seconds."""
        return self.identity.decay_tau * math.log(2.0)
