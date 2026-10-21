"""CPU-only appearance descriptors and head-pose estimation.

No embedding network is used: the descriptor is a compact colour-layout
signature of an expanded face crop (face + hair + shoulders).  It is cheap,
deterministic, and good enough to answer the only question the identity layer
asks of it -- "does this look like the person I was following a few seconds
ago?" -- while being honest that it is *not* a biometric.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..types import Detection


@dataclass
class DetectionBatch:
    """Everything the perception layer found in one frame."""

    persons: list[Detection] = field(default_factory=list)
    faces: list[Detection] = field(default_factory=list)
    face_descriptors: list[np.ndarray] = field(default_factory=list)
    face_yaws: list[float] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def __bool__(self) -> bool:  # pragma: no cover - convenience
        return bool(self.persons or self.faces)


def face_descriptor(
    frame: np.ndarray,
    box,
    expand: float = 2.2,
    grid: int = 4,
    bins: int = 8,
) -> np.ndarray:
    """Colour-layout signature of an expanded face crop.

    Combines a per-channel HSV histogram with a coarse spatial luminance
    layout, so two people with similar skin tone but different hair/clothing
    still separate.
    """
    h, w = frame.shape[:2]
    x1 = int(max(0, box.x1 - box.width * (expand - 1) / 2))
    y1 = int(max(0, box.y1 - box.height * (expand - 1) * 0.35))
    x2 = int(min(w, box.x2 + box.width * (expand - 1) / 2))
    y2 = int(min(h, box.y2 + box.height * (expand - 1) * 0.65))
    if x2 - x1 < 6 or y2 - y1 < 6:
        return np.zeros(0, dtype=np.float64)

    crop = frame[y1:y2, x1:x2]
    if crop.ndim == 2:
        crop = np.stack([crop] * 3, axis=-1)

    hsv = _hsv(crop)
    # Per-channel histograms of the HSV crop.
    hist_parts = []
    for c in range(3):
        hh, _ = np.histogram(hsv[..., c], bins=bins, range=(0.0, 1.0))
        total = hh.sum()
        hist_parts.append(hh / total if total > 0 else hh)
    layout = _layout(crop, grid)
    desc = np.concatenate(hist_parts + [layout])
    desc = desc - desc.mean()
    norm = float(np.linalg.norm(desc))
    return desc / norm if norm > 0 else desc


def _hsv(crop: np.ndarray) -> np.ndarray:
    """Convert BGR uint8 to float HSV without requiring OpenCV."""
    try:
        import cv2

        return cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(np.float32) / np.array(
            [180.0, 255.0, 255.0], dtype=np.float32
        )
    except Exception:  # pragma: no cover - exercised only without cv2
        arr = crop.astype(np.float32) / 255.0
        mx = arr.max(axis=2)
        mn = arr.min(axis=2)
        return np.stack([mx, mx - mn, mx], axis=-1)


def _layout(crop: np.ndarray, grid: int) -> np.ndarray:
    """Coarse spatial luminance layout of the crop."""
    try:
        import cv2

        small = cv2.resize(crop, (grid, grid), interpolation=cv2.INTER_AREA)
        if small.ndim == 3:
            small = small.mean(axis=2)
        vals = small.astype(np.float32).ravel() / 255.0
    except Exception:  # pragma: no cover
        vals = np.zeros(grid * grid, dtype=np.float32)
    vals = vals - vals.mean()
    norm = float(np.linalg.norm(vals))
    return vals / norm if norm > 0 else vals


def estimate_yaw(landmarks: list[tuple[float, float]] | None) -> float:
    """Rough head yaw in degrees from 5-point face landmarks.

    Landmark order (YuNet): right eye, left eye, nose, right mouth, left mouth.
    The nose offset from the eye midpoint, normalised by inter-ocular distance,
    is a monotonic proxy for yaw.
    """
    if not landmarks or len(landmarks) < 3:
        return 0.0
    re_x, _ = landmarks[0]
    le_x, _ = landmarks[1]
    nose_x, _ = landmarks[2]
    eye_mid = 0.5 * (re_x + le_x)
    eye_dist = abs(le_x - re_x)
    if eye_dist < 1e-3:
        return 0.0
    return float((nose_x - eye_mid) / eye_dist * 100.0)


def estimate_pitch(landmarks: list[tuple[float, float]] | None) -> float:
    """Rough head pitch in degrees (nose above/below the eye-mouth midline)."""
    if not landmarks or len(landmarks) < 5:
        return 0.0
    _, re_y = landmarks[0]
    _, le_y = landmarks[1]
    _, nose_y = landmarks[2]
    _, rm_y = landmarks[3]
    _, lm_y = landmarks[4]
    eye_y = 0.5 * (re_y + le_y)
    mouth_y = 0.5 * (rm_y + lm_y)
    span = abs(mouth_y - eye_y)
    if span < 1e-3:
        return 0.0
    return float((nose_y - (eye_y + mouth_y) / 2.0) / span * 100.0)
