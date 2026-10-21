"""Small geometric / numeric helpers shared across the package."""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np

from ..types import Box


def iou_matrix(a: Sequence[Box], b: Sequence[Box]) -> np.ndarray:
    """Pairwise IoU between two lists of boxes -> ``(len(a), len(b))``."""
    out = np.zeros((len(a), len(b)), dtype=np.float64)
    for i, ba in enumerate(a):
        for j, bb in enumerate(b):
            out[i, j] = ba.iou(bb)
    return out


def overlap_ratio_matrix(a: Sequence[Box], b: Sequence[Box]) -> np.ndarray:
    """Fraction of each box in ``a`` covered by each box in ``b``."""
    out = np.zeros((len(a), len(b)), dtype=np.float64)
    for i, ba in enumerate(a):
        if ba.area <= 0:
            continue
        for j, bb in enumerate(b):
            out[i, j] = ba.intersection_area(bb) / ba.area
    return out


def nms(boxes: Iterable[Box], scores: Iterable[float], iou_thr: float = 0.5) -> list[int]:
    """Greedy non-maximum suppression; returns kept indices, best first."""
    boxes = list(boxes)
    scores = list(scores)
    order = sorted(range(len(boxes)), key=lambda i: scores[i], reverse=True)
    keep: list[int] = []
    while order:
        i = order.pop(0)
        keep.append(i)
        remaining = []
        for j in order:
            if boxes[i].iou(boxes[j]) <= iou_thr:
                remaining.append(j)
        order = remaining
    return keep


def ema(previous: float, value: float, alpha: float) -> float:
    """Exponential moving average; ``alpha`` is the weight of the new value."""
    alpha = min(1.0, max(0.0, alpha))
    return (1.0 - alpha) * previous + alpha * value


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def cosine_similarity(a: np.ndarray | None, b: np.ndarray | None) -> float:
    """Cosine similarity of two 1-D descriptors, tolerant of degenerate input."""
    if a is None or b is None:
        return 0.0
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size == 0 or a.size != b.size:
        return 0.0
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 1e-9 or nb <= 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def log_ratio_score(observed: float, expected: float, sigma: float) -> float:
    """Score in ``(0, 1]`` for a positive ratio, Gaussian in log space.

    ``sigma`` controls tolerance: at ``|ln(o/e)| == sigma`` the score is
    ``exp(-0.5) ~= 0.61``.
    """
    if observed <= 0 or expected <= 0:
        return 0.0
    z = math.log(observed / expected) / max(sigma, 1e-6)
    return math.exp(-0.5 * z * z)


def band_score(value: float, low: float, high: float, tolerance: float) -> float:
    """1 inside ``[low, high]``, decaying linearly over ``tolerance`` outside."""
    if low <= value <= high:
        return 1.0
    if tolerance <= 0:
        return 0.0
    if value < low:
        d = low - value
    else:
        d = value - high
    return clamp(1.0 - d / tolerance, 0.0, 1.0)


def normalized_histogram(image: np.ndarray, bins: int = 8) -> np.ndarray:
    """Cheap CPU-only appearance descriptor: normalised per-channel histogram.

    Used only for coarse re-identification of a face crop after a long gap.
    """
    if image is None or image.size == 0:
        return np.zeros(0, dtype=np.float64)
    img = image.astype(np.float32)
    channels = [img] if img.ndim == 2 else [img[..., c] for c in range(img.shape[2])]
    parts = []
    for ch in channels:
        hist, _ = np.histogram(ch, bins=bins, range=(0.0, 255.0))
        total = hist.sum()
        parts.append(hist / total if total > 0 else hist)
    desc = np.concatenate(parts).astype(np.float64)
    # Mean-centre so that cosine similarity behaves like a correlation.
    desc -= desc.mean()
    norm = np.linalg.norm(desc)
    return desc / norm if norm > 0 else desc
