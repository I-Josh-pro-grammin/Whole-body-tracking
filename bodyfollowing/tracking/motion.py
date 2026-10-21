"""Motion-consistency gating for body association.

IoU alone is blind to *direction*.  When two people walk toward each other and
their boxes overlap, the target's track can latch onto the other person's box
simply because it overlaps the prediction -- an identity swap that no amount
of downstream cleverness can undo.

The gate multiplies an IoU score by a plausibility factor derived from the
track's learned velocity.  It deliberately only engages on **large**
displacements: detector jitter is a few pixels, which is the same order as the
per-frame motion of a walking person, so any per-frame direction test would be
measuring noise.  A jump is only judged once it exceeds a size-aware
significance threshold (a fraction of the person's own bounding box).
"""

from __future__ import annotations

import math

import numpy as np

from ..types import Box


def motion_factor(
    predicted: Box,
    detection: Box,
    velocity: tuple[float, float, float, float],
    min_speed: float = 1.5,
    pause: float = 0.2,
    floor: float = 0.12,
    jump_ratio: float = 3.0,
    significance_scale: float = 0.08,
    significance_px: float = 10.0,
) -> float:
    """Plausibility in ``(0, 1]`` of moving from ``predicted`` to ``detection``."""
    vx, vy = velocity[0], velocity[1]
    speed = math.hypot(vx, vy)
    if speed < min_speed:
        return 1.0

    dx = detection.cx - predicted.cx
    dy = detection.cy - predicted.cy
    dist = math.hypot(dx, dy)

    scale = max(predicted.width, predicted.height)
    significant = max(significance_px, significance_scale * scale, 2.0 * speed)
    if dist < significant:
        # Ordinary jitter around the prediction: nothing to judge.
        return 1.0

    factor = 1.0
    ratio = dist / speed
    if ratio > jump_ratio:
        factor *= max(floor, jump_ratio / ratio)

    cos = (vx * dx + vy * dy) / (speed * dist)
    if cos < pause:
        factor = min(factor, max(floor, (1.0 + cos) / 2.0))
    return factor


def motion_gate_matrix(
    predicted: list[Box],
    detected: list[Box],
    velocities: list[tuple[float, float, float, float]],
    min_speed: float = 1.5,
    pause: float = 0.2,
    floor: float = 0.12,
    jump_ratio: float = 3.0,
    significance_scale: float = 0.08,
    significance_px: float = 10.0,
) -> np.ndarray:
    """Pairwise gating factors (same shape as an IoU matrix)."""
    gate = np.ones((len(predicted), len(detected)), dtype=np.float64)
    for i, pred in enumerate(predicted):
        for j, det in enumerate(detected):
            gate[i, j] = motion_factor(
                pred,
                det,
                velocities[i],
                min_speed=min_speed,
                pause=pause,
                floor=floor,
                jump_ratio=jump_ratio,
                significance_scale=significance_scale,
                significance_px=significance_px,
            )
    return gate
