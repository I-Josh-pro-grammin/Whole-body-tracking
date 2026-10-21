"""Spatial face-to-body mapping.

A face is not free-floating: on a standing or walking person it lives inside
the *head band* at the top of the body box, centred on the body's midline, and
its size is a fixed fraction of the body height.  This module turns those
three facts into a pair score, then solves the one-to-one assignment.

It also reports two honesty signals:

``contamination``
    How much of the face is covered by *some other* body.  When somebody walks
    in front of the target, the face we see belongs to the occluder; using it
    to confirm the target would be a lie.

``ambiguous``
    The runner-up body scored within ``ambiguity_margin``.  Two people
    crossing at the same depth can produce this; refusing to decide is the
    correct behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import AssociationConfig
from ..types import BodyTrack, Box, FaceTrack
from ..utils.assignment import match_scores
from ..utils.geometry import band_score, log_ratio_score


@dataclass
class FaceBodyMatch:
    """One accepted face -> body assignment."""

    face_id: int
    body_id: int
    score: float
    contamination: float = 0.0
    ambiguous: bool = False
    components: dict[str, float] = field(default_factory=dict)

    @property
    def is_trustworthy(self) -> bool:
        """A match is usable as identity evidence only if it is unambiguous."""
        return (not self.ambiguous) and self.contamination < 0.35

    def to_dict(self) -> dict[str, float]:
        return {
            "face_id": self.face_id,
            "body_id": self.body_id,
            "score": round(self.score, 4),
            "contamination": round(self.contamination, 4),
            "ambiguous": self.ambiguous,
            **{k: round(v, 4) for k, v in self.components.items()},
        }


@dataclass
class AssociationResult:
    matches: list[FaceBodyMatch] = field(default_factory=list)
    unmatched_faces: list[int] = field(default_factory=list)
    unmatched_bodies: list[int] = field(default_factory=list)
    score_matrix: np.ndarray | None = None

    def for_body(self, body_id: int) -> FaceBodyMatch | None:
        best = None
        for match in self.matches:
            if match.body_id == body_id and (best is None or match.score > best.score):
                best = match
        return best

    def for_face(self, face_id: int) -> FaceBodyMatch | None:
        for match in self.matches:
            if match.face_id == face_id:
                return match
        return None

    @property
    def pairs(self) -> list[tuple[int, int, float]]:
        return [(m.face_id, m.body_id, m.score) for m in self.matches]


def pair_score(
    body: Box,
    face: Box,
    cfg: AssociationConfig,
) -> tuple[float, dict[str, float]]:
    """Score how well ``face`` belongs to the person in ``body`` (0..1)."""
    if body.height <= 0 or body.width <= 0 or face.area <= 0:
        return 0.0, {"vertical": 0.0, "horizontal": 0.0, "scale": 0.0, "overlap": 0.0}

    # 1) Vertical: the face centre must sit in the top slice of the body.
    v_ratio = (face.cy - body.y1) / body.height
    s_vertical = band_score(
        v_ratio,
        cfg.head_band_top,
        cfg.head_band_bottom,
        tolerance=max(0.12, cfg.head_band_bottom),
    )

    # 2) Horizontal: near the body midline.
    half_w = max(body.width / 2.0, 1.0)
    h_offset = abs(face.cx - body.cx) / half_w
    s_horizontal = float(np.exp(-0.5 * (h_offset / cfg.horizontal_tolerance) ** 2))

    # 3) Scale: face height relative to body height.
    expected = cfg.expected_face_ratio * body.height
    s_scale = log_ratio_score(face.height, expected, cfg.face_ratio_sigma)

    # 4) Overlap: the face should be inside the body box.
    overlap = body.intersection_area(face) / face.area if face.area > 0 else 0.0
    s_overlap = max(0.0, min(1.0, overlap))

    score = (
        cfg.w_vertical * s_vertical
        + cfg.w_horizontal * s_horizontal
        + cfg.w_scale * s_scale
        + cfg.w_overlap * s_overlap
    )
    # Killing conditions: a face floating outside the body cannot be its head.
    if s_overlap < 0.25 or s_vertical < 0.05:
        score *= 0.25
    return float(max(0.0, min(1.0, score))), {
        "vertical": s_vertical,
        "horizontal": s_horizontal,
        "scale": s_scale,
        "overlap": s_overlap,
    }


def contamination_for(face: Box, bodies: list[Box], owner_index: int) -> float:
    """Largest fraction of ``face`` covered by a body other than the owner."""
    worst = 0.0
    if face.area <= 0:
        return 0.0
    for idx, body in enumerate(bodies):
        if idx == owner_index:
            continue
        worst = max(worst, body.intersection_area(face) / face.area)
    return float(min(1.0, worst))


def associate(
    bodies: list[BodyTrack],
    faces: list[FaceTrack],
    cfg: AssociationConfig,
    observed_only: bool = True,
) -> AssociationResult:
    """Solve the face -> body assignment for one frame.

    ``observed_only`` restricts matching to faces the detector actually saw in
    *this* frame.  Coasted face tracks (``time_since_update > 0``) are useful
    for drawing, but using one as identity evidence would silently extend a
    CONFIRMED claim past the last real observation -- exactly the dishonesty
    the three-state model exists to prevent.
    """
    result = AssociationResult()
    if observed_only:
        faces = [f for f in faces if f.time_since_update == 0]
    if not bodies or not faces:
        result.unmatched_faces = [f.track_id for f in faces]
        result.unmatched_bodies = [b.track_id for b in bodies]
        return result


    body_boxes = [b.box for b in bodies]
    face_boxes = [f.box for f in faces]

    scores = np.zeros((len(bodies), len(faces)), dtype=np.float64)
    parts: dict[tuple[int, int], dict[str, float]] = {}
    for bi, body in enumerate(body_boxes):
        for fi, face in enumerate(face_boxes):
            s, comp = pair_score(body, face, cfg)
            scores[bi, fi] = s
            parts[(bi, fi)] = comp
    result.score_matrix = scores

    # Record ambiguity *before* the one-to-one solve, using raw scores.
    ambiguous_faces: set[int] = set()
    if len(bodies) > 1:
        for fi in range(len(faces)):
            column = np.sort(scores[:, fi])[::-1]
            if column[0] > 0 and (column[0] - column[1]) < cfg.ambiguity_margin:
                ambiguous_faces.add(fi)

    for bi, fi, s in match_scores(scores, cfg.min_score):
        result.matches.append(
            FaceBodyMatch(
                face_id=faces[fi].track_id,
                body_id=bodies[bi].track_id,
                score=s,
                contamination=contamination_for(face_boxes[fi], body_boxes, bi),
                ambiguous=fi in ambiguous_faces,
                components=parts.get((bi, fi), {}),
            )
        )

    used_bodies = {m.body_id for m in result.matches}
    used_faces = {m.face_id for m in result.matches}
    result.unmatched_bodies = [b.track_id for b in bodies if b.track_id not in used_bodies]
    result.unmatched_faces = [f.track_id for f in faces if f.track_id not in used_faces]
    return result
