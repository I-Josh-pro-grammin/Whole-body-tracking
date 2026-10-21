"""Real-pixel scripted scenes built from real person crops.

The kinematic simulator exercises the algorithms with exact ground truth but
no pixels.  This module closes that gap *without hardware*: it lifts real
people out of a real video and re-stages them on a canvas with a scripted
script -- a head turn (face region replaced so no detector can find a face),
an occlusion (another person walks in front), and a path crossing.

The detector therefore sees genuine human appearance, and because the canvas
is generated we still know exactly who is where.  That gives labelled
accuracy on real pixels with no camera attached.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from ..perception.appearance import DetectionBatch
from ..types import Box

try:  # optional: only needed when rendering / extracting
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


@dataclass
class Sprite:
    """A real person cut out of a video frame."""

    image: np.ndarray                      # BGR crop
    face_rect: tuple[int, int, int, int] | None  # x, y, w, h relative to crop
    source_frame: int = 0

    @property
    def size(self) -> tuple[int, int]:
        h, w = self.image.shape[:2]
        return w, h


@dataclass
class SpriteScript:
    """How one sprite moves through the scene."""

    gt_id: int
    sprite: Sprite
    x0: float
    y0: float
    vx: float = 0.0
    vy: float = 0.0
    scale: float = 1.0
    face_hidden: tuple[tuple[float, float], ...] = ()
    layer: int = 0

    def placement(self, t: float) -> tuple[int, int, int, int]:
        """Canvas rect (x, y, w, h) for this sprite at time ``t``."""
        w = int(round(self.sprite.size[0] * self.scale))
        h = int(round(self.sprite.size[1] * self.scale))
        x = int(round(self.x0 + self.vx * t - w / 2))
        y = int(round(self.y0 + self.vy * t - h / 2))
        return x, y, w, h

    def face_box(self, t: float) -> Box | None:
        if self.sprite.face_rect is None:
            return None
        x, y, w, h = self.placement(t)
        fx, fy, fw, fh = self.sprite.face_rect
        sx = w / max(self.sprite.size[0], 1)
        sy = h / max(self.sprite.size[1], 1)
        return Box(
            x + fx * sx,
            y + fy * sy,
            x + (fx + fw) * sx,
            y + (fy + fh) * sy,
        )


@dataclass
class CompositeGroundTruth:
    gt_id: int
    body_box: Box
    face_box: Box | None
    body_visible: bool
    face_visible: bool
    layer: int
    in_frame: bool


@dataclass
class CompositeFrame:
    index: int
    timestamp: float
    frame_size: tuple[int, int]
    image: np.ndarray
    detections: DetectionBatch | None
    ground_truth: dict[int, CompositeGroundTruth]

    def frame_array(self) -> np.ndarray:
        return self.image


def _require_cv2() -> None:
    if cv2 is None:  # pragma: no cover
        raise RuntimeError("OpenCV is required for composited scenes")


def extract_sprites(
    video_path: str,
    detector,
    max_sprites: int = 3,
    scan_frames: int = 150,
    min_person_height: int = 140,
) -> list[Sprite]:
    """Pull distinct, well-framed people out of a real video."""
    _require_cv2()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {video_path}")

    candidates: list[tuple[float, int, Sprite]] = []
    idx = 0
    try:
        while idx < scan_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % 5 == 0:  # sample to keep it quick
                batch = detector.detect(frame)
                for person in batch.persons:
                    if person.box.height < min_person_height:
                        continue
                    face = _best_face_for(person.box, batch.faces)
                    if face is None:
                        continue
                    x1, y1, x2, y2 = person.box.as_int()
                    x1, y1 = max(0, x1), max(0, y1)
                    x2 = min(frame.shape[1], x2)
                    y2 = min(frame.shape[0], y2)
                    if x2 - x1 < 40 or y2 - y1 < 80:
                        continue
                    crop = frame[y1:y2, x1:x2].copy()
                    fx1, fy1, fx2, fy2 = face.box.as_int()
                    face_rect = (
                        max(0, fx1 - x1),
                        max(0, fy1 - y1),
                        max(1, fx2 - fx1),
                        max(1, fy2 - fy1),
                    )
                    score = face.box.area * person.score
                    candidates.append(
                        (score, idx, Sprite(image=crop, face_rect=face_rect, source_frame=idx))
                    )
            idx += 1
    finally:
        cap.release()

    candidates.sort(key=lambda c: c[0], reverse=True)
    sprites: list[Sprite] = []
    for _score, _idx, sprite in candidates:
        if len(sprites) >= max_sprites:
            break
        # Prefer visually different people (different source frames).
        if any(abs(sprite.source_frame - s.source_frame) < 20 for s in sprites):
            continue
        sprites.append(sprite)
    return sprites


def _best_face_for(body: Box, faces: list) -> object | None:
    best, best_score = None, 0.0
    for face in faces:
        score = _face_body_fit(body, face.box)
        if score > best_score:
            best, best_score = face, score
    return best if best_score > 0.5 else None


def _face_body_fit(body: Box, face: Box) -> float:
    if body.height <= 0 or face.area <= 0:
        return 0.0
    ratio = (face.cy - body.y1) / body.height
    if not (-0.05 <= ratio <= 0.4):
        return 0.0
    if body.intersection_area(face) / face.area < 0.5:
        return 0.0
    return 1.0 - abs(ratio - 0.12)


class CompositeScene:
    """A scripted canvas scene built from real sprites."""

    def __init__(
        self,
        name: str,
        background: np.ndarray,
        scripts: list[SpriteScript],
        fps: float = 30.0,
        duration: float = 9.0,
        blur_background: bool = True,
    ) -> None:
        _require_cv2()
        self.name = name
        self.fps = fps
        self.duration = duration
        self.scripts = scripts
        h, w = background.shape[:2]
        self.width, self.height = w, h
        self.background = (
            cv2.GaussianBlur(background, (31, 31), 0) if blur_background else background.copy()
        )

    def __len__(self) -> int:
        return int(round(self.duration * self.fps))

    # ---------------------------------------------------------------- geometry
    def _coverage(self, box: Box, others: list[Box]) -> float:
        if box.area <= 0 or not others:
            return 0.0
        steps = 20
        xs = np.linspace(box.x1, box.x2, steps)
        ys = np.linspace(box.y1, box.y2, steps)
        gx, gy = np.meshgrid(xs, ys)
        covered = np.zeros_like(gx, dtype=bool)
        for other in others:
            covered |= (gx >= other.x1) & (gx <= other.x2) & (gy >= other.y1) & (gy <= other.y2)
        return float(covered.mean())

    def ground_truth_at(self, t: float) -> dict[int, CompositeGroundTruth]:
        truth: dict[int, CompositeGroundTruth] = {}
        boxes = {s.gt_id: s.placement(t) for s in self.scripts}
        body_boxes = {
            gt: Box(x, y, x + w, y + h) for gt, (x, y, w, h) in boxes.items()
        }
        for script in self.scripts:
            body = body_boxes[script.gt_id]
            face = script.face_box(t)
            front = [body_boxes[o.gt_id] for o in self.scripts if o.layer > script.layer]
            hidden = any(lo <= t <= hi for lo, hi in script.face_hidden)
            inside = (
                body.x2 > 0 and body.x1 < self.width and body.y2 > 0 and body.y1 < self.height
            )
            truth[script.gt_id] = CompositeGroundTruth(
                gt_id=script.gt_id,
                body_box=body,
                face_box=None if hidden else face,
                body_visible=inside and self._coverage(body, front) < 0.6,
                face_visible=(
                    inside
                    and not hidden
                    and face is not None
                    and self._coverage(face, front) < 0.3
                ),
                layer=script.layer,
                in_frame=inside,
            )
        return truth

    # ----------------------------------------------------------------- render
    def render(self, t: float) -> np.ndarray:
        canvas = self.background.copy()
        for script in sorted(self.scripts, key=lambda s: s.layer):
            x, y, w, h = script.placement(t)
            if w <= 1 or h <= 1:
                continue
            sprite = cv2.resize(script.sprite.image, (w, h), interpolation=cv2.INTER_AREA)

            if script.sprite.face_rect is not None and any(
                lo <= t <= hi for lo, hi in script.face_hidden
            ):
                # Simulated head turn: paint over the face with blurred
                # background so no face detector can find it, while the body
                # remains perfectly visible.
                fx, fy, fw, fh = script.sprite.face_rect
                sx = w / max(script.sprite.size[0], 1)
                sy = h / max(script.sprite.size[1], 1)
                px, py = int(fx * sx), int(fy * sy)
                pw, ph = max(2, int(fw * sx)), max(2, int(fh * sy))
                patch = cv2.GaussianBlur(
                    sprite[py : py + ph, px : px + pw], (21, 21), 0
                )
                patch = (patch * 0.75 + np.array([40, 40, 40]) * 0.25).astype(np.uint8)
                sprite[py : py + ph, px : px + pw] = patch

            _paste(canvas, sprite, x, y)
        return canvas

    def frames(self) -> Iterator[CompositeFrame]:
        n = len(self)
        dt = 1.0 / self.fps
        for i in range(n):
            t = i * dt
            yield CompositeFrame(
                index=i,
                timestamp=t,
                frame_size=(self.width, self.height),
                image=self.render(t),
                detections=None,  # the real detector runs on these pixels
                ground_truth=self.ground_truth_at(t),
            )


def _paste(canvas: np.ndarray, sprite: np.ndarray, x: int, y: int) -> None:
    """Paste ``sprite`` at ``(x, y)`` clipping to the canvas."""
    ch, cw = canvas.shape[:2]
    sh, sw = sprite.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(cw, x + sw), min(ch, y + sh)
    if x2 <= x1 or y2 <= y1:
        return
    canvas[y1:y2, x1:x2] = sprite[y1 - y : y2 - y, x1 - x : x2 - x]
