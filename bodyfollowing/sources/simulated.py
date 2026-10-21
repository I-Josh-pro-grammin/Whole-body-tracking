"""Hardware-free scenario simulator.

No webcam is needed to evaluate identity continuity.  This module generates a
**detector-level** observation stream from a scripted scene: actors walk with
constant velocity, can turn their head away (face disappears while the body
stays visible), can be occluded by another actor walking in front, and can
cross paths.  Realistic detector imperfection is injected on top: box jitter,
random misses and confidence noise.

Why simulate detections rather than pixels?  Because it isolates the question
the module is supposed to answer -- *does identity survive a gap in face
evidence?* -- with exact ground truth and no confound from a detector's
failure to recognise a rendered cartoon.  The real-pixel path is exercised
separately by the webcam/video sources and the composited-scene harness.

Every actor carries a ground-truth ``gt_id``.  That is what makes the
evaluation honest: we can measure whether a CONFIRMED claim was actually
correct, instead of assuming it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterator

import numpy as np

from ..perception.appearance import DetectionBatch
from ..types import Box, Detection

DESCRIPTOR_DIM = 40  # 3 channels x 8 histogram bins + 4x4 layout


@dataclass
class NoiseConfig:
    """Detector imperfection model."""

    body_jitter: float = 0.015     # box-size-relative Gaussian noise
    face_jitter: float = 0.030
    body_miss: float = 0.04        # probability a visible body is missed
    face_miss: float = 0.08        # probability a visible face is missed
    score_low: float = 0.72
    score_high: float = 0.97
    descriptor_noise: float = 0.06
    # Real detectors fuse two heavily overlapping people into one box.  This
    # is the failure mode that makes trackers swap identities during a
    # crossing, so the simulator models it explicitly.
    body_merge_iou: float = 0.55
    body_merge_prob: float = 0.75
    seed: int = 7


@dataclass
class ActorSpec:
    """A scripted person in the scene."""

    gt_id: int
    x0: float
    y0: float
    vx: float = 0.0
    vy: float = 0.0
    body_h: float = 250.0
    aspect: float = 0.40
    # Time windows where the head is turned away and the face is invisible.
    face_hidden: tuple[tuple[float, float], ...] = ()
    # 0 = background, 1 = in front of layer-0 actors.
    layer: int = 0
    desc_seed: int | None = None

    def body_box_at(self, t: float) -> Box:
        cx = self.x0 + self.vx * t
        cy = self.y0 + self.vy * t
        w = self.body_h * self.aspect
        return Box.from_cxcyw(cx, cy, w, self.body_h)

    def face_box_at(self, t: float) -> Box:
        body = self.body_box_at(t)
        fh = 0.17 * self.body_h
        fw = 0.78 * fh
        return Box.from_cxcyw(body.cx, body.y1 + 0.075 * self.body_h, fw, fh)


@dataclass
class GroundTruthActor:
    """What is actually true about one actor in one frame."""

    gt_id: int
    body_box: Box
    face_box: Box | None
    body_visible: bool
    face_visible: bool
    layer: int
    in_frame: bool


@dataclass
class SimFrame:
    """One simulated frame: detections plus the truth they should support."""

    index: int
    timestamp: float
    frame_size: tuple[int, int]
    detections: DetectionBatch
    ground_truth: dict[int, GroundTruthActor]

    def frame_array(self) -> np.ndarray:
        """A blank image of the right shape (the pipeline only needs the size)."""
        w, h = self.frame_size
        return np.zeros((h, w, 3), dtype=np.uint8)


class KinematicScene:
    """Scripted multi-actor scene with detector-level simulation."""

    def __init__(
        self,
        name: str,
        actors: list[ActorSpec],
        width: int = 640,
        height: int = 480,
        fps: float = 30.0,
        duration: float = 9.0,
        noise: NoiseConfig | None = None,
    ) -> None:
        self.name = name
        self.actors = actors
        self.width = width
        self.height = height
        self.fps = fps
        self.duration = duration
        self.noise = noise or NoiseConfig()
        self._rng = np.random.default_rng(self.noise.seed)
        self._descriptors = self._make_descriptors()

    # ------------------------------------------------------------- appearance
    def _make_descriptors(self) -> dict[int, np.ndarray]:
        out: dict[int, np.ndarray] = {}
        for actor in self.actors:
            rng = np.random.default_rng(
                actor.desc_seed if actor.desc_seed is not None else 1000 + actor.gt_id
            )
            vec = rng.normal(size=DESCRIPTOR_DIM)
            out[actor.gt_id] = vec / np.linalg.norm(vec)
        return out

    # ---------------------------------------------------------------- geometry
    def _in_frame(self, box: Box, margin: float = 0.0) -> bool:
        return (
            box.x2 > margin
            and box.x1 < self.width - margin
            and box.y2 > margin
            and box.y1 < self.height - margin
        )

    def _coverage(self, box: Box, others: list[Box]) -> float:
        """Fraction of ``box`` covered by the union of ``others``."""
        if box.area <= 0 or not others:
            return 0.0
        # Rasterise on a coarse grid: robust to overlapping occluders.
        steps = 24
        xs = np.linspace(box.x1, box.x2, steps)
        ys = np.linspace(box.y1, box.y2, steps)
        gx, gy = np.meshgrid(xs, ys)
        covered = np.zeros_like(gx, dtype=bool)
        for other in others:
            covered |= (
                (gx >= other.x1) & (gx <= other.x2) & (gy >= other.y1) & (gy <= other.y2)
            )
        return float(covered.mean())

    def ground_truth_at(self, t: float, index: int) -> dict[int, GroundTruthActor]:
        truth: dict[int, GroundTruthActor] = {}
        boxes = {a.gt_id: a.body_box_at(t) for a in self.actors}

        for actor in self.actors:
            body = boxes[actor.gt_id]
            face = actor.face_box_at(t)
            # Occluders are actors on a nearer layer.
            front = [
                boxes[o.gt_id]
                for o in self.actors
                if o.layer > actor.layer
            ]
            body_cover = self._coverage(body, front)
            face_cover = self._coverage(face, front)
            head_turned = any(lo <= t <= hi for lo, hi in actor.face_hidden)
            in_frame = self._in_frame(body, margin=-8)

            truth[actor.gt_id] = GroundTruthActor(
                gt_id=actor.gt_id,
                body_box=body,
                face_box=None if head_turned else face,
                body_visible=in_frame and body_cover < 0.65,
                face_visible=in_frame and not head_turned and face_cover < 0.35,
                layer=actor.layer,
                in_frame=in_frame,
            )
        return truth

    # -------------------------------------------------------------- detections
    def _jitter(self, box: Box, sigma: float) -> Box:
        sx = box.width * sigma
        sy = box.height * sigma
        return Box(
            box.x1 + self._rng.normal(0, sx),
            box.y1 + self._rng.normal(0, sy),
            box.x2 + self._rng.normal(0, sx),
            box.y2 + self._rng.normal(0, sy),
        )

    def _score(self) -> float:
        lo, hi = self.noise.score_low, self.noise.score_high
        return float(np.clip(self._rng.uniform(lo, hi), 0.01, 0.999))

    def _merged_actors(self, ordered: list[GroundTruthActor]) -> set[int]:
        """Which visible actors the detector fuses into somebody else's box.

        When two body boxes overlap a lot, a real detector typically returns a
        single box for the pair.  The actor with the smaller visible box is the
        one that disappears, and it disappears *stochastically*, which is what
        makes the resulting track loss realistic.
        """
        merged: set[int] = set()
        visible = [a for a in ordered if a.body_visible and a.in_frame]
        for i, a in enumerate(visible):
            for b in visible[i + 1 :]:
                if a.body_box.iou(b.body_box) <= self.noise.body_merge_iou:
                    continue
                loser = a if a.body_box.area <= b.body_box.area else b
                if self._rng.random() < self.noise.body_merge_prob:
                    merged.add(loser.gt_id)
        return merged

    def detections_at(
        self, t: float, truth: dict[int, GroundTruthActor]
    ) -> DetectionBatch:
        persons: list[Detection] = []
        faces: list[Detection] = []
        descriptors: list[np.ndarray] = []
        yaws: list[float] = []

        # Back-to-front so that nearer actors appear later (and occlude).
        ordered = sorted(truth.values(), key=lambda a: a.layer)
        merged_away = self._merged_actors(ordered)
        for actor_truth in ordered:
            if actor_truth.body_visible:
                if self._rng.random() >= self.noise.body_miss and (
                    actor_truth.gt_id not in merged_away
                ):
                    box = actor_truth.body_box.clip(self.width, self.height)
                    box = self._jitter(box, self.noise.body_jitter)
                    persons.append(
                        Detection(box=box, score=self._score(), class_name="person")
                    )

            if actor_truth.face_visible and actor_truth.face_box is not None:
                if self._rng.random() >= self.noise.face_miss:
                    fbox = actor_truth.face_box.clip(self.width, self.height)
                    fbox = self._jitter(fbox, self.noise.face_jitter)
                    faces.append(Detection(box=fbox, score=self._score(), class_name="face"))
                    base = self._descriptors[actor_truth.gt_id]
                    noisy = base + self._rng.normal(
                        0, self.noise.descriptor_noise, size=base.shape
                    )
                    norm = np.linalg.norm(noisy)
                    descriptors.append(noisy / norm if norm > 0 else noisy)
                    yaws.append(float(self._rng.normal(0, 4)))

        return DetectionBatch(
            persons=persons,
            faces=faces,
            face_descriptors=descriptors,
            face_yaws=yaws,
            meta={"truth": truth},
        )

    # ------------------------------------------------------------------- frames
    def frames(self) -> Iterator[SimFrame]:
        n = int(round(self.duration * self.fps))
        dt = 1.0 / self.fps
        for i in range(n):
            t = i * dt
            truth = self.ground_truth_at(t, i)
            yield SimFrame(
                index=i,
                timestamp=t,
                frame_size=(self.width, self.height),
                detections=self.detections_at(t, truth),
                ground_truth=truth,
            )

    def __len__(self) -> int:
        return int(round(self.duration * self.fps))


# --------------------------------------------------------------------------- library
def _walker(gt_id: int, x0: float, y0: float, vx: float, **kw) -> ActorSpec:
    return ActorSpec(gt_id=gt_id, x0=x0, y0=y0, vx=vx, **kw)


def head_turn_scene() -> KinematicScene:
    """The target turns their head away for ~2.2 s; the body never leaves."""
    return KinematicScene(
        name="head_turn",
        actors=[
            _walker(1, 120, 300, 62.0, body_h=250.0, face_hidden=((3.0, 5.2),)),
        ],
        duration=9.0,
    )


def occlusion_scene() -> KinematicScene:
    """A passer-by crosses in front of the target and hides the face."""
    return KinematicScene(
        name="occlusion",
        actors=[
            _walker(1, 90, 300, 55.0, body_h=255.0),
            # Layer 1 => walks in front, crossing the target's head around t~4 s.
            _walker(2, 640, 300, -62.0, body_h=268.0, layer=1, desc_seed=99),
        ],
        duration=9.0,
    )


def path_crossing_scene() -> KinematicScene:
    """Two people cross at the same depth; the tracker can swap them."""
    return KinematicScene(
        name="path_crossing",
        actors=[
            _walker(1, 110, 300, 70.0, body_h=250.0),
            _walker(2, 530, 300, -70.0, body_h=244.0, desc_seed=99),
        ],
        duration=8.0,
    )


def long_absence_scene() -> KinematicScene:
    """Face hidden for 6 s: confidence must fall to UNKNOWN, then recover."""
    return KinematicScene(
        name="long_absence",
        actors=[
            _walker(1, 90, 300, 52.0, body_h=250.0, face_hidden=((2.0, 8.0),)),
        ],
        duration=11.0,
    )


def combined_scene() -> KinematicScene:
    """Head turn + occlusion + a second person crossing: hardest case."""
    return KinematicScene(
        name="combined",
        actors=[
            _walker(1, 80, 310, 58.0, body_h=252.0, face_hidden=((2.2, 3.6), (5.5, 7.0))),
            _walker(2, 620, 310, -64.0, body_h=258.0, layer=1, desc_seed=99),
        ],
        duration=11.0,
    )


SCENARIOS = {
    "head_turn": head_turn_scene,
    "occlusion": occlusion_scene,
    "path_crossing": path_crossing_scene,
    "long_absence": long_absence_scene,
    "combined": combined_scene,
}


def build_scenario(name: str, seed: int | None = None) -> KinematicScene:
    """Build a named scenario, optionally with a different detector noise seed."""
    if name not in SCENARIOS:
        raise KeyError(f"unknown scenario '{name}'; choose from {sorted(SCENARIOS)}")
    scene = SCENARIOS[name]()
    if seed is not None:
        scene.noise = replace(scene.noise, seed=seed)
        scene._rng = np.random.default_rng(seed)
    return scene
