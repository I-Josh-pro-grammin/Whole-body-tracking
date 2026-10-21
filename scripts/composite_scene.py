#!/usr/bin/env python
"""Ground-truth evaluation on *real pixels* with no camera attached.

Real people are lifted out of a video, re-staged on a canvas with a scripted
choreography (head turn, occlusion, path crossing), and then fed through the
real detector + pipeline.  Because the canvas is generated, the ground truth
is exact.

    python scripts/composite_scene.py --video data/videos/walking.mp4
    python scripts/composite_scene.py --video data/videos/walking.mp4 --scenario head_turn
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from bodyfollowing.config import PipelineConfig  # noqa: E402
from bodyfollowing.evaluation.metrics import check_expectation, evaluate_scene  # noqa: E402
from bodyfollowing.evaluation.metrics import Expectation  # noqa: E402
from bodyfollowing.perception.detector import ModelNotFound, build_detector  # noqa: E402
from bodyfollowing.sources.composited import (  # noqa: E402
    CompositeScene,
    SpriteScript,
    extract_sprites,
)


def load_background(video: str, size: tuple[int, int]) -> np.ndarray:
    import cv2

    cap = cv2.VideoCapture(video)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"could not read a frame from {video}")
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)


def make_scripts(sprites, person_height: float, canvas_w: int) -> list[SpriteScript]:
    """Scale sprites to a uniform standing height and assign roles."""
    scripts: list[SpriteScript] = []
    for idx, sprite in enumerate(sprites):
        scale = person_height / max(sprite.size[1], 1)
        scripts.append(
            SpriteScript(
                gt_id=idx + 1,
                sprite=sprite,
                x0=0.0,
                y0=0.0,
                scale=scale,
            )
        )
    return scripts


def build_scene(kind: str, sprites, background: np.ndarray, canvas=(640, 360), fps=30.0) -> CompositeScene:
    w, h = canvas
    person_h = 200.0
    scripts = make_scripts(sprites, person_h, w)
    if not scripts:
        raise RuntimeError("no usable sprites were extracted from the video")

    target = scripts[0]
    ground_y = h * 0.72

    if kind == "head_turn":
        target.x0, target.y0 = -60.0, ground_y
        target.vx = 52.0
        target.face_hidden = ((2.5, 5.0),)
        scene = CompositeScene(kind, background, [target], fps=fps, duration=8.0)
    elif kind == "occlusion":
        target.x0, target.y0 = -40.0, ground_y
        target.vx = 48.0
        scene = CompositeScene(kind, background, [target], fps=fps, duration=8.0)
        if len(scripts) > 1:
            occluder = scripts[1]
            occluder.x0, occluder.y0 = w + 60.0, ground_y - 6
            occluder.vx = -58.0
            occluder.layer = 1
            occluder.scale *= 1.08
            scene.scripts.append(occluder)
    elif kind == "path_crossing":
        target.x0, target.y0 = -50.0, ground_y
        target.vx = 58.0
        scene = CompositeScene(kind, background, [target], fps=fps, duration=8.0)
        if len(scripts) > 1:
            other = scripts[1]
            other.x0, other.y0 = w + 50.0, ground_y - 4
            other.vx = -58.0
            other.layer = 1
            scene.scripts.append(other)
    elif kind == "combined":
        target.x0, target.y0 = -50.0, ground_y
        target.vx = 46.0
        target.face_hidden = ((1.8, 3.2), (5.0, 6.2))
        scene = CompositeScene(kind, background, [target], fps=fps, duration=9.0)
        if len(scripts) > 1:
            other = scripts[1]
            other.x0, other.y0 = w + 60.0, ground_y - 5
            other.vx = -62.0
            other.layer = 1
            other.scale *= 1.06
            scene.scripts.append(other)
    else:
        raise ValueError(f"unknown composite scenario {kind!r}")
    return scene


EXPECTATIONS = {
    "head_turn": Expectation(
        description="Real pixels: the head turns away for 2.5 s.",
        require_temporary_during_face_loss=True,
        require_recovery=True,
        max_wrong_person_frames=0,
        min_centring_active_rate=0.5,
    ),
    "occlusion": Expectation(
        description="Real pixels: a real person walks in front of the target.",
        require_temporary_during_face_loss=True,
        require_recovery=True,
        max_wrong_person_frames=0,
        min_centring_active_rate=0.5,
    ),
    "path_crossing": Expectation(
        description="Real pixels: two people cross at similar depth.",
        require_temporary_during_face_loss=False,
        require_recovery=False,
        max_wrong_person_frames=0,
        min_centring_active_rate=0.5,
    ),
    "combined": Expectation(
        description="Real pixels: head turn + occlusion together.",
        require_temporary_during_face_loss=True,
        require_recovery=True,
        max_wrong_person_frames=0,
        min_centring_active_rate=0.4,
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default="data/videos/walking.mp4")
    parser.add_argument("--scenario", default=None,
                        help="one of head_turn, occlusion, path_crossing, combined")
    parser.add_argument("--out", default="reports")
    parser.add_argument("--frames", type=int, default=None, help="limit frames")
    parser.add_argument("--save-video", action="store_true",
                        help="write annotated videos and figure snapshots")
    args = parser.parse_args()

    import cv2

    canvas = (640, 360)
    background = load_background(args.video, canvas)

    try:
        detector = build_detector(PipelineConfig().detector)
    except ModelNotFound as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"extracting real people from {args.video} ...")
    sprites = extract_sprites(args.video, detector, max_sprites=2)
    if not sprites:
        print("error: no usable person crops found in the video", file=sys.stderr)
        return 2
    print(f"  found {len(sprites)} sprite(s): "
          + ", ".join(f"{s.size[0]}x{s.size[1]}" for s in sprites))

    kinds = [args.scenario] if args.scenario else list(EXPECTATIONS)
    all_ok = True
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"video": args.video, "canvas": list(canvas), "scenes": {}}

    for kind in kinds:
        scene = build_scene(kind, sprites, background)
        if args.frames:
            scene.duration = args.frames / scene.fps
        print(f"\n[{kind}] {len(scene)} frames")
        result = evaluate_scene(
            scene,
            config=PipelineConfig(),
            target_gt_id=1,
            lock_time=1.2,
            detector=detector,
        )
        checks = check_expectation(result, EXPECTATIONS[kind])
        m = result.metrics
        print(
            f"  states={m['state_fraction_post_lock']} "
            f"precision={m['confirmed_precision']} "
            f"wrong={m['wrong_person_frames']} "
            f"ids={m['identities_claimed']} "
            f"recovery={m['recovery_mean_s']}s "
            f"centring={m['centring_active_rate']:.0%}"
        )
        for check in checks:
            print(f"  {'PASS' if check.ok else 'FAIL'} {check.name}: {check.detail}")
            all_ok = all_ok and check.ok
        report["scenes"][kind] = {
            "frames": len(scene),
            "metrics": m,
            "checks": [c.as_dict() for c in checks],
            "passed": all(c.ok for c in checks),
        }
        if args.save_video:
            save_overlay_video(
                scene, detector, out_dir / f"composite_{kind}.mp4", shots=out_dir / "figures"
            )

    json_path = out_dir / "composite_scenes.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nreport -> {json_path}")
    return 0 if all_ok else 1


def save_overlay_video(scene, detector, path: Path, every: int = 1, shots: Path | None = None) -> None:
    """Re-render the scene with pipeline annotations (visual evidence)."""
    import cv2

    from bodyfollowing.evaluation.visualize import draw_overlay
    from bodyfollowing.pipeline import BodyFollowingPipeline

    cfg = PipelineConfig()
    cfg.auto_lock = False
    pipeline = BodyFollowingPipeline(cfg, detector=detector)
    writer = None
    shot_times = [scene.duration * f for f in (0.15, 0.45, 0.8)]
    saved = 0
    locked = False
    for sim in scene.frames():
        result = pipeline.step(sim.image, now=sim.timestamp)
        # Mirror the evaluator: a user selects the target at t=1.2 s.
        if not locked and sim.timestamp >= 1.2:
            target = sim.ground_truth.get(1)
            if target is not None and target.body_visible:
                locked = (
                    pipeline.lock_at_point(
                        target.body_box.cx, target.body_box.cy, result.bodies
                    )
                    is not None
                )
        vis = draw_overlay(sim.image, result)
        if shots is not None and saved < len(shot_times) and sim.timestamp >= shot_times[saved]:
            shots.mkdir(parents=True, exist_ok=True)
            out = shots / f"composite_{scene.name}_{saved}.png"
            cv2.imwrite(str(out), vis)
            saved += 1
        if sim.index % every:
            continue
        if writer is None:
            h, w = vis.shape[:2]
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                                     scene.fps, (w, h))
        writer.write(vis)
    if writer is not None:
        writer.release()
        print(f"  video -> {path}")


if __name__ == "__main__":
    raise SystemExit(main())
