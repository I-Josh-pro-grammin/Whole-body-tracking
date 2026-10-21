#!/usr/bin/env python
"""Run the full pipeline on a *real* video (real pixels, real detector).

An unlabelled clip has no ground-truth identities, so this script reports
consistency and cost rather than accuracy:

* how often a detected face could be tied to a detected body,
* the three-state distribution and the state timeline,
* how stable the identity id was (few ids == strong continuity),
* how long CONFIRMED survived each face disappearance,
* end-to-end throughput on the CPU.

Labelled *accuracy* is measured by ``scripts/run_scenarios.py``.

    python scripts/evaluate_video.py --video data/videos/walking.mp4
    python scripts/evaluate_video.py --video data/videos/walking.mp4 --save out.mp4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from bodyfollowing.config import PipelineConfig  # noqa: E402
from bodyfollowing.evaluation.visualize import draw_overlay  # noqa: E402
from bodyfollowing.perception.detector import ModelNotFound, build_detector  # noqa: E402
from bodyfollowing.pipeline import BodyFollowingPipeline  # noqa: E402
from bodyfollowing.sources.video import VideoSource  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--max-frames", type=int, default=600)
    parser.add_argument("--save", default=None, help="annotated output video")
    parser.add_argument("--out", default="reports/video_eval.json")
    parser.add_argument("--every", type=int, default=1, help="detect every Nth frame")
    parser.add_argument("--no-auto-lock", action="store_true")
    parser.add_argument("--tau", type=float, default=None)
    args = parser.parse_args()

    config = PipelineConfig()
    config.auto_lock = not args.no_auto_lock
    config.detector.face_every = max(1, args.every)
    if args.tau is not None:
        config.identity.decay_tau = args.tau

    try:
        detector = build_detector(config.detector)
    except ModelNotFound as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    pipeline = BodyFollowingPipeline(config, detector=detector)
    source = VideoSource(args.video)

    writer = None
    records: list[dict] = []
    frame_ms: list[float] = []
    persons_per_frame: list[int] = []
    faces_per_frame: list[int] = []
    associate_ok = 0
    associate_total = 0

    try:
        source.open()
        for frame in source.frames(max_frames=args.max_frames):
            t0 = time.perf_counter()
            result = pipeline.step(frame.image, now=frame.timestamp)
            frame_ms.append((time.perf_counter() - t0) * 1000.0)

            persons_per_frame.append(len(result.bodies))
            faces_per_frame.append(len(result.faces))
            associate_ok += len(result.face_body_pairs)
            associate_total += len(result.faces)

            records.append(
                {
                    "t": round(frame.timestamp, 3),
                    "state": result.claim.state.value,
                    "confidence": round(result.claim.confidence, 3),
                    "identity": result.claim.identity_id,
                    "body": result.claim.body_track_id,
                    "faces": len(result.faces),
                    "bodies": len(result.bodies),
                    "pan": round(result.command.pan, 3),
                    "tilt": round(result.command.tilt, 3),
                    "reason": result.claim.reason,
                }
            )

            if args.save:
                import cv2

                vis = draw_overlay(frame.image, result)
                if writer is None:
                    h, w = vis.shape[:2]
                    writer = cv2.VideoWriter(
                        str(args.save),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        max(source.fps, 1.0),
                        (w, h),
                    )
                writer.write(vis)
    finally:
        source.close()
        if writer is not None:
            writer.release()

    summary = summarise(records, frame_ms, persons_per_frame, faces_per_frame,
                        associate_ok, associate_total, args.video)
    print(json.dumps(summary, indent=2))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "records": records}, indent=2),
                   encoding="utf-8")
    print(f"\nwritten -> {out}")
    return 0


def summarise(
    records: list[dict],
    frame_ms: list[float],
    persons: list[int],
    faces: list[int],
    associate_ok: int,
    associate_total: int,
    video: str,
) -> dict:
    if not records:
        return {"video": video, "frames": 0}

    n = len(records)
    states = [r["state"] for r in records]
    counts = {s: states.count(s) for s in ("confirmed", "temporary", "unknown")}
    identities = sorted({r["identity"] for r in records if r["identity"] is not None})

    # Compress the state timeline for readability.
    timeline: list[dict] = []
    for rec in records:
        if not timeline or timeline[-1]["state"] != rec["state"]:
            timeline.append(
                {
                    "state": rec["state"],
                    "start": rec["t"],
                    "end": rec["t"],
                    "identity": rec["identity"],
                    "confidence": [rec["confidence"]],
                }
            )
        else:
            timeline[-1]["end"] = rec["t"]
            timeline[-1]["confidence"].append(rec["confidence"])
    for seg in timeline:
        confs = seg.pop("confidence")
        seg["confidence_mean"] = round(float(np.mean(confs)), 3)

    # How long each CONFIRMED stretch lasted after faces stopped being found.
    holds: list[float] = []
    lost_at = None
    had_face = True
    for rec in records:
        has_face = rec["faces"] > 0 and rec["state"] == "confirmed"
        if had_face and not has_face and lost_at is None:
            lost_at = rec["t"]
        if lost_at is not None and rec["state"] != "confirmed":
            holds.append(round(rec["t"] - lost_at, 2))
            lost_at = None
        had_face = has_face

    pan_values = [abs(r["pan"]) for r in records]
    return {
        "video": str(video),
        "frames": n,
        "duration_s": round(records[-1]["t"], 2),
        "throughput_fps": round(1000.0 / float(np.mean(frame_ms)), 2) if frame_ms else None,
        "latency_ms_mean": round(float(np.mean(frame_ms)), 1) if frame_ms else None,
        "latency_ms_p95": round(float(np.percentile(frame_ms, 95)), 1) if frame_ms else None,
        "state_fraction": {k: round(v / n, 3) for k, v in counts.items()},
        "state_timeline": timeline,
        "identities_claimed": len(identities),
        "identity_ids": identities,
        "face_body_association_rate": (
            round(associate_ok / associate_total, 3) if associate_total else None
        ),
        "faces_detected_total": associate_total,
        "mean_bodies_per_frame": round(float(np.mean(persons)), 2) if persons else None,
        "mean_faces_per_frame": round(float(np.mean(faces)), 2) if faces else None,
        "confirmed_hold_after_face_loss_s": holds,
        "confirmed_hold_mean_s": round(float(np.mean(holds)), 2) if holds else None,
        "centring_active_rate": round(
            sum(1 for r in records if abs(r["pan"]) > 1e-6 or abs(r["tilt"]) > 1e-6) / n, 3
        ),
        "mean_abs_pan": round(float(np.mean(pan_values)), 3) if pan_values else None,
        "note": (
            "Unlabelled video: these are consistency statistics, not accuracy. "
            "Labelled accuracy is in the simulated scenarios."
        ),
    }


if __name__ == "__main__":
    raise SystemExit(main())
