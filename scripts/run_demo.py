#!/usr/bin/env python
"""Live demo: follow a person with the webcam (or any video file).

    python scripts/run_demo.py --source 0
    python scripts/run_demo.py --source data/videos/walking.mp4 --save reports/demo.mp4
    python scripts/run_demo.py --source 0 --no-display --max-frames 200

Controls (when a window is shown)
---------------------------------
    click   select the person to follow
    e       enroll the locked person under a name (remember them for good)
    u       unlock the target
    s       save a snapshot
    q/ESC   quit

Enrollment
----------
    The registry lives in reports/enrollments.json.  Enroll yourself once,
    then every later run recognizes and locks you on sight -- no clicking,
    and no drifting onto whoever else walks into frame.

    python scripts/run_demo.py --list-people
    python scripts/run_demo.py --enroll josh          # name the locked target
    python scripts/run_demo.py --forget josh
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
from bodyfollowing.identity.enrollment import EnrollmentBook  # noqa: E402
from bodyfollowing.perception.detector import ModelNotFound, build_detector  # noqa: E402
from bodyfollowing.pipeline import BodyFollowingPipeline  # noqa: E402
from bodyfollowing.sources.video import VideoSource  # noqa: E402


def parse_source(value: str):
    try:
        return int(value)
    except ValueError:
        return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="0", help="camera index or video path")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--save", default=None, help="write an annotated video here")
    parser.add_argument("--snapshot-dir", default="reports/frames")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--every", type=int, default=1, help="detect every Nth frame")
    parser.add_argument("--no-auto-lock", action="store_true")
    parser.add_argument("--tau", type=float, default=None, help="decay time constant")
    parser.add_argument("--summary", default=None, help="write JSON summary here")
    parser.add_argument(
        "--enrollments",
        default="reports/enrollments.json",
        help="registry of named people (JSON)",
    )
    parser.add_argument("--enroll", default=None, metavar="NAME", help="enroll the locked target under NAME, then exit")
    parser.add_argument("--forget", default=None, metavar="NAME", help="remove NAME from the registry and exit")
    parser.add_argument("--list-people", action="store_true", help="show who is enrolled, then exit")
    args = parser.parse_args()

    config = PipelineConfig()
    config.auto_lock = not args.no_auto_lock
    config.detector.face_every = max(1, args.every)
    if args.tau is not None:
        config.identity.decay_tau = args.tau

    book_path = Path(args.enrollments)
    config.enrollment.path = str(book_path)
    book = EnrollmentBook(
        book_path,
        similarity_threshold=config.enrollment.similarity_threshold,
        margin=config.enrollment.margin,
        max_samples=config.enrollment.max_samples,
    )

    if args.list_people:
        if len(book) == 0:
            print(f"no one enrolled (registry: {book_path})")
        else:
            print(f"enrolled ({len(book)}) - registry: {book_path}")
            for name in book.names():
                print(f"  {name}  ({len(book.people[name].samples)} observations)")
        return 0

    if args.forget:
        removed = book.forget(args.forget)
        book.save()
        print(f"forgot {args.forget!r}" if removed else f"{args.forget!r} was not enrolled")
        return 0 if removed else 1

    try:
        detector = build_detector(config.detector)
    except ModelNotFound as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    pipeline = BodyFollowingPipeline(config, detector=detector, enrollment=book)
    source = VideoSource(
        parse_source(args.source),
        width=None if isinstance(parse_source(args.source), str) else 640,
        real_time=False,
    )

    try:
        source.open()
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    writer = None
    snapshot_dir = Path(args.snapshot_dir)
    state_log: list[dict] = []
    last_state = None
    last_bodies = []
    frame_times: list[float] = []

    def on_mouse(event, x, y, flags, param):
        if event == 1:  # left button down
            ident = pipeline.lock_at_point(x, y, last_bodies)
            print(f"[lock] at ({x},{y}) -> {getattr(ident, 'identity_id', None)}")

    display = not args.no_display
    if display:
        try:
            import cv2

            cv2.namedWindow("body following", cv2.WINDOW_NORMAL)
            cv2.setMouseCallback("body following", on_mouse)
        except Exception as exc:
            print(f"[warn] no display available ({exc}); running headless")
            display = False

    print(f"source: {args.source}  fps={source.fps:.1f}  size={source.size}")
    if len(book):
        print(
            f"[enroll] {len(book)} known: {', '.join(book.names())} "
            "-- recognized people lock on sight"
        )
    else:
        print("[enroll] no one enrolled yet -- press 'e' over a locked person to teach me")
    started = time.monotonic()
    processed = 0

    try:
        for frame in source.frames(max_frames=args.max_frames):
            t0 = time.perf_counter()
            result = pipeline.step(frame.image, now=frame.timestamp)
            frame_times.append(time.perf_counter() - t0)
            last_bodies = result.bodies
            processed += 1

            if result.claim.state is not last_state:
                print(
                    f"[{frame.timestamp:6.2f}s] {result.claim.state.value:<9} "
                    f"conf={result.claim.confidence:.2f}  {result.claim.reason}"
                )
                last_state = result.claim.state
                state_log.append(
                    {
                        "t": round(frame.timestamp, 3),
                        "state": result.claim.state.value,
                        "confidence": round(result.claim.confidence, 3),
                        "identity": result.claim.identity_id,
                        "reason": result.claim.reason,
                    }
                )

            vis = draw_overlay(frame.image, result)

            if args.save:
                import cv2

                if writer is None:
                    h, w = vis.shape[:2]
                    writer = cv2.VideoWriter(
                        str(args.save),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        max(source.fps, 1.0),
                        (w, h),
                    )
                writer.write(vis)

            if display:
                import cv2

                cv2.imshow("body following", vis)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("u"):
                    pipeline.unlock()
                    print("[lock] cleared")
                if key == ord("s"):
                    snapshot_dir.mkdir(parents=True, exist_ok=True)
                    path = snapshot_dir / f"demo_{frame.index:05d}.png"
                    cv2.imwrite(str(path), vis)
                    print(f"[snap] {path}")
    finally:
        source.close()
        if writer is not None:
            writer.release()
        if display:
            import cv2

            cv2.destroyAllWindows()

    elapsed = time.monotonic() - started
    fps = processed / elapsed if elapsed > 0 else 0.0
    mean_ms = 1000 * float(np.mean(frame_times)) if frame_times else 0.0
    print(
        f"\nprocessed {processed} frames in {elapsed:.1f}s "
        f"({fps:.1f} fps end-to-end, {mean_ms:.1f} ms/frame)"
    )

    if args.summary:
        Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary).write_text(
            json.dumps(
                {
                    "source": str(args.source),
                    "frames": processed,
                    "fps": round(fps, 2),
                    "mean_frame_ms": round(mean_ms, 2),
                    "state_log": state_log,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"summary -> {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
