#!/usr/bin/env python
"""Download the CPU-only model weights used by the pipeline.

* ``yolov8n.pt`` -- person detector (Ultralytics, COCO class 0)
* ``face_detection_yunet_2023mar.onnx`` -- YuNet face detector + 5 landmarks

Everything is fetched once into ``models/`` and then runs offline.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"

YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
YUNET_PATH = MODELS / "face_detection_yunet_2023mar.onnx"


def download(url: str, dest: Path, min_bytes: int = 10_000) -> None:
    if dest.exists() and dest.stat().st_size >= min_bytes:
        print(f"[skip] {dest.name} already present ({dest.stat().st_size} bytes)")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[get ] {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as response, tmp.open("wb") as fh:
        data = response.read()
        fh.write(data)
    if len(data) < min_bytes:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"download too small ({len(data)} bytes) for {url}")
    tmp.replace(dest)
    print(f"[ok  ] {dest.name} ({len(data)} bytes)")


def download_yolo() -> Path:
    """Fetch the YOLO person weights and park them in ``models/``."""
    import shutil

    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")
    src = Path(getattr(model, "ckpt_path", "yolov8n.pt"))
    dest = MODELS / "yolov8n.pt"
    if src.exists() and src.resolve() != dest.resolve():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), dest)
    target = dest if dest.exists() else src
    print(f"[ok  ] YOLO person model ready at {target}")
    return target


def verify() -> None:
    import cv2
    import numpy as np

    create = getattr(cv2, "FaceDetectorYN_create", None) or getattr(
        cv2.FaceDetectorYN, "create", None
    )
    detector = create(str(YUNET_PATH), "", (320, 320), 0.6, 0.3, 5000)
    detector.setInputSize((320, 320))
    blank = np.zeros((320, 320, 3), dtype=np.uint8)
    ok, faces = detector.detect(blank)
    print(f"[ok  ] YuNet loaded (detect on blank frame -> ok={ok}, faces="
          f"{0 if faces is None else len(faces)})")


def main() -> int:
    MODELS.mkdir(parents=True, exist_ok=True)
    try:
        download(YUNET_URL, YUNET_PATH)
    except Exception as exc:
        print(f"[warn] YuNet download failed: {exc}", file=sys.stderr)
        return 1
    try:
        download_yolo()
    except Exception as exc:
        print(f"[warn] YOLO weights unavailable: {exc}", file=sys.stderr)
        return 1
    verify()
    print("\nAll models ready. Try: python scripts/run_demo.py --source 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
