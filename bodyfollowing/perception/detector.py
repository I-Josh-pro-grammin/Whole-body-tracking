"""CPU-only perception: YOLO person detection + YuNet face detection.

Both models run on the CPU.  YOLO gives whole-body boxes (the tracking
anchor); YuNet gives faces plus 5 landmarks, which are used both for the
face-to-body geometry and for a coarse head-yaw estimate.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import DetectorConfig
from ..types import Box, Detection
from .appearance import DetectionBatch, estimate_yaw, face_descriptor

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ModelNotFound(RuntimeError):
    """Raised when a model file or its ML runtime is unavailable."""


# --------------------------------------------------------------------------- person
class YoloPersonDetector:
    """Person-only YOLO detector (COCO class 0)."""

    def __init__(self, config: DetectorConfig | None = None) -> None:
        self.config = config or DetectorConfig()
        try:
            from ultralytics import YOLO  # imported lazily: heavy
        except Exception as exc:  # pragma: no cover - environment dependent
            raise ModelNotFound(
                "ultralytics is not installed; run `pip install ultralytics`"
            ) from exc
        try:
            self.model = YOLO(self.config.person_model)
        except Exception as exc:  # pragma: no cover - needs network on first run
            raise ModelNotFound(
                f"could not load person model '{self.config.person_model}': {exc}"
            ) from exc

    def detect(self, frame: np.ndarray) -> list[Detection]:
        results = self.model.predict(
            frame,
            imgsz=self.config.img_size,
            conf=self.config.person_conf,
            classes=[0],  # person
            device=self.config.device,
            verbose=False,
        )
        out: list[Detection] = []
        for res in results:
            boxes = getattr(res, "boxes", None)
            if boxes is None or len(boxes) == 0:
                continue
            xyxy = boxes.xyxy.cpu().numpy()
            conf = boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), c in zip(xyxy, conf):
                if x2 - x1 < 8 or y2 - y1 < 16:
                    continue
                out.append(
                    Detection(
                        box=Box(float(x1), float(y1), float(x2), float(y2)),
                        score=float(c),
                        class_id=0,
                        class_name="person",
                    )
                )
        return out


# --------------------------------------------------------------------------- face
class YuNetFaceDetector:
    """OpenCV YuNet face detector (ONNX, CPU)."""

    def __init__(self, config: DetectorConfig | None = None) -> None:
        self.config = config or DetectorConfig()
        try:
            import cv2
        except Exception as exc:  # pragma: no cover
            raise ModelNotFound("opencv-python is not installed") from exc
        self._cv2 = cv2

        model_path = Path(self.config.face_model)
        if not model_path.is_absolute():
            model_path = PROJECT_ROOT / model_path
        if not model_path.exists():
            raise ModelNotFound(
                f"face model not found at {model_path}. "
                "Run `python scripts/download_models.py` first."
            )

        create = getattr(cv2, "FaceDetectorYN_create", None) or getattr(
            cv2.FaceDetectorYN, "create", None
        )
        if create is None:  # pragma: no cover
            raise ModelNotFound(
                "this OpenCV build lacks FaceDetectorYN; install opencv-contrib-python"
            )
        self.detector = create(
            str(model_path), "", (320, 320), self.config.face_conf, 0.3, 5000
        )
        self._input_size: tuple[int, int] | None = None

    def detect(
        self, frame: np.ndarray
    ) -> tuple[list[Detection], list[np.ndarray], list[float]]:
        h, w = frame.shape[:2]
        if self._input_size != (w, h):
            self.detector.setInputSize((w, h))
            self._input_size = (w, h)

        ok, faces = self.detector.detect(frame)
        detections: list[Detection] = []
        descriptors: list[np.ndarray] = []
        yaws: list[float] = []
        if not ok or faces is None:
            return detections, descriptors, yaws

        for row in faces:
            row = np.asarray(row, dtype=np.float32).ravel()
            if row.size < 15:
                continue
            x, y, fw, fh = row[0:4]
            score = float(row[14])
            if fh < self.config.min_face_px or fw < self.config.min_face_px:
                continue
            box = Box(float(x), float(y), float(x + fw), float(y + fh))
            landmarks = [
                (float(row[4 + 2 * i]), float(row[5 + 2 * i])) for i in range(5)
            ]
            detections.append(
                Detection(
                    box=box,
                    score=score,
                    class_id=1,
                    class_name="face",
                    landmarks=landmarks,
                )
            )
            descriptors.append(face_descriptor(frame, box))
            yaws.append(estimate_yaw(landmarks))
        return detections, descriptors, yaws


# ----------------------------------------------------------------------- composite
class CompositeDetector:
    """Runs person + face detection and packages the result."""

    def __init__(
        self,
        config: DetectorConfig | None = None,
        person_detector=None,
        face_detector=None,
    ) -> None:
        self.config = config or DetectorConfig()
        self.person_detector = person_detector or YoloPersonDetector(self.config)
        self.face_detector = face_detector or YuNetFaceDetector(self.config)
        self._frame_count = 0
        self._last_faces: list[Detection] = []
        self._last_descriptors: list[np.ndarray] = []
        self._last_yaws: list[float] = []

    def detect(self, frame: np.ndarray) -> DetectionBatch:
        cfg = self.config
        persons = self.person_detector.detect(frame)

        run_faces = (self._frame_count % max(1, cfg.face_every)) == 0
        if run_faces:
            faces, descriptors, yaws = self.face_detector.detect(frame)
            self._last_faces, self._last_descriptors, self._last_yaws = (
                faces,
                descriptors,
                yaws,
            )
        else:
            faces, descriptors, yaws = [], [], []
        self._frame_count += 1

        return DetectionBatch(
            persons=persons,
            faces=faces,
            face_descriptors=descriptors,
            face_yaws=yaws,
            meta={"face_frame": run_faces},
        )


def build_detector(config: DetectorConfig | None = None) -> CompositeDetector:
    """Construct the default CPU detector stack with a clear failure mode."""
    config = config or DetectorConfig()
    try:
        return CompositeDetector(config)
    except ModelNotFound as exc:
        raise ModelNotFound(
            f"{exc}\nHint: run `python scripts/download_models.py` "
            "and ensure `pip install -r requirements.txt` completed."
        ) from exc
