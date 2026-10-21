"""ArcFace face embeddings for named-person recognition."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class ArcFaceEmbedder:
    """Align YuNet landmarks and extract normalized ArcFace embeddings.

    InsightFace supplies the ArcFace recognition ONNX model in its
    ``buffalo_l`` pack. The pack is downloaded to the project model directory
    on first use, keeping its files local to this checkout.
    """

    def __init__(self, model_pack: str = "buffalo_l", root: str | Path | None = None):
        try:
            from insightface.app import FaceAnalysis
            from insightface.utils import face_align
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "ArcFace recognition requires InsightFace; install project "
                "dependencies with `python -m pip install -r requirements.txt`."
            ) from exc

        self._face_align = face_align
        try:
            self._app = FaceAnalysis(
                name=model_pack,
                allowed_modules=["recognition"],
                providers=["CPUExecutionProvider"],
                root=str(root) if root is not None else None,
            )
            self._app.prepare(ctx_id=-1)
            self._model = self._app.models["recognition"]
        except Exception as exc:  # pragma: no cover - model/runtime dependent
            raise RuntimeError(
                f"Could not load ArcFace model pack {model_pack!r}: {exc}"
            ) from exc

    def embed(
        self, image: np.ndarray, landmarks: list[tuple[float, float]] | None
    ) -> np.ndarray | None:
        """Return a unit-length 512-D embedding for a five-point face."""
        if landmarks is None or len(landmarks) != 5:
            return None
        try:
            aligned = self._face_align.norm_crop(
                image,
                landmark=np.asarray(landmarks, dtype=np.float32),
                image_size=self._model.input_size[0],
            )
            embedding = np.asarray(self._model.get_feat(aligned), dtype=np.float32).ravel()
        except Exception:
            return None
        norm = float(np.linalg.norm(embedding))
        if not np.isfinite(norm) or norm <= 1e-9:
            return None
        return embedding / norm
