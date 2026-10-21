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
            from insightface.model_zoo import get_model
            from insightface.utils.storage import ensure_available
            from insightface.utils import face_align
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "ArcFace recognition requires InsightFace; install project "
                "dependencies with `python -m pip install -r requirements.txt`."
            ) from exc

        self._face_align = face_align
        try:
            model_dir = Path(
                ensure_available(
                    "models",
                    model_pack,
                    root=str(root) if root is not None else "~/.insightface",
                )
            )
            model_path = model_dir / "w600k_r50.onnx"
            if not model_path.is_file():
                raise FileNotFoundError(
                    f"ArcFace recognition model not found: {model_path}"
                )
            self._model = get_model(
                str(model_path), providers=["CPUExecutionProvider"]
            )
            if self._model is None:
                raise RuntimeError(f"InsightFace did not recognize {model_path}")
            self._model.prepare(ctx_id=-1)
        except Exception as exc:  # pragma: no cover - model/runtime dependent
            raise RuntimeError(
                f"Could not load ArcFace model pack {model_pack!r}: "
                f"{type(exc).__name__}: {exc!r}"
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
