"""LVFace/MobileFaceNet embedding extraction."""

from __future__ import annotations

import cv2
import numpy as np

from .backends import InferenceBackend
from .schema import l2_normalize


class FaceRecognizer:
    """Extract normalized identity embeddings from aligned 112x112 BGR faces."""

    _SUPPORTED_MODELS = {"lvface", "mobilefacenet", "arcface"}

    def __init__(self, backend: InferenceBackend, *, model_type: str = "lvface"):
        normalized = str(model_type).strip().lower().replace("-", "")
        aliases = {
            "lvfacet": "lvface",
            "lvface": "lvface",
            "mobilefacenet": "mobilefacenet",
            "mbf": "mobilefacenet",
            "arcface": "arcface",
        }
        try:
            self.model_type = aliases[normalized]
        except KeyError as error:
            raise ValueError(f"unsupported face recognizer type: {model_type}") from error
        self.backend = backend

    def embed(self, aligned_face: np.ndarray, *, flip_tta: bool = False) -> np.ndarray:
        primary = self._embed_once(aligned_face)
        if not flip_tta:
            return primary
        flipped = self._embed_once(cv2.flip(aligned_face, 1))
        return l2_normalize(primary + flipped)

    def _embed_once(self, aligned_face: np.ndarray) -> np.ndarray:
        if aligned_face is None or aligned_face.shape != (112, 112, 3):
            raise ValueError(
                "face recognizer expects an aligned BGR image with shape (112, 112, 3)"
            )
        # LVFace's official ONNX example and InsightFace's ArcFace models use
        # the same RGB CHW normalization to [-1, 1].
        rgb = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
        tensor = np.transpose(rgb, (2, 0, 1)).astype(np.float32)
        tensor = ((tensor / 255.0) - 0.5) / 0.5
        outputs = self.backend.infer(np.ascontiguousarray(tensor[None]))
        if len(outputs) != 1:
            raise ValueError(
                f"face recognizer must expose one embedding output; got {len(outputs)}"
            )
        return l2_normalize(np.asarray(outputs[0]).reshape(-1))

    def close(self) -> None:
        self.backend.close()
