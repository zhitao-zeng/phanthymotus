"""Small data types shared by the face-identification pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FaceDetection:
    """One SCRFD detection in original-image pixel coordinates."""

    bbox: np.ndarray
    score: float
    landmarks: np.ndarray

    def __post_init__(self) -> None:
        bbox = np.asarray(self.bbox, dtype=np.float32).reshape(-1)
        landmarks = np.asarray(self.landmarks, dtype=np.float32)
        if bbox.shape != (4,):
            raise ValueError(f"face bbox must have shape (4,), got {bbox.shape}")
        if landmarks.shape != (5, 2):
            raise ValueError(
                f"face landmarks must have shape (5, 2), got {landmarks.shape}"
            )
        if not np.all(np.isfinite(bbox)) or not np.all(np.isfinite(landmarks)):
            raise ValueError("face detection contains non-finite coordinates")
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            raise ValueError(f"face bbox is empty: {bbox.tolist()}")
        object.__setattr__(self, "bbox", bbox)
        object.__setattr__(self, "landmarks", landmarks)
        object.__setattr__(self, "score", float(self.score))


@dataclass(frozen=True)
class IdentityMatch:
    """Best gallery match and its component cosine scores."""

    person_id: str
    score: float
    centroid_score: float
    subcenter_score: float


def l2_normalize(vector: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= eps:
        raise ValueError("embedding has zero or non-finite norm")
    return np.ascontiguousarray(vector / norm, dtype=np.float32)


def empty_face_payload() -> dict:
    """Return the leaderboard's single-face representation for no detection."""

    return {
        "detect_confidence": 0.0,
        "bbox_relative": None,
        "identity": None,
    }
