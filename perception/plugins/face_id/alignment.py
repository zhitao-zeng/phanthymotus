"""Five-point ArcFace/LVFace alignment."""

from __future__ import annotations

import cv2
import numpy as np


ARCFACE_112_TEMPLATE = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def estimate_face_transform(landmarks: np.ndarray) -> np.ndarray:
    points = np.asarray(landmarks, dtype=np.float32)
    if points.shape != (5, 2) or not np.all(np.isfinite(points)):
        raise ValueError("five finite facial landmarks are required for alignment")
    matrix, _ = cv2.estimateAffinePartial2D(
        points,
        ARCFACE_112_TEMPLATE,
        method=cv2.LMEDS,
    )
    if matrix is None or matrix.shape != (2, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("could not estimate a stable five-point face transform")
    return np.asarray(matrix, dtype=np.float32)


def align_face(image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("face alignment expects a BGR HWC image")
    matrix = estimate_face_transform(landmarks)
    aligned = cv2.warpAffine(
        image,
        matrix,
        (112, 112),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    if aligned.shape != (112, 112, 3):
        raise RuntimeError(f"unexpected aligned face shape: {aligned.shape}")
    return aligned
