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

_RESCUE_SUBSETS = (
    ("drop_left_eye", (1, 2, 3, 4)),
    ("drop_right_eye", (0, 2, 3, 4)),
    ("drop_nose", (0, 1, 3, 4)),
    ("drop_left_mouth", (0, 1, 2, 4)),
    ("drop_right_mouth", (0, 1, 2, 3)),
    ("eyes_only", (0, 1)),
)


def _validated_landmarks(landmarks: np.ndarray) -> np.ndarray:
    points = np.asarray(landmarks, dtype=np.float32)
    if points.shape != (5, 2) or not np.all(np.isfinite(points)):
        raise ValueError("five finite facial landmarks are required for alignment")
    return points


def _estimate_subset_transform(
    points: np.ndarray,
    indices: tuple[int, ...],
) -> np.ndarray:
    selected = np.asarray(indices, dtype=np.int32)
    matrix, _ = cv2.estimateAffinePartial2D(
        points[selected],
        ARCFACE_112_TEMPLATE[selected],
        method=cv2.LMEDS,
    )
    if matrix is None or matrix.shape != (2, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("could not estimate a stable five-point face transform")
    return np.asarray(matrix, dtype=np.float32)


def estimate_face_transform(landmarks: np.ndarray) -> np.ndarray:
    points = _validated_landmarks(landmarks)
    return _estimate_subset_transform(points, (0, 1, 2, 3, 4))


def alignment_rmse(landmarks: np.ndarray) -> float:
    points = _validated_landmarks(landmarks)
    matrix = estimate_face_transform(points)
    homogeneous = np.column_stack([points, np.ones(5, dtype=np.float32)])
    projected = homogeneous @ matrix.T
    return float(
        np.sqrt(np.mean(np.sum((projected - ARCFACE_112_TEMPLATE) ** 2, axis=1)))
    )


def _warp_face(image: np.ndarray, matrix: np.ndarray) -> np.ndarray:
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


def rescue_face_alignments(
    image: np.ndarray,
    landmarks: np.ndarray,
) -> list[tuple[str, np.ndarray]]:
    """Return alternate alignments that can ignore one corrupted landmark."""

    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("face alignment expects a BGR HWC image")
    points = _validated_landmarks(landmarks)
    candidates: list[tuple[str, np.ndarray]] = []
    baseline = estimate_face_transform(points)
    seen = {np.round(baseline, decimals=5).tobytes()}
    for name, indices in _RESCUE_SUBSETS:
        try:
            matrix = _estimate_subset_transform(points, indices)
        except ValueError:
            continue
        key = np.round(matrix, decimals=5).tobytes()
        if key in seen:
            continue
        seen.add(key)
        candidates.append((name, _warp_face(image, matrix)))
    return candidates


def align_face(image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("face alignment expects a BGR HWC image")
    matrix = estimate_face_transform(landmarks)
    return _warp_face(image, matrix)
