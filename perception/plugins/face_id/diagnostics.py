"""Numeric face diagnostics that never include image content."""

from __future__ import annotations

import cv2
import numpy as np

from .alignment import ARCFACE_112_TEMPLATE, estimate_face_transform
from .schema import FaceDetection, IdentityMatch


def detection_quality(
    image: np.ndarray,
    detection: FaceDetection,
    aligned: np.ndarray,
) -> dict:
    """Return bounded geometry and image-quality signals for one face."""

    image_height, image_width = image.shape[:2]
    x1, y1, x2, y2 = [float(value) for value in detection.bbox]
    box_width = max(1.0, x2 - x1)
    box_height = max(1.0, y2 - y1)
    eye_distance = float(
        np.linalg.norm(detection.landmarks[0] - detection.landmarks[1])
    )
    matrix = estimate_face_transform(detection.landmarks)
    homogeneous = np.column_stack(
        [detection.landmarks, np.ones(5, dtype=np.float32)]
    )
    projected = homogeneous @ matrix.T
    alignment_rmse = float(
        np.sqrt(np.mean(np.sum((projected - ARCFACE_112_TEMPLATE) ** 2, axis=1)))
    )
    gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
    eye_midpoint = (detection.landmarks[0] + detection.landmarks[1]) * 0.5
    yaw_proxy = float(
        (detection.landmarks[2, 0] - eye_midpoint[0])
        / max(1e-6, eye_distance)
    )
    border_pixels = np.concatenate(
        [aligned[0], aligned[-1], aligned[:, 0], aligned[:, -1]], axis=0
    )
    return {
        "alignment_rmse": round(alignment_rmse, 4),
        "bbox_area_ratio": round(
            box_width * box_height / max(1.0, image_width * image_height), 6
        ),
        "bbox_relative_xyxy": [
            round(x1 / max(1.0, image_width), 6),
            round(y1 / max(1.0, image_height), 6),
            round(x2 / max(1.0, image_width), 6),
            round(y2 / max(1.0, image_height), 6),
        ],
        "bbox_height_ratio": round(box_height / max(1.0, image_height), 6),
        "bbox_width_ratio": round(box_width / max(1.0, image_width), 6),
        "blur_variance": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 3),
        "border_black_fraction": round(
            float(np.mean(np.all(border_pixels == 0, axis=1))), 6
        ),
        "brightness_mean": round(float(np.mean(gray)), 3),
        "eye_distance_ratio": round(eye_distance / min(box_width, box_height), 6),
        "yaw_proxy": round(yaw_proxy, 6),
    }


def ranked_matches(matches: list[IdentityMatch]) -> list[dict]:
    return [
        {
            "centroid_score": round(float(match.centroid_score), 6),
            "person_id": match.person_id,
            "score": round(float(match.score), 6),
            "subcenter_score": round(float(match.subcenter_score), 6),
        }
        for match in matches
    ]
