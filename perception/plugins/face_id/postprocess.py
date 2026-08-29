"""Detection-box calibration and leaderboard payload conversion."""

from __future__ import annotations

import numpy as np

from .schema import FaceDetection, IdentityMatch, empty_face_payload


def calibrate_bbox(
    bbox: np.ndarray,
    image_shape: tuple[int, ...],
    *,
    x_scale: float = 1.0,
    y_scale: float = 1.0,
    y_shift: float = 0.0,
) -> np.ndarray:
    x1, y1, x2, y2 = np.asarray(bbox, dtype=np.float32).reshape(4)
    width = float(x2 - x1)
    height = float(y2 - y1)
    center_x = float((x1 + x2) * 0.5)
    center_y = float((y1 + y2) * 0.5 + y_shift * height)
    new_width = max(1.0, width * float(x_scale))
    new_height = max(1.0, height * float(y_scale))
    image_height, image_width = image_shape[:2]
    calibrated = np.array(
        [
            center_x - new_width * 0.5,
            center_y - new_height * 0.5,
            center_x + new_width * 0.5,
            center_y + new_height * 0.5,
        ],
        dtype=np.float32,
    )
    calibrated[[0, 2]] = np.clip(calibrated[[0, 2]], 0, image_width)
    calibrated[[1, 3]] = np.clip(calibrated[[1, 3]], 0, image_height)
    return calibrated


def normalized_xywh(bbox: np.ndarray, image_shape: tuple[int, ...]) -> list[float]:
    image_height, image_width = image_shape[:2]
    if image_height <= 0 or image_width <= 0:
        raise ValueError(f"invalid image shape: {image_shape}")
    x1, y1, x2, y2 = np.asarray(bbox, dtype=np.float32).reshape(4)
    values = [
        x1 / image_width,
        y1 / image_height,
        (x2 - x1) / image_width,
        (y2 - y1) / image_height,
    ]
    return [round(float(np.clip(value, 0.0, 1.0)), 6) for value in values]


def face_payload(
    detection: FaceDetection | None,
    match: IdentityMatch | None,
    image_shape: tuple[int, ...],
    *,
    x_scale: float = 1.0,
    y_scale: float = 1.0,
    y_shift: float = 0.0,
    match_confidence: float | None = None,
) -> dict:
    if detection is None:
        return empty_face_payload()
    bbox = calibrate_bbox(
        detection.bbox,
        image_shape,
        x_scale=x_scale,
        y_scale=y_scale,
        y_shift=y_shift,
    )
    identity = None
    if match is not None:
        identity = {
            "person_id": match.person_id,
            "confidence": round(
                float(match_confidence if match_confidence is not None else match.score),
                6,
            ),
        }
    return {
        "detect_confidence": round(float(np.clip(detection.score, 0.0, 1.0)), 6),
        "bbox_relative": normalized_xywh(bbox, image_shape),
        "identity": identity,
    }
