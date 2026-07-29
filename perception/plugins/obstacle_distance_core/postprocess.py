from __future__ import annotations

import math
from numbers import Integral, Real

import numpy as np

from .contracts import ErrorCode, ObstacleDistanceError


def _invalid_depth(message: str) -> ObstacleDistanceError:
    return ObstacleDistanceError(ErrorCode.INVALID_DEPTH, message)


def _integer_pair(value: object, name: str) -> tuple[int, int]:
    try:
        items = tuple(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise _invalid_depth(f"{name} must contain two positive integers") from None
    if len(items) != 2 or any(
        isinstance(item, bool) or not isinstance(item, Integral) or item <= 0
        for item in items
    ):
        raise _invalid_depth(f"{name} must contain two positive integers")
    return int(items[0]), int(items[1])


def _roi_bounds(
    value: object,
    source_height: int,
    source_width: int,
) -> tuple[int, int, int, int]:
    try:
        items = tuple(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise _invalid_depth("roi must contain four valid integer bounds") from None
    if len(items) != 4 or any(
        isinstance(item, bool) or not isinstance(item, Integral) for item in items
    ):
        raise _invalid_depth("roi must contain four valid integer bounds")

    row_start, row_end, col_start, col_end = (int(item) for item in items)
    if not (
        0 <= row_start < row_end <= source_height
        and 0 <= col_start < col_end <= source_width
    ):
        raise _invalid_depth("roi bounds must be ordered and within source_size")
    return row_start, row_end, col_start, col_end


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise _invalid_depth(f"{name} must be a finite number")
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError):
        raise _invalid_depth(f"{name} must be a finite number") from None
    if not math.isfinite(converted):
        raise _invalid_depth(f"{name} must be a finite number")
    return converted


def validate_depth_map(depth: object) -> np.ndarray:
    try:
        depth_array = np.asarray(depth, dtype=np.float32)
    except Exception:
        raise _invalid_depth("depth map must be convertible to a numeric array") from None

    if depth_array.ndim != 2:
        raise _invalid_depth("depth map must be two-dimensional")
    if depth_array.size == 0:
        raise _invalid_depth("depth map must not be empty")
    if not np.isfinite(depth_array).any():
        raise _invalid_depth("depth map must contain at least one finite value")
    return depth_array


def scaled_roi(
    depth_shape: object,
    source_size: object,
    roi: object,
) -> tuple[slice, slice]:
    depth_height, depth_width = _integer_pair(depth_shape, "depth_shape")
    source_height, source_width = _integer_pair(source_size, "source_size")
    row_start, row_end, col_start, col_end = _roi_bounds(
        roi,
        source_height,
        source_width,
    )

    scaled_row_start = max(
        0,
        min(depth_height, row_start * depth_height // source_height),
    )
    scaled_row_end = max(
        0,
        min(
            depth_height,
            (row_end * depth_height + source_height - 1) // source_height,
        ),
    )
    scaled_col_start = max(
        0,
        min(depth_width, col_start * depth_width // source_width),
    )
    scaled_col_end = max(
        0,
        min(
            depth_width,
            (col_end * depth_width + source_width - 1) // source_width,
        ),
    )

    if scaled_row_start >= scaled_row_end or scaled_col_start >= scaled_col_end:
        raise _invalid_depth("scaled roi must not be empty")
    return (
        slice(scaled_row_start, scaled_row_end),
        slice(scaled_col_start, scaled_col_end),
    )


def indoor_distance_m(
    depth_m: object,
    *,
    source_size: object,
    roi: object,
    min_depth_m: object,
    max_depth_m: object,
    percentile: object,
    min_valid_pixels: object,
) -> float:
    depth_array = validate_depth_map(depth_m)
    minimum = _finite_number(min_depth_m, "min_depth_m")
    maximum = _finite_number(max_depth_m, "max_depth_m")
    percentile_value = _finite_number(percentile, "percentile")

    if minimum < 0 or maximum <= minimum:
        raise _invalid_depth(
            "depth range must satisfy 0 <= min_depth_m < max_depth_m"
        )
    if not 0 <= percentile_value <= 100:
        raise _invalid_depth("percentile must be between 0 and 100")
    if (
        isinstance(min_valid_pixels, bool)
        or not isinstance(min_valid_pixels, Integral)
        or min_valid_pixels <= 0
    ):
        raise _invalid_depth("min_valid_pixels must be a positive integer")
    required_count = int(min_valid_pixels)

    row_slice, col_slice = scaled_roi(depth_array.shape, source_size, roi)
    roi_depth = depth_array[row_slice, col_slice]
    valid_mask = (
        np.isfinite(roi_depth)
        & (roi_depth >= minimum)
        & (roi_depth <= maximum)
    )
    valid_depth = roi_depth[valid_mask]
    valid_count = int(valid_depth.size)
    if valid_count < required_count:
        raise ObstacleDistanceError(
            ErrorCode.NO_VALID_DEPTH,
            f"found {valid_count} valid depth pixels; require at least {required_count}",
        )

    result = float(np.percentile(valid_depth, percentile_value))
    if not math.isfinite(result):
        raise _invalid_depth("depth percentile must be finite")
    return result
