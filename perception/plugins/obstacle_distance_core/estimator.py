from __future__ import annotations

import math
import time
from copy import deepcopy
from dataclasses import dataclass
from numbers import Real
from typing import Callable, Mapping

from .contracts import (
    CameraCalibration,
    DepthBackend,
    ErrorCode,
    InstanceSegmentationBackend,
    ObstacleDistanceError,
    SceneDomain,
)
from .geometry import approximate_vehicle_distance_m, vehicle_distance_m
from .postprocess import indoor_distance_m
from .routing import resolve_scene


@dataclass(frozen=True)
class DistanceResult:
    distance_m: float
    near_obstacle: bool
    decision_threshold_m: float
    scene: str
    status: str
    error_code: str | None
    fallback: bool
    approximate_geometry: bool
    latency_ms: float
    timestamp: float


def _finite_number(
    value: object,
    *,
    name: str,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be a finite number") from None
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be a finite number")
    if positive and converted <= 0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and converted < 0:
        raise ValueError(f"{name} must be nonnegative")
    return converted


def _invalid_calibration(message: str) -> ObstacleDistanceError:
    return ObstacleDistanceError(ErrorCode.INVALID_CALIBRATION, message)


def _camera_calibration(value: object) -> CameraCalibration:
    if isinstance(value, CameraCalibration):
        return value
    if not isinstance(value, Mapping):
        raise _invalid_calibration("camera calibration is invalid")
    try:
        return CameraCalibration(
            fx=value["fx"],
            fy=value["fy"],
            cx=value["cx"],
            cy=value["cy"],
            camera_to_ego=tuple(value["camera_to_ego"]),
            bumper_xy=tuple(value["bumper_xy"]),
        )
    except ObstacleDistanceError:
        raise
    except Exception:
        raise _invalid_calibration("camera calibration is invalid") from None


class ObstacleDistanceEstimator:
    def __init__(
        self,
        depth_backend: DepthBackend | None,
        segmentation_backend: InstanceSegmentationBackend | None,
        config: Mapping[str, object],
        *,
        monotonic: Callable[[], float] = time.monotonic,
        wall_time: Callable[[], float] = time.time,
    ) -> None:
        if not isinstance(config, Mapping):
            raise ValueError("estimator config must be a mapping")
        self._config = deepcopy(dict(config))
        self._monotonic = monotonic
        self._wall_time = wall_time
        self._decision_threshold_m = _finite_number(
            self._config.get("decision_threshold_m", 1.0),
            name="decision_threshold_m",
            positive=True,
        )
        self._fallback_distance_m = _finite_number(
            self._config.get("fallback_distance_m", 3.0),
            name="fallback_distance_m",
            nonnegative=True,
        )
        self._soft_timeout_s = _finite_number(
            self._config.get("soft_timeout_s", 2.5),
            name="soft_timeout_s",
            positive=True,
        )

        self._mode = self._config.get("mode", "model")
        if self._mode == "model":
            if not isinstance(depth_backend, DepthBackend):
                raise ValueError("model mode requires a depth backend")
            if not isinstance(
                segmentation_backend,
                InstanceSegmentationBackend,
            ):
                raise ValueError(
                    "model mode requires a segmentation backend"
                )
            self._constant_distance_m = None
        elif self._mode == "diagnostic_constant":
            if "constant_distance_m" not in self._config:
                raise ValueError(
                    "diagnostic_constant mode requires constant_distance_m"
                )
            self._constant_distance_m = _finite_number(
                self._config["constant_distance_m"],
                name="constant_distance_m",
                nonnegative=True,
            )
        else:
            raise ValueError("mode must be model or diagnostic_constant")

        self._depth_backend = depth_backend
        self._segmentation_backend = segmentation_backend

    def _latency_ms(self, started_monotonic: float) -> float:
        try:
            elapsed = (float(self._monotonic()) - started_monotonic) * 1000.0
        except Exception:
            return 0.0
        if not math.isfinite(elapsed) or elapsed < 0:
            return 0.0
        return elapsed

    def _result(
        self,
        *,
        distance_m: float,
        scene: str,
        status: str,
        error_code: str | None,
        fallback: bool,
        approximate_geometry: bool,
        started_monotonic: float,
        timestamp: float,
    ) -> DistanceResult:
        return DistanceResult(
            distance_m=distance_m,
            near_obstacle=distance_m < self._decision_threshold_m,
            decision_threshold_m=self._decision_threshold_m,
            scene=scene,
            status=status,
            error_code=error_code,
            fallback=fallback,
            approximate_geometry=approximate_geometry,
            latency_ms=self._latency_ms(started_monotonic),
            timestamp=timestamp,
        )

    def _fallback(
        self,
        *,
        code: ErrorCode,
        scene: str,
        started_monotonic: float,
        timestamp: float,
    ) -> DistanceResult:
        return self._result(
            distance_m=self._fallback_distance_m,
            scene=scene,
            status="error",
            error_code=code.value,
            fallback=True,
            approximate_geometry=False,
            started_monotonic=started_monotonic,
            timestamp=timestamp,
        )

    def _check_deadline(self, deadline_monotonic: float) -> None:
        if self._monotonic() >= deadline_monotonic:
            raise ObstacleDistanceError(
                ErrorCode.TIMEOUT,
                "obstacle distance inference timed out",
            )

    def _indoor_distance(self, prediction) -> float:
        config = self._config.get("indoor")
        if not isinstance(config, Mapping):
            raise ObstacleDistanceError(
                ErrorCode.INVALID_DEPTH,
                "indoor configuration is invalid",
            )
        return indoor_distance_m(
            prediction.depth_m,
            source_size=(
                prediction.source_height,
                prediction.source_width,
            ),
            roi=config.get("roi"),
            min_depth_m=config.get("min_depth_m"),
            max_depth_m=config.get("max_depth_m"),
            percentile=config.get("percentile"),
            min_valid_pixels=config.get("min_valid_pixels"),
        )

    def _vehicle_distance(self, prediction, instances) -> tuple[float, bool]:
        config = self._config.get("vehicle")
        if not isinstance(config, Mapping):
            raise ObstacleDistanceError(
                ErrorCode.INVALID_DEPTH,
                "vehicle configuration is invalid",
            )
        allowed_classes = config.get("allowed_classes")
        try:
            allowed = set(allowed_classes)
        except Exception:
            allowed = allowed_classes
        common = {
            "instances": instances,
            "allowed_classes": allowed,
            "min_confidence": config.get("min_confidence"),
            "percentile": config.get("percentile"),
            "min_depth_m": config.get("min_depth_m"),
            "max_depth_m": config.get("max_depth_m"),
        }

        calibration_value = config.get("calibration")
        calibration_missing = calibration_value is None or (
            isinstance(calibration_value, Mapping)
            and len(calibration_value) == 0
        )
        if calibration_missing:
            if config.get("allow_approximate_geometry") is not True:
                raise ObstacleDistanceError(
                    ErrorCode.MISSING_CALIBRATION,
                    "camera calibration is required for vehicle geometry",
                )
            distance = approximate_vehicle_distance_m(
                prediction.depth_m,
                camera_to_bumper_offset_m=config.get(
                    "camera_to_bumper_offset_m"
                ),
                **common,
            )
            return distance, True

        calibration = _camera_calibration(calibration_value)
        distance = vehicle_distance_m(
            prediction.depth_m,
            calibration=calibration,
            **common,
        )
        return distance, False

    @staticmethod
    def _validated_distance(value: object) -> float:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ObstacleDistanceError(
                ErrorCode.MODEL_ERROR,
                "estimated distance is invalid",
            )
        try:
            distance = float(value)
        except (TypeError, ValueError, OverflowError):
            raise ObstacleDistanceError(
                ErrorCode.MODEL_ERROR,
                "estimated distance is invalid",
            ) from None
        if not math.isfinite(distance) or distance < 0:
            raise ObstacleDistanceError(
                ErrorCode.MODEL_ERROR,
                "estimated distance is invalid",
            )
        return distance

    def estimate(
        self,
        image_bytes: bytes,
        scene_hint: SceneDomain | str | None = None,
        source_name: str | None = None,
        timestamp: float | None = None,
    ) -> DistanceResult:
        started_monotonic = float(self._monotonic())
        result_timestamp = (
            timestamp if timestamp is not None else float(self._wall_time())
        )
        deadline_monotonic = started_monotonic + self._soft_timeout_s
        scene = "unknown"
        try:
            if not isinstance(image_bytes, bytes) or not image_bytes:
                raise ObstacleDistanceError(
                    ErrorCode.INVALID_IMAGE,
                    "image bytes must be nonempty",
                )
            domain = resolve_scene(
                scene_hint=scene_hint,
                source_name=source_name,
                fixed_scene=self._config.get("fixed_scene"),
                suffix_map=self._config.get("suffix_map"),
            )
            scene = domain.value

            if self._mode == "diagnostic_constant":
                distance = self._validated_distance(
                    self._constant_distance_m
                )
                return self._result(
                    distance_m=distance,
                    scene=scene,
                    status="diagnostic_constant",
                    error_code=None,
                    fallback=False,
                    approximate_geometry=False,
                    started_monotonic=started_monotonic,
                    timestamp=result_timestamp,
                )

            self._check_deadline(deadline_monotonic)
            prediction = self._depth_backend.predict_depth(
                image_bytes,
                domain,
                deadline_monotonic,
            )
            self._check_deadline(deadline_monotonic)

            approximate_geometry = False
            if domain is SceneDomain.INDOOR:
                distance = self._indoor_distance(prediction)
            else:
                self._check_deadline(deadline_monotonic)
                instances = self._segmentation_backend.predict_instances(
                    image_bytes,
                    deadline_monotonic,
                )
                self._check_deadline(deadline_monotonic)
                distance, approximate_geometry = self._vehicle_distance(
                    prediction,
                    instances,
                )
            distance = self._validated_distance(distance)
            return self._result(
                distance_m=distance,
                scene=scene,
                status="ok",
                error_code=None,
                fallback=False,
                approximate_geometry=approximate_geometry,
                started_monotonic=started_monotonic,
                timestamp=result_timestamp,
            )
        except ObstacleDistanceError as error:
            return self._fallback(
                code=error.code,
                scene=scene,
                started_monotonic=started_monotonic,
                timestamp=result_timestamp,
            )
        except Exception:
            return self._fallback(
                code=ErrorCode.MODEL_ERROR,
                scene=scene,
                started_monotonic=started_monotonic,
                timestamp=result_timestamp,
            )
