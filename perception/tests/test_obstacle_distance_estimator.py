import math
import sys
import types
import unittest
from dataclasses import FrozenInstanceError, fields
from unittest import mock

import numpy as np

from perception.plugins.obstacle_distance_core.backend_loader import (
    create_model_backends,
    load_backend_factory,
)
from perception.plugins.obstacle_distance_core.contracts import (
    DepthPrediction,
    ErrorCode,
    InstanceMask,
    ObstacleDistanceError,
    SceneDomain,
)
from perception.plugins.obstacle_distance_core.estimator import (
    DistanceResult,
    ObstacleDistanceEstimator,
)


CAMERA_TO_EGO = (
    0.0,
    0.0,
    1.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
)


def model_config(**changes):
    config = {
        "mode": "model",
        "decision_threshold_m": 1.0,
        "fallback_distance_m": 3.0,
        "soft_timeout_s": 2.0,
        "suffix_map": {".nyu": "indoor", ".kitti": "vehicle"},
        "indoor": {
            "roi": [0, 2, 0, 2],
            "min_depth_m": 0.3,
            "max_depth_m": 10.0,
            "percentile": 1.0,
            "min_valid_pixels": 1,
        },
        "vehicle": {
            "allowed_classes": ["car"],
            "min_confidence": 0.25,
            "percentile": 1.0,
            "min_depth_m": 0.3,
            "max_depth_m": 80.0,
            "allow_approximate_geometry": False,
            "camera_to_bumper_offset_m": 1.0,
            "calibration": {
                "fx": 1.0,
                "fy": 1.0,
                "cx": 0.0,
                "cy": 0.0,
                "camera_to_ego": CAMERA_TO_EGO,
                "bumper_xy": (1.0, 0.0),
            },
        },
    }
    config.update(changes)
    return config


class _Clock:
    def __init__(self, values):
        self._values = iter(values)

    def __call__(self):
        return next(self._values)


class _DepthBackend:
    def __init__(self, depth, *, error=None, after_call=None):
        self.depth = np.asarray(depth, dtype=np.float32)
        self.error = error
        self.after_call = after_call
        self.calls = []

    def predict_depth(self, image_bytes, domain, deadline_monotonic):
        self.calls.append((image_bytes, domain, deadline_monotonic))
        if self.after_call is not None:
            self.after_call()
        if self.error is not None:
            raise self.error
        return DepthPrediction(
            depth_m=self.depth,
            source_height=self.depth.shape[0],
            source_width=self.depth.shape[1],
        )


class _SegmentationBackend:
    def __init__(self, instances=(), *, error=None, after_call=None):
        self.instances = instances
        self.error = error
        self.after_call = after_call
        self.calls = []

    def predict_instances(self, image_bytes, deadline_monotonic):
        self.calls.append((image_bytes, deadline_monotonic))
        if self.after_call is not None:
            self.after_call()
        if self.error is not None:
            raise self.error
        return self.instances


class _NonCallableDepthBackend:
    predict_depth = 1


class _WrongSignatureDepthBackend:
    def predict_depth(self, image_bytes, domain):
        raise AssertionError("must not be called")


class _NonCallableSegmentationBackend:
    predict_instances = 1


class _WrongSignatureSegmentationBackend:
    def predict_instances(self, image_bytes):
        raise AssertionError("must not be called")


def target_instance(shape=(2, 2)):
    return InstanceMask("car", 0.9, np.ones(shape, dtype=bool))


class DistanceResultContractTest(unittest.TestCase):
    def test_fields_are_exact_and_result_is_frozen(self):
        self.assertEqual(
            [field.name for field in fields(DistanceResult)],
            [
                "distance_m",
                "near_obstacle",
                "decision_threshold_m",
                "scene",
                "status",
                "error_code",
                "fallback",
                "approximate_geometry",
                "latency_ms",
                "timestamp",
            ],
        )
        result = DistanceResult(
            1.0,
            False,
            1.0,
            "indoor",
            "ok",
            None,
            False,
            False,
            2.0,
            3.0,
        )
        with self.assertRaises(FrozenInstanceError):
            result.distance_m = 2.0


class EstimatorInitializationTest(unittest.TestCase):
    def test_model_mode_requires_both_protocol_backends(self):
        depth = _DepthBackend([[1.0]])
        segmentation = _SegmentationBackend()
        invalid_pairs = (
            (None, segmentation),
            (depth, None),
            (object(), segmentation),
            (depth, object()),
            (_NonCallableDepthBackend(), segmentation),
            (_WrongSignatureDepthBackend(), segmentation),
            (depth, _NonCallableSegmentationBackend()),
            (depth, _WrongSignatureSegmentationBackend()),
        )
        for depth_backend, segmentation_backend in invalid_pairs:
            with self.subTest(
                depth_backend=depth_backend,
                segmentation_backend=segmentation_backend,
            ):
                with self.assertRaises(ValueError):
                    ObstacleDistanceEstimator(
                        depth_backend,
                        segmentation_backend,
                        model_config(),
                    )

    def test_numeric_configuration_is_finite_non_boolean_and_in_range(self):
        invalid_values = {
            "decision_threshold_m": (0, -1, math.nan, math.inf, True, "1"),
            "fallback_distance_m": (-1, math.nan, math.inf, False, "3"),
            "soft_timeout_s": (0, -1, math.nan, math.inf, True, "2"),
        }
        depth = _DepthBackend([[1.0]])
        segmentation = _SegmentationBackend()
        for key, values in invalid_values.items():
            for value in values:
                with self.subTest(key=key, value=value):
                    with self.assertRaises(ValueError):
                        ObstacleDistanceEstimator(
                            depth,
                            segmentation,
                            model_config(**{key: value}),
                        )

    def test_configuration_is_defensively_copied(self):
        config = model_config()
        estimator = ObstacleDistanceEstimator(
            _DepthBackend([[2.0, 2.0], [2.0, 2.0]]),
            _SegmentationBackend(),
            config,
            monotonic=_Clock([0.0, 0.0, 0.1]),
            wall_time=lambda: 10.0,
        )
        config["indoor"]["roi"] = [0, 1, 0, 1]
        config["decision_threshold_m"] = 99.0

        result = estimator.estimate(b"image", scene_hint="indoor")

        self.assertEqual(result.distance_m, 2.0)
        self.assertEqual(result.decision_threshold_m, 1.0)

    def test_diagnostic_mode_requires_explicit_nonnegative_constant(self):
        for constant in (None, -1, math.nan, math.inf, True, "1"):
            config = model_config(mode="diagnostic_constant")
            if constant is not None:
                config["constant_distance_m"] = constant
            with self.subTest(constant=constant):
                with self.assertRaises(ValueError):
                    ObstacleDistanceEstimator(None, None, config)

    def test_unknown_mode_does_not_enable_diagnostic_behavior(self):
        with self.assertRaises(ValueError):
            ObstacleDistanceEstimator(
                None,
                None,
                model_config(mode="constant", constant_distance_m=0.5),
            )


class EstimatorInferenceTest(unittest.TestCase):
    def make_estimator(
        self,
        *,
        depth=None,
        segmentation=None,
        config=None,
        monotonic=None,
        wall_time=None,
    ):
        return ObstacleDistanceEstimator(
            depth or _DepthBackend([[2.0, 2.0], [2.0, 2.0]]),
            segmentation or _SegmentationBackend([target_instance()]),
            config or model_config(),
            monotonic=monotonic
            or _Clock([10.0, 10.0, 10.1, 10.1, 10.2, 10.2]),
            wall_time=wall_time or (lambda: 1000.0),
        )

    def test_indoor_uses_nyu_domain_and_roi_percentile(self):
        depth = _DepthBackend([[0.5, 2.0], [2.0, 2.0]])
        segmentation = _SegmentationBackend()
        result = self.make_estimator(
            depth=depth,
            segmentation=segmentation,
        ).estimate(b"rgb", scene_hint="indoor")

        self.assertAlmostEqual(result.distance_m, 0.545)
        self.assertEqual(depth.calls, [(b"rgb", SceneDomain.INDOOR, 12.0)])
        self.assertEqual(segmentation.calls, [])
        self.assertEqual(result.status, "ok")
        self.assertFalse(result.fallback)
        self.assertFalse(result.approximate_geometry)

    def test_vehicle_calls_depth_and_segmentation_with_same_deadline(self):
        depth = _DepthBackend([[3.0, 3.0], [3.0, 3.0]])
        segmentation = _SegmentationBackend([target_instance()])
        result = self.make_estimator(
            depth=depth,
            segmentation=segmentation,
        ).estimate(b"rgb", source_name="frame.KITTI")

        self.assertEqual(depth.calls, [(b"rgb", SceneDomain.VEHICLE, 12.0)])
        self.assertEqual(segmentation.calls, [(b"rgb", 12.0)])
        self.assertEqual(result.scene, "vehicle")
        self.assertEqual(result.distance_m, 2.0)
        self.assertFalse(result.approximate_geometry)

    def test_missing_calibration_requires_explicit_approximation(self):
        vehicle = dict(model_config()["vehicle"])
        vehicle["calibration"] = {}
        exact = self.make_estimator(
            config=model_config(vehicle=vehicle),
        ).estimate(b"rgb", scene_hint="vehicle")
        self.assertEqual(exact.error_code, ErrorCode.MISSING_CALIBRATION.value)
        self.assertTrue(exact.fallback)

        vehicle["allow_approximate_geometry"] = True
        approximate = self.make_estimator(
            config=model_config(vehicle=vehicle),
        ).estimate(b"rgb", scene_hint="vehicle")
        self.assertEqual(approximate.distance_m, 1.0)
        self.assertTrue(approximate.approximate_geometry)
        self.assertFalse(approximate.fallback)

    def test_invalid_calibration_never_falls_back_to_approximate_geometry(self):
        vehicle = dict(model_config()["vehicle"])
        vehicle["allow_approximate_geometry"] = True
        vehicle["calibration"] = {"fx": -1.0}
        result = self.make_estimator(
            config=model_config(vehicle=vehicle),
        ).estimate(b"rgb", scene_hint="vehicle")

        self.assertEqual(result.error_code, ErrorCode.INVALID_CALIBRATION.value)
        self.assertFalse(result.approximate_geometry)

    def test_near_threshold_is_strict(self):
        at_threshold = self.make_estimator(
            depth=_DepthBackend([[1.0]]),
            config=model_config(
                indoor={
                    "roi": [0, 1, 0, 1],
                    "min_depth_m": 0.0,
                    "max_depth_m": 10.0,
                    "percentile": 1.0,
                    "min_valid_pixels": 1,
                }
            ),
        ).estimate(b"rgb", scene_hint="indoor")
        below_threshold = self.make_estimator(
            depth=_DepthBackend([[0.999]]),
            config=model_config(
                indoor={
                    "roi": [0, 1, 0, 1],
                    "min_depth_m": 0.0,
                    "max_depth_m": 10.0,
                    "percentile": 1.0,
                    "min_valid_pixels": 1,
                }
            ),
        ).estimate(b"rgb", scene_hint="indoor")

        self.assertFalse(at_threshold.near_obstacle)
        self.assertTrue(below_threshold.near_obstacle)

    def test_empty_image_and_missing_scene_return_structured_fallbacks(self):
        depth = _DepthBackend([[1.0]])
        estimator = self.make_estimator(depth=depth)
        empty = estimator.estimate(b"", scene_hint="indoor")
        self.assertEqual(empty.error_code, ErrorCode.INVALID_IMAGE.value)
        self.assertEqual(empty.scene, "unknown")
        self.assertEqual(depth.calls, [])

        missing = self.make_estimator().estimate(b"rgb")
        self.assertEqual(missing.error_code, ErrorCode.MISSING_SCENE.value)
        self.assertEqual(missing.scene, "unknown")
        self.assertEqual(missing.status, "error")
        self.assertTrue(missing.fallback)
        self.assertEqual(missing.distance_m, 3.0)

    def test_domain_error_preserves_stable_code(self):
        depth = _DepthBackend(
            [[1.0]],
            error=ObstacleDistanceError(
                ErrorCode.NO_VALID_DEPTH,
                "safe detail",
            ),
        )
        result = self.make_estimator(depth=depth).estimate(
            b"rgb",
            scene_hint="indoor",
        )
        self.assertEqual(result.error_code, ErrorCode.NO_VALID_DEPTH.value)

    def test_model_exception_is_sanitized_and_not_reclassified_as_timeout(self):
        secret = "api-token-private"
        clock = _Clock([0.0, 0.0, 999.0])
        result = self.make_estimator(
            depth=_DepthBackend([[1.0]], error=RuntimeError(secret)),
            monotonic=clock,
        ).estimate(b"private-input", scene_hint="indoor")

        self.assertEqual(result.error_code, ErrorCode.MODEL_ERROR.value)
        self.assertNotIn(secret, repr(result))
        self.assertNotIn("private-input", repr(result))

    def test_invalid_final_distance_is_model_error_fallback(self):
        for value in (math.nan, math.inf, -0.1):
            with self.subTest(value=value):
                with mock.patch(
                    "perception.plugins.obstacle_distance_core.estimator."
                    "indoor_distance_m",
                    return_value=value,
                ):
                    result = self.make_estimator().estimate(
                        b"rgb",
                        scene_hint="indoor",
                    )
                self.assertEqual(
                    result.error_code,
                    ErrorCode.MODEL_ERROR.value,
                )
                self.assertTrue(result.fallback)

    def test_deadline_expired_before_depth_does_not_call_backend(self):
        depth = _DepthBackend([[1.0]])
        result = self.make_estimator(
            depth=depth,
            monotonic=_Clock([10.0, 12.0, 12.0]),
        ).estimate(b"rgb", scene_hint="indoor")

        self.assertEqual(result.error_code, ErrorCode.TIMEOUT.value)
        self.assertEqual(depth.calls, [])

    def test_deadline_expired_after_depth_skips_segmentation(self):
        depth = _DepthBackend([[3.0]])
        segmentation = _SegmentationBackend([target_instance((1, 1))])
        result = self.make_estimator(
            depth=depth,
            segmentation=segmentation,
            monotonic=_Clock([10.0, 10.0, 12.0, 12.0]),
        ).estimate(b"rgb", scene_hint="vehicle")

        self.assertEqual(result.error_code, ErrorCode.TIMEOUT.value)
        self.assertEqual(len(depth.calls), 1)
        self.assertEqual(segmentation.calls, [])

    def test_deadline_is_rechecked_immediately_before_segmentation(self):
        depth = _DepthBackend([[3.0]])
        segmentation = _SegmentationBackend([target_instance((1, 1))])
        result = self.make_estimator(
            depth=depth,
            segmentation=segmentation,
            monotonic=_Clock([10.0, 10.0, 10.5, 12.0, 12.0]),
        ).estimate(b"rgb", scene_hint="vehicle")

        self.assertEqual(result.error_code, ErrorCode.TIMEOUT.value)
        self.assertEqual(len(depth.calls), 1)
        self.assertEqual(segmentation.calls, [])

    def test_deadline_expired_after_segmentation_is_timeout(self):
        result = self.make_estimator(
            monotonic=_Clock([10.0, 10.0, 10.5, 10.5, 12.0, 12.0]),
        ).estimate(b"rgb", scene_hint="vehicle")
        self.assertEqual(result.error_code, ErrorCode.TIMEOUT.value)

    def test_latency_is_nonnegative_and_explicit_timestamp_wins(self):
        result = self.make_estimator(
            monotonic=_Clock([10.0, 10.0, 10.1, 9.0]),
            wall_time=lambda: 777.0,
        ).estimate(
            b"rgb",
            scene_hint="indoor",
            timestamp=123.5,
        )
        self.assertEqual(result.latency_ms, 0.0)
        self.assertEqual(result.timestamp, 123.5)

    def test_diagnostic_mode_is_explicit_and_skips_backends(self):
        config = model_config(
            mode="diagnostic_constant",
            constant_distance_m=0.5,
        )
        estimator = ObstacleDistanceEstimator(
            None,
            None,
            config,
            monotonic=_Clock([1.0, 1.1]),
            wall_time=lambda: 22.0,
        )
        result = estimator.estimate(b"rgb", scene_hint="indoor")

        self.assertEqual(result.distance_m, 0.5)
        self.assertTrue(result.near_obstacle)
        self.assertEqual(result.status, "diagnostic_constant")
        self.assertIsNone(result.error_code)
        self.assertFalse(result.fallback)
        self.assertEqual(result.timestamp, 22.0)


class BackendLoaderTest(unittest.TestCase):
    def setUp(self):
        self.module_name = "_obstacle_distance_test_factory"
        self.module = types.ModuleType(self.module_name)
        sys.modules[self.module_name] = self.module

    def tearDown(self):
        sys.modules.pop(self.module_name, None)

    def test_factory_path_must_be_strict_module_colon_attribute(self):
        for path in (
            "",
            "module",
            ":factory",
            "module:",
            "module:factory:extra",
            " module:factory",
            "module:factory ",
            123,
        ):
            with self.subTest(path=path):
                with self.assertRaises(ValueError):
                    load_backend_factory(path)

    def test_import_attribute_and_callable_errors_are_stable(self):
        with self.assertRaises(RuntimeError) as import_error:
            load_backend_factory("_missing_private_module:factory")
        self.assertNotIn("_missing_private_module", str(import_error.exception))

        with self.assertRaises(RuntimeError) as attribute_error:
            load_backend_factory(f"{self.module_name}:missing_secret_attribute")
        self.assertNotIn("missing_secret_attribute", str(attribute_error.exception))

        self.module.factory = object()
        with self.assertRaises(TypeError):
            load_backend_factory(f"{self.module_name}:factory")

    def test_dynamic_module_attribute_exception_is_sanitized(self):
        secret = "dynamic-module-secret"

        def fail_attribute(name):
            raise RuntimeError(secret)

        self.module.__getattr__ = fail_attribute
        with self.assertRaises(RuntimeError) as raised:
            load_backend_factory(f"{self.module_name}:dynamic_factory")

        self.assertNotIn(secret, str(raised.exception))
        self.assertNotIn("dynamic_factory", str(raised.exception))

    def test_loads_callable(self):
        self.module.factory = lambda config: (object(), object())
        self.assertIs(
            load_backend_factory(f"{self.module_name}:factory"),
            self.module.factory,
        )

    def test_diagnostic_mode_skips_factory_import(self):
        with mock.patch(
            "perception.plugins.obstacle_distance_core.backend_loader."
            "load_backend_factory",
            side_effect=AssertionError("must not import"),
        ):
            self.assertEqual(
                create_model_backends(
                    {
                        "mode": "diagnostic_constant",
                        "backend_factory": "private.module:factory",
                    }
                ),
                (None, None),
            )

    def test_model_factory_must_exist_and_return_exactly_two_protocols(self):
        with self.assertRaises(ValueError):
            create_model_backends({"mode": "model"})

        invalid_results = (
            None,
            (),
            (object(),),
            (object(), object()),
            (_NonCallableDepthBackend(), _SegmentationBackend()),
            (_WrongSignatureDepthBackend(), _SegmentationBackend()),
            (_DepthBackend([[1.0]]), _NonCallableSegmentationBackend()),
            (_DepthBackend([[1.0]]), _WrongSignatureSegmentationBackend()),
            (
                _DepthBackend([[1.0]]),
                _SegmentationBackend(),
                object(),
            ),
        )
        for index, result in enumerate(invalid_results):
            setattr(self.module, f"factory_{index}", lambda config, value=result: value)
            with self.subTest(result=result):
                with self.assertRaises((TypeError, RuntimeError)):
                    create_model_backends(
                        {
                            "mode": "model",
                            "backend_factory": (
                                f"{self.module_name}:factory_{index}"
                            ),
                        }
                    )

    def test_factory_exception_is_sanitized(self):
        secret = "password=do-not-leak"

        def factory(config):
            raise RuntimeError(secret)

        self.module.factory = factory
        with self.assertRaises(RuntimeError) as raised:
            create_model_backends(
                {
                    "mode": "model",
                    "backend_factory": f"{self.module_name}:factory",
                    "secret": secret,
                }
            )
        self.assertNotIn(secret, str(raised.exception))

    def test_success_returns_depth_and_segmentation_backends(self):
        expected = (_DepthBackend([[1.0]]), _SegmentationBackend())
        received = []

        def factory(config):
            received.append(config)
            return expected

        self.module.factory = factory
        config = {
            "mode": "model",
            "backend_factory": f"{self.module_name}:factory",
        }
        self.assertEqual(create_model_backends(config), expected)
        self.assertIs(received[0], config)


if __name__ == "__main__":
    unittest.main()
