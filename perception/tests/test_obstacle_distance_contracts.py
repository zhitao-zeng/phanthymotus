import inspect
import math
import unittest
from dataclasses import FrozenInstanceError

from perception.plugins import obstacle_distance_core
from perception.plugins.obstacle_distance_core import (
    CameraCalibration,
    DepthBackend,
    DepthPrediction,
    ErrorCode,
    InstanceMask,
    InstanceSegmentationBackend,
    ObstacleDistanceError,
    SceneDomain,
)
from perception.plugins.obstacle_distance_core.routing import resolve_scene


class SceneRoutingTest(unittest.TestCase):
    def test_scene_hint_takes_priority_over_suffix_and_fixed_scene(self):
        self.assertEqual(
            resolve_scene(
                scene_hint=SceneDomain.INDOOR,
                source_name="camera.jpg",
                suffix_map={".jpg": SceneDomain.VEHICLE},
                fixed_scene=SceneDomain.VEHICLE,
            ),
            SceneDomain.INDOOR,
        )

    def test_suffix_matching_is_case_insensitive(self):
        self.assertEqual(
            resolve_scene(
                source_name="frames/obstacle.JpG",
                suffix_map={".JPG": SceneDomain.VEHICLE},
                fixed_scene=SceneDomain.INDOOR,
            ),
            SceneDomain.VEHICLE,
        )

    def test_suffix_map_accepts_enum_and_string_values(self):
        cases = (
            (SceneDomain.INDOOR, SceneDomain.INDOOR),
            ("vehicle", SceneDomain.VEHICLE),
        )
        for configured, expected in cases:
            with self.subTest(configured=configured):
                self.assertEqual(
                    resolve_scene(
                        source_name="frame.png",
                        suffix_map={".png": configured},
                    ),
                    expected,
                )

    def test_fixed_scene_is_used_as_fallback(self):
        self.assertEqual(
            resolve_scene(
                source_name="frame.unknown",
                suffix_map={".png": "indoor"},
                fixed_scene="vehicle",
            ),
            SceneDomain.VEHICLE,
        )

    def test_source_without_suffix_skips_suffix_routing(self):
        self.assertEqual(
            resolve_scene(
                source_name="README",
                suffix_map={".png": SceneDomain.INDOOR},
                fixed_scene=SceneDomain.VEHICLE,
            ),
            SceneDomain.VEHICLE,
        )
        with self.assertRaises(ObstacleDistanceError) as raised:
            resolve_scene(
                source_name="README",
                suffix_map={".png": SceneDomain.INDOOR},
            )
        self.assertEqual(raised.exception.code, ErrorCode.MISSING_SCENE)

    def test_suffix_map_keys_must_be_nonempty_strings(self):
        source_name = ("private-source-name-" * 400) + ".png"
        for invalid_key in (None, ""):
            with self.subTest(invalid_key=invalid_key):
                with self.assertRaises(ObstacleDistanceError) as raised:
                    resolve_scene(
                        source_name=source_name,
                        suffix_map={invalid_key: SceneDomain.INDOOR},
                        fixed_scene=SceneDomain.VEHICLE,
                    )
                self.assertEqual(
                    raised.exception.code,
                    ErrorCode.MISSING_SCENE,
                )
                self.assertNotIn(source_name, str(raised.exception))

    def test_missing_scene_raises_stable_error(self):
        with self.assertRaises(ObstacleDistanceError) as raised:
            resolve_scene()

        self.assertEqual(raised.exception.code, ErrorCode.MISSING_SCENE)

    def test_invalid_scene_at_each_priority_level_does_not_fall_through(self):
        cases = (
            {
                "scene_hint": "warehouse",
                "fixed_scene": SceneDomain.INDOOR,
            },
            {
                "source_name": "frame.jpg",
                "suffix_map": {".jpg": "road"},
                "fixed_scene": SceneDomain.INDOOR,
            },
            {"fixed_scene": "road"},
        )
        for kwargs in cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ObstacleDistanceError) as raised:
                    resolve_scene(**kwargs)
                self.assertEqual(raised.exception.code, ErrorCode.MISSING_SCENE)

    def test_routing_errors_do_not_echo_large_inputs(self):
        source_name = ("private-image-bytes-" * 400) + ".jpg"
        invalid_scene = "untrusted-scene-value-" * 300

        with self.assertRaises(ObstacleDistanceError) as raised:
            resolve_scene(
                source_name=source_name,
                suffix_map={".jpg": invalid_scene},
            )

        message = str(raised.exception)
        self.assertNotIn(source_name, message)
        self.assertNotIn(invalid_scene, message)


class EnumAndErrorContractTest(unittest.TestCase):
    def test_scene_domain_values_are_stable(self):
        self.assertEqual(
            {member.name: member.value for member in SceneDomain},
            {"INDOOR": "indoor", "VEHICLE": "vehicle"},
        )

    def test_error_code_values_are_stable(self):
        self.assertEqual(
            {member.name: member.value for member in ErrorCode},
            {
                "INVALID_IMAGE": "invalid_image",
                "MISSING_SCENE": "missing_scene",
                "MODEL_ERROR": "model_error",
                "TIMEOUT": "timeout",
                "INVALID_DEPTH": "invalid_depth",
                "NO_VALID_DEPTH": "no_valid_depth",
                "NO_TARGET_INSTANCE": "no_target_instance",
                "MISSING_CALIBRATION": "missing_calibration",
                "INVALID_CALIBRATION": "invalid_calibration",
            },
        )

    def test_obstacle_distance_error_retains_code_and_readable_message(self):
        error = ObstacleDistanceError(ErrorCode.TIMEOUT, "depth model timed out")

        self.assertEqual(error.code, ErrorCode.TIMEOUT)
        self.assertEqual(error.message, "depth model timed out")
        self.assertIn("depth model timed out", str(error))


class DepthPredictionContractTest(unittest.TestCase):
    def test_valid_prediction_is_frozen_and_preserves_objects(self):
        depth = object()
        uncertainty = object()
        prediction = DepthPrediction(depth, 480, 640, uncertainty)

        self.assertIs(prediction.depth_m, depth)
        self.assertEqual((prediction.source_height, prediction.source_width), (480, 640))
        self.assertIs(prediction.uncertainty, uncertainty)
        with self.assertRaises(FrozenInstanceError):
            prediction.source_height = 720

    def test_depth_m_is_required(self):
        with self.assertRaises(ValueError):
            DepthPrediction(None, 480, 640)

    def test_source_dimensions_must_be_positive_integers(self):
        invalid_dimensions = (
            (0, 640),
            (-1, 640),
            (480, 0),
            (480, -1),
            (480.0, 640),
            (480, 640.0),
            (True, 640),
            (480, False),
        )
        for height, width in invalid_dimensions:
            with self.subTest(height=height, width=width):
                with self.assertRaises(ValueError):
                    DepthPrediction(object(), height, width)


class InstanceMaskContractTest(unittest.TestCase):
    def test_valid_instance_mask_is_frozen(self):
        mask = object()
        instance = InstanceMask("person", 0.75, mask)

        self.assertEqual(instance.class_name, "person")
        self.assertEqual(instance.confidence, 0.75)
        self.assertIs(instance.mask, mask)
        with self.assertRaises(FrozenInstanceError):
            instance.confidence = 0.5

    def test_class_name_must_be_nonempty(self):
        for class_name in ("", "   "):
            with self.subTest(class_name=class_name):
                with self.assertRaises(ValueError):
                    InstanceMask(class_name, 0.5, object())

    def test_confidence_must_be_finite_and_in_unit_interval(self):
        invalid_confidences = (
            -0.01,
            1.01,
            math.inf,
            -math.inf,
            math.nan,
            "0.5",
        )
        for confidence in invalid_confidences:
            with self.subTest(confidence=confidence):
                with self.assertRaises(ValueError):
                    InstanceMask("person", confidence, object())

    def test_oversized_integer_confidence_uses_validation_error(self):
        with self.assertRaises(ValueError):
            InstanceMask("person", 10**10000, object())

    def test_mask_is_required(self):
        with self.assertRaises(ValueError):
            InstanceMask("person", 0.5, None)


class CameraCalibrationContractTest(unittest.TestCase):
    def setUp(self):
        self.matrix = tuple(float(index) for index in range(16))
        self.bumper = (1.0, 0.25)

    def assert_invalid_calibration(self, **changes):
        values = {
            "fx": 500.0,
            "fy": 510.0,
            "cx": 320.0,
            "cy": 240.0,
            "camera_to_ego": self.matrix,
            "bumper_xy": self.bumper,
        }
        values.update(changes)
        with self.assertRaises(ObstacleDistanceError) as raised:
            CameraCalibration(**values)
        self.assertEqual(raised.exception.code, ErrorCode.INVALID_CALIBRATION)

    def test_valid_calibration_is_frozen(self):
        calibration = CameraCalibration(
            fx=500.0,
            fy=510.0,
            cx=320.0,
            cy=240.0,
            camera_to_ego=self.matrix,
            bumper_xy=self.bumper,
        )

        self.assertEqual(calibration.camera_to_ego, self.matrix)
        self.assertEqual(calibration.bumper_xy, self.bumper)
        with self.assertRaises(FrozenInstanceError):
            calibration.fx = 600.0

    def test_focal_lengths_must_be_positive_and_finite(self):
        for field in ("fx", "fy"):
            for value in (0.0, -1.0, math.inf, -math.inf, math.nan, "500"):
                with self.subTest(field=field, value=value):
                    self.assert_invalid_calibration(**{field: value})

    def test_principal_point_must_be_finite(self):
        for field in ("cx", "cy"):
            for value in (math.inf, -math.inf, math.nan, "320"):
                with self.subTest(field=field, value=value):
                    self.assert_invalid_calibration(**{field: value})

    def test_camera_to_ego_requires_exactly_sixteen_finite_values(self):
        invalid_matrices = (
            self.matrix[:-1],
            self.matrix + (16.0,),
            self.matrix[:-1] + (math.inf,),
            None,
        )
        for matrix in invalid_matrices:
            with self.subTest(matrix=matrix):
                self.assert_invalid_calibration(camera_to_ego=matrix)

    def test_bumper_requires_exactly_two_finite_values(self):
        invalid_bumpers = (
            (1.0,),
            (1.0, 2.0, 3.0),
            (1.0, math.nan),
            None,
        )
        for bumper in invalid_bumpers:
            with self.subTest(bumper=bumper):
                self.assert_invalid_calibration(bumper_xy=bumper)

    def test_oversized_integers_use_stable_calibration_error(self):
        oversized = 10**10000
        cases = (
            ("fx", {"fx": oversized}),
            ("camera_to_ego", {"camera_to_ego": self.matrix[:-1] + (oversized,)}),
            ("bumper_xy", {"bumper_xy": (1.0, oversized)}),
        )
        for field, changes in cases:
            with self.subTest(field=field):
                self.assert_invalid_calibration(**changes)


class BackendProtocolContractTest(unittest.TestCase):
    def test_structural_depth_backend_is_runtime_checkable(self):
        class FakeDepthBackend:
            def predict_depth(self, image_bytes, domain, deadline_monotonic):
                return DepthPrediction(object(), 1, 1)

        backend = FakeDepthBackend()
        self.assertIsInstance(backend, DepthBackend)
        self.assertIn(
            "deadline_monotonic",
            inspect.signature(backend.predict_depth).parameters,
        )
        self.assertIn(
            "deadline_monotonic",
            inspect.signature(DepthBackend.predict_depth).parameters,
        )

    def test_structural_instance_backend_is_runtime_checkable(self):
        class FakeInstanceBackend:
            def predict_instances(self, image_bytes, deadline_monotonic):
                return []

        backend = FakeInstanceBackend()
        self.assertIsInstance(backend, InstanceSegmentationBackend)
        self.assertIn(
            "deadline_monotonic",
            inspect.signature(backend.predict_instances).parameters,
        )
        self.assertIn(
            "deadline_monotonic",
            inspect.signature(
                InstanceSegmentationBackend.predict_instances
            ).parameters,
        )


class PublicApiContractTest(unittest.TestCase):
    def test_all_contains_only_stable_model_api(self):
        expected = {
            "SceneDomain",
            "ErrorCode",
            "ObstacleDistanceError",
            "DepthPrediction",
            "InstanceMask",
            "CameraCalibration",
            "DepthBackend",
            "InstanceSegmentationBackend",
        }

        self.assertEqual(set(obstacle_distance_core.__all__), expected)
        self.assertEqual(len(obstacle_distance_core.__all__), len(expected))
        for name in expected:
            self.assertIs(
                getattr(obstacle_distance_core, name),
                globals()[name],
            )
        self.assertFalse(hasattr(obstacle_distance_core, "resolve_scene"))


if __name__ == "__main__":
    unittest.main()
