import math
import unittest
import warnings
from unittest import mock

import numpy as np

from perception.plugins.obstacle_distance_core import geometry
from perception.plugins.obstacle_distance_core.contracts import (
    CameraCalibration,
    ErrorCode,
    InstanceMask,
    ObstacleDistanceError,
)
from perception.plugins.obstacle_distance_core.geometry import (
    approximate_vehicle_distance_m,
    vehicle_distance_m,
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


def calibration(
    *,
    camera_to_ego=CAMERA_TO_EGO,
    bumper_xy=(1.0, 0.0),
):
    return CameraCalibration(
        fx=1.0,
        fy=1.0,
        cx=1.0,
        cy=1.0,
        camera_to_ego=camera_to_ego,
        bumper_xy=bumper_xy,
    )


def instance(
    mask,
    *,
    class_name="car",
    confidence=0.9,
):
    return InstanceMask(class_name, confidence, mask)


class WarningArrayConversion:
    def __array__(self, dtype=None, copy=None):
        warnings.warn("private mask conversion warning", UserWarning)
        return np.ones((3, 3), dtype=bool)


class VehicleDistanceTest(unittest.TestCase):
    def call_distance(self, depth, *, instances, **changes):
        config = {
            "calibration": calibration(),
            "allowed_classes": {"car", "truck"},
            "min_confidence": 0.5,
            "percentile": 50.0,
            "min_depth_m": 0.5,
            "max_depth_m": 10.0,
        }
        config.update(changes)
        return vehicle_distance_m(depth, instances=instances, **config)

    def test_center_depth_maps_camera_forward_to_ego_x_from_bumper(self):
        depth = np.full((3, 3), 99.0, dtype=np.float32)
        depth[1, 1] = 3.0
        center = np.zeros((3, 3), dtype=bool)
        center[1, 1] = True
        dog = np.ones((3, 3), dtype=bool)

        result = self.call_distance(
            depth,
            instances=[
                instance(center),
                instance(dog, class_name="dog", confidence=1.0),
            ],
        )

        self.assertIs(type(result), float)
        self.assertEqual(result, 2.0)

    def test_combines_masks_and_filters_confidence_class_and_closed_depth_range(self):
        depth = np.array(
            [
                [0.5, 1.0, 2.0, 2.5],
                [9.0, 9.0, 9.0, 9.0],
            ],
            dtype=np.float32,
        )
        first = np.array(
            [[True, True, False, False], [False, False, False, False]]
        )
        second = np.array(
            [[False, False, True, True], [False, False, False, False]]
        )
        ignored_low_confidence = np.array(
            [[False, False, False, False], [True, False, False, False]]
        )
        ignored_class = np.array(
            [[False, False, False, False], [False, True, False, False]]
        )

        result = self.call_distance(
            depth,
            instances=[
                instance(first, class_name="car"),
                instance(second, class_name="truck"),
                instance(ignored_low_confidence, confidence=0.49),
                instance(ignored_class, class_name="dog", confidence=1.0),
            ],
            calibration=calibration(bumper_xy=(0.0, 0.0)),
            min_depth_m=0.5,
            max_depth_m=2.5,
            percentile=50,
        )

        expected = float(
            np.percentile(
                [
                    math.hypot(0.5, -0.5),
                    math.hypot(1.0, 0.0),
                    math.hypot(2.0, 2.0),
                    math.hypot(2.5, 5.0),
                ],
                50,
            )
        )
        self.assertAlmostEqual(result, expected)
        self.assertGreater(result, min(expected, 0.5))

    def test_first_percentile_of_many_pixels_is_not_minimum_distance(self):
        depth = np.full((200, 1), 5.0, dtype=np.float32)
        depth[0, 0] = 1.0
        target = np.ones(depth.shape, dtype=bool)
        simple_calibration = CameraCalibration(
            fx=1.0,
            fy=1.0,
            cx=0.0,
            cy=0.0,
            camera_to_ego=CAMERA_TO_EGO,
            bumper_xy=(0.0, 0.0),
        )

        result = self.call_distance(
            depth,
            instances=[instance(target)],
            calibration=simple_calibration,
            percentile=1.0,
        )

        self.assertAlmostEqual(result, 5.0)
        self.assertGreater(result, 1.0)
        self.assertNotEqual(result, float(np.min(depth)))

    def test_extrinsic_rotation_and_translation_change_horizontal_distance(self):
        depth = np.array([[2.0, 2.0]], dtype=np.float32)
        right_pixel = np.array([[False, True]], dtype=bool)
        identity = (
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
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        )
        rotated_translated = (
            0.0,
            0.0,
            1.0,
            3.0,
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

        identity_result = self.call_distance(
            depth,
            instances=[instance(right_pixel)],
            calibration=calibration(
                camera_to_ego=identity,
                bumper_xy=(0.0, 0.0),
            ),
        )
        transformed_result = self.call_distance(
            depth,
            instances=[instance(right_pixel)],
            calibration=calibration(
                camera_to_ego=rotated_translated,
                bumper_xy=(0.0, 0.0),
            ),
        )

        self.assertAlmostEqual(identity_result, math.hypot(0.0, 2.0))
        self.assertAlmostEqual(transformed_result, math.hypot(5.0, 0.0))
        self.assertNotEqual(identity_result, transformed_result)

    def test_rejects_invalid_selected_mask_shapes_and_dtypes(self):
        invalid_masks = (
            np.ones((2, 2), dtype=bool),
            np.ones(3, dtype=bool),
            np.ones((3, 3, 1), dtype=bool),
            np.ones((3, 3), dtype=np.uint8),
            np.ones((3, 3), dtype=object),
            np.ones((3, 3), dtype=np.complex64),
        )
        depth = np.ones((3, 3), dtype=np.float32)

        for mask in invalid_masks:
            with self.subTest(shape=mask.shape, dtype=mask.dtype):
                with self.assertRaises(ObstacleDistanceError) as raised:
                    self.call_distance(depth, instances=[instance(mask)])
                self.assertEqual(raised.exception.code, ErrorCode.INVALID_DEPTH)

    def test_no_target_and_no_valid_depth_use_distinct_codes(self):
        depth = np.full((3, 3), 3.0, dtype=np.float32)
        mask = np.ones((3, 3), dtype=bool)

        with self.assertRaises(ObstacleDistanceError) as no_target:
            self.call_distance(
                depth,
                instances=[instance(mask, class_name="dog")],
            )
        with self.assertRaises(ObstacleDistanceError) as no_valid:
            self.call_distance(
                depth,
                instances=[instance(mask)],
                min_depth_m=4.0,
                max_depth_m=5.0,
            )

        self.assertEqual(no_target.exception.code, ErrorCode.NO_TARGET_INSTANCE)
        self.assertEqual(no_valid.exception.code, ErrorCode.NO_VALID_DEPTH)

    def test_accepts_instance_generator(self):
        mask = np.ones((1, 1), dtype=bool)
        instances = (value for value in [instance(mask)])

        result = self.call_distance([[3.0]], instances=instances)

        self.assertAlmostEqual(result, math.hypot(2.0, -3.0))

    def test_empty_instance_generator_raises_no_target(self):
        instances = (value for value in ())

        with self.assertRaises(ObstacleDistanceError) as raised:
            self.call_distance([[3.0]], instances=instances)

        self.assertEqual(raised.exception.code, ErrorCode.NO_TARGET_INSTANCE)

    def test_instance_generator_runtime_error_is_safe_invalid_depth(self):
        secret = "private iterator failure detail"
        mask = np.ones((1, 1), dtype=bool)

        def failing_instances():
            yield instance(mask)
            raise RuntimeError(secret)

        with self.assertRaises(ObstacleDistanceError) as raised:
            self.call_distance([[3.0]], instances=failing_instances())

        self.assertEqual(raised.exception.code, ErrorCode.INVALID_DEPTH)
        self.assertNotIn(secret, str(raised.exception))

    def test_full_mask_geometry_is_processed_in_row_chunks(self):
        depth = np.full((9, 64), 3.0, dtype=np.float32)
        target = np.ones(depth.shape, dtype=bool)
        block_shapes = []
        original_nonzero = np.nonzero

        def recording_nonzero(block):
            block_shapes.append(block.shape)
            return original_nonzero(block)

        with (
            mock.patch.object(geometry, "_GEOMETRY_CHUNK_ROWS", 2),
            mock.patch.object(
                geometry.np,
                "nonzero",
                side_effect=recording_nonzero,
            ),
            mock.patch.object(
                geometry.np,
                "vstack",
                side_effect=AssertionError("full point matrix is forbidden"),
            ),
        ):
            result = self.call_distance(
                depth,
                instances=[instance(target)],
                calibration=calibration(bumper_xy=(0.0, 0.0)),
            )

        per_row = np.hypot(
            3.0,
            (np.arange(depth.shape[1], dtype=np.float64) - 1.0) * 3.0,
        )
        expected = float(np.percentile(np.tile(per_row, depth.shape[0]), 50.0))
        self.assertAlmostEqual(result, expected, places=5)
        self.assertGreater(len(block_shapes), 1)
        self.assertTrue(all(shape[0] <= 2 for shape in block_shapes))

    def test_missing_calibration_has_stable_code(self):
        with self.assertRaises(ObstacleDistanceError) as raised:
            self.call_distance(
                [[1.0]],
                instances=[instance([[True]])],
                calibration=None,
            )

        self.assertEqual(raised.exception.code, ErrorCode.MISSING_CALIBRATION)

    def test_rejects_non_rigid_camera_to_ego(self):
        invalid_matrices = (
            (
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
                1.0,
                0.0,
                0.0,
                0.0,
                1.0,
                1.0,
            ),
            (
                2.0,
                0.0,
                0.0,
                0.0,
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
                1.0,
            ),
            (
                -1.0,
                0.0,
                0.0,
                0.0,
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
                1.0,
            ),
        )

        for matrix in invalid_matrices:
            with self.subTest(matrix=matrix):
                with self.assertRaises(ObstacleDistanceError) as raised:
                    self.call_distance(
                        [[1.0]],
                        instances=[instance([[True]])],
                        calibration=calibration(camera_to_ego=matrix),
                    )
                self.assertEqual(
                    raised.exception.code,
                    ErrorCode.INVALID_CALIBRATION,
                )

    def test_invalid_calibration_object_has_stable_code(self):
        with self.assertRaises(ObstacleDistanceError) as raised:
            self.call_distance(
                [[1.0]],
                instances=[instance([[True]])],
                calibration=object(),
            )

        self.assertEqual(raised.exception.code, ErrorCode.INVALID_CALIBRATION)

    def test_rejects_invalid_configuration_values(self):
        cases = (
            ("allowed_classes", set()),
            ("allowed_classes", {"car", ""}),
            ("allowed_classes", {"car", 1}),
            ("allowed_classes", ["car"]),
            ("min_confidence", True),
            ("min_confidence", math.nan),
            ("min_confidence", math.inf),
            ("min_confidence", -0.01),
            ("min_confidence", 1.01),
            ("percentile", False),
            ("percentile", math.nan),
            ("percentile", -math.inf),
            ("percentile", -0.01),
            ("percentile", 100.01),
            ("min_depth_m", True),
            ("min_depth_m", math.nan),
            ("min_depth_m", -math.inf),
            ("min_depth_m", -0.01),
            ("max_depth_m", False),
            ("max_depth_m", math.nan),
            ("max_depth_m", math.inf),
            ("max_depth_m", 0.5),
        )
        mask = np.ones((1, 1), dtype=bool)

        for key, value in cases:
            with self.subTest(key=key, value=value):
                changes = {key: value}
                if key == "max_depth_m" and value == 0.5:
                    changes["min_depth_m"] = 0.5
                with self.assertRaises(ObstacleDistanceError) as raised:
                    self.call_distance(
                        [[1.0]],
                        instances=[instance(mask)],
                        **changes,
                    )
                self.assertEqual(raised.exception.code, ErrorCode.INVALID_DEPTH)

    def test_selected_mask_conversion_warning_is_suppressed(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = self.call_distance(
                np.ones((3, 3), dtype=np.float32),
                instances=[instance(WarningArrayConversion())],
            )

        self.assertIs(type(result), float)
        self.assertEqual(caught, [])

    def test_error_does_not_leak_large_depth_mask_or_configuration(self):
        secret = "private-depth-mask-" * 500
        bad_mask = np.array([[secret]], dtype=object)

        with self.assertRaises(ObstacleDistanceError) as raised:
            self.call_distance(
                [[1.0]],
                instances=[instance(bad_mask)],
                allowed_classes={"car", secret},
            )

        message = str(raised.exception)
        self.assertNotIn(secret, message)
        self.assertLess(len(message), 200)


class ApproximateVehicleDistanceTest(unittest.TestCase):
    def call_distance(self, depth, *, instances, **changes):
        config = {
            "allowed_classes": {"car"},
            "min_confidence": 0.5,
            "percentile": 50.0,
            "min_depth_m": 0.0,
            "max_depth_m": 10.0,
            "camera_to_bumper_offset_m": 1.0,
        }
        config.update(changes)
        return approximate_vehicle_distance_m(
            depth,
            instances=instances,
            **config,
        )

    def test_subtracts_offset_clamps_zero_and_uses_percentile(self):
        depth = np.array([[0.5, 2.0, 5.0]], dtype=np.float32)
        mask = np.ones((1, 3), dtype=bool)

        result = self.call_distance(
            depth,
            instances=[instance(mask)],
            percentile=50,
            camera_to_bumper_offset_m=1.0,
        )

        self.assertIs(type(result), float)
        self.assertEqual(result, 1.0)

    def test_rejects_invalid_offset(self):
        mask = np.ones((1, 1), dtype=bool)
        invalid_offsets = (
            True,
            math.nan,
            math.inf,
            -math.inf,
            -0.01,
            "1.0",
        )

        for offset in invalid_offsets:
            with self.subTest(offset=offset):
                with self.assertRaises(ObstacleDistanceError) as raised:
                    self.call_distance(
                        [[1.0]],
                        instances=[instance(mask)],
                        camera_to_bumper_offset_m=offset,
                    )
                self.assertEqual(raised.exception.code, ErrorCode.INVALID_DEPTH)

    def test_reuses_target_mask_and_depth_validation(self):
        mask = np.ones((1, 1), dtype=bool)

        with self.assertRaises(ObstacleDistanceError) as no_target:
            self.call_distance(
                [[1.0]],
                instances=[instance(mask, class_name="dog")],
            )
        with self.assertRaises(ObstacleDistanceError) as no_valid:
            self.call_distance(
                [[20.0]],
                instances=[instance(mask)],
            )

        self.assertEqual(no_target.exception.code, ErrorCode.NO_TARGET_INSTANCE)
        self.assertEqual(no_valid.exception.code, ErrorCode.NO_VALID_DEPTH)

    def test_full_mask_approximation_is_processed_in_row_chunks(self):
        depth = np.full((7, 32), 3.0, dtype=np.float32)
        target = np.ones(depth.shape, dtype=bool)
        block_shapes = []
        original_nonzero = np.nonzero

        def recording_nonzero(block):
            block_shapes.append(block.shape)
            return original_nonzero(block)

        with (
            mock.patch.object(geometry, "_GEOMETRY_CHUNK_ROWS", 2),
            mock.patch.object(
                geometry.np,
                "nonzero",
                side_effect=recording_nonzero,
            ),
        ):
            result = self.call_distance(
                depth,
                instances=[instance(target)],
            )

        self.assertEqual(result, 2.0)
        self.assertGreater(len(block_shapes), 1)
        self.assertTrue(all(shape[0] <= 2 for shape in block_shapes))


if __name__ == "__main__":
    unittest.main()
