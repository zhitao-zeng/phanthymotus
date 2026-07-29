import math
import unittest

import numpy as np

from perception.plugins.obstacle_distance_core.contracts import (
    ErrorCode,
    ObstacleDistanceError,
)
from perception.plugins.obstacle_distance_core.postprocess import (
    indoor_distance_m,
    scaled_roi,
    validate_depth_map,
)


FIXED_SOURCE_SIZE = (480, 640)
FIXED_ROI = (144, 432, 160, 480)


class FailingDepthConversion:
    def __float__(self):
        raise TypeError("private conversion detail")


class ValidateDepthMapTest(unittest.TestCase):
    def test_returns_two_dimensional_float32_array(self):
        result = validate_depth_map([[1, 2], [3, 4]])

        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_array_equal(
            result,
            np.array([[1, 2], [3, 4]], dtype=np.float32),
        )

    def test_rejects_non_two_dimensional_depth(self):
        invalid_depths = (
            1.0,
            [1.0, 2.0],
            [[[1.0]]],
        )
        for depth in invalid_depths:
            with self.subTest(depth=depth):
                with self.assertRaises(ObstacleDistanceError) as raised:
                    validate_depth_map(depth)
                self.assertEqual(raised.exception.code, ErrorCode.INVALID_DEPTH)

    def test_rejects_empty_depth(self):
        with self.assertRaises(ObstacleDistanceError) as raised:
            validate_depth_map(np.empty((0, 3), dtype=np.float32))

        self.assertEqual(raised.exception.code, ErrorCode.INVALID_DEPTH)

    def test_rejects_depth_without_finite_values(self):
        with self.assertRaises(ObstacleDistanceError) as raised:
            validate_depth_map([[math.nan, math.inf], [-math.inf, math.nan]])

        self.assertEqual(raised.exception.code, ErrorCode.INVALID_DEPTH)

    def test_conversion_failure_uses_safe_invalid_depth_error(self):
        depth = [[FailingDepthConversion()]]

        with self.assertRaises(ObstacleDistanceError) as raised:
            validate_depth_map(depth)

        self.assertEqual(raised.exception.code, ErrorCode.INVALID_DEPTH)
        self.assertNotIn("private conversion detail", str(raised.exception))
        self.assertLess(len(str(raised.exception)), 200)


class ScaledRoiTest(unittest.TestCase):
    def test_fixed_source_size_roi_is_unchanged_at_native_resolution(self):
        self.assertEqual(
            scaled_roi((480, 640), FIXED_SOURCE_SIZE, FIXED_ROI),
            (slice(144, 432), slice(160, 480)),
        )

    def test_non_native_resolution_uses_floor_for_start_and_ceil_for_end(self):
        self.assertEqual(
            scaled_roi(
                (7, 11),
                FIXED_SOURCE_SIZE,
                (101, 379, 123, 517),
            ),
            (slice(1, 6), slice(2, 9)),
        )

    def test_rejects_invalid_source_size(self):
        invalid_source_sizes = (
            None,
            (),
            (480,),
            (480, 640, 3),
            (0, 640),
            (-1, 640),
            (480.0, 640),
            (True, 640),
            (480, False),
        )
        for source_size in invalid_source_sizes:
            with self.subTest(source_size=source_size):
                with self.assertRaises(ObstacleDistanceError) as raised:
                    scaled_roi((480, 640), source_size, FIXED_ROI)
                self.assertEqual(raised.exception.code, ErrorCode.INVALID_DEPTH)

    def test_rejects_invalid_roi(self):
        invalid_rois = (
            None,
            (),
            (0, 1, 2),
            (0, 1, 2, 3, 4),
            (0.0, 1, 2, 3),
            (False, 1, 2, 3),
            (-1, 1, 2, 3),
            (1, 1, 2, 3),
            (2, 1, 2, 3),
            (0, 481, 2, 3),
            (0, 1, -1, 3),
            (0, 1, 3, 3),
            (0, 1, 4, 3),
            (0, 1, 2, 641),
        )
        for roi in invalid_rois:
            with self.subTest(roi=roi):
                with self.assertRaises(ObstacleDistanceError) as raised:
                    scaled_roi((480, 640), FIXED_SOURCE_SIZE, roi)
                self.assertEqual(raised.exception.code, ErrorCode.INVALID_DEPTH)

    def test_rejects_invalid_depth_shape(self):
        invalid_shapes = (
            None,
            (480,),
            (480, 640, 3),
            (0, 640),
            (480.0, 640),
            (True, 640),
        )
        for depth_shape in invalid_shapes:
            with self.subTest(depth_shape=depth_shape):
                with self.assertRaises(ObstacleDistanceError) as raised:
                    scaled_roi(depth_shape, FIXED_SOURCE_SIZE, FIXED_ROI)
                self.assertEqual(raised.exception.code, ErrorCode.INVALID_DEPTH)


class IndoorDistanceTest(unittest.TestCase):
    def call_distance(self, depth, **changes):
        config = {
            "source_size": FIXED_SOURCE_SIZE,
            "roi": FIXED_ROI,
            "min_depth_m": 0.1,
            "max_depth_m": 10.0,
            "percentile": 1.0,
            "min_valid_pixels": 1,
        }
        config.update(changes)
        return indoor_distance_m(depth, **config)

    def test_native_fixed_roi_ignores_global_minimum_outside_roi(self):
        depth = np.full(FIXED_SOURCE_SIZE, 8.0, dtype=np.float32)
        depth[FIXED_ROI[0] : FIXED_ROI[1], FIXED_ROI[2] : FIXED_ROI[3]] = 3.0
        depth[0, 0] = 0.1

        result = self.call_distance(depth)

        self.assertEqual(result, 3.0)

    def test_first_percentile_is_not_the_roi_minimum(self):
        depth = np.full((20, 20), 2.0, dtype=np.float32)
        depth[0, 0] = 0.1

        result = self.call_distance(
            depth,
            source_size=(20, 20),
            roi=(0, 20, 0, 20),
        )

        self.assertAlmostEqual(result, 2.0)
        self.assertGreater(result, 0.1)

    def test_scaled_roi_boundaries_drive_distance_selection(self):
        depth = np.full((7, 11), 9.0, dtype=np.float32)
        depth[1:6, 2:9] = 4.0

        result = self.call_distance(
            depth,
            roi=(101, 379, 123, 517),
            percentile=50,
        )

        self.assertEqual(result, 4.0)

    def test_filters_nonfinite_zero_and_out_of_range_with_closed_bounds(self):
        depth = np.array(
            [[math.nan, math.inf, -math.inf, 0.0, 0.49, 0.5, 1.0, 2.5, 2.51]],
            dtype=np.float32,
        )

        result = self.call_distance(
            depth,
            source_size=(1, 9),
            roi=(0, 1, 0, 9),
            min_depth_m=0.5,
            max_depth_m=2.5,
            percentile=50,
            min_valid_pixels=3,
        )

        self.assertEqual(result, 1.0)

    def test_too_few_valid_pixels_raises_no_valid_depth_without_array(self):
        depth = np.array([[1.0, 2.0, 99_999.0]], dtype=np.float32)

        with self.assertRaises(ObstacleDistanceError) as raised:
            self.call_distance(
                depth,
                source_size=(1, 3),
                roi=(0, 1, 0, 3),
                min_valid_pixels=3,
            )

        self.assertEqual(raised.exception.code, ErrorCode.NO_VALID_DEPTH)
        self.assertIn("2", str(raised.exception))
        self.assertNotIn("99999", str(raised.exception))
        self.assertNotIn("[", str(raised.exception))

    def test_rejects_invalid_percentile(self):
        for percentile in (-0.1, 100.1, math.nan, math.inf, -math.inf, "1", True):
            with self.subTest(percentile=percentile):
                with self.assertRaises(ObstacleDistanceError) as raised:
                    self.call_distance([[1.0]], percentile=percentile)
                self.assertEqual(raised.exception.code, ErrorCode.INVALID_DEPTH)

    def test_rejects_invalid_depth_range(self):
        invalid_ranges = (
            (-0.1, 10.0),
            (1.0, 1.0),
            (2.0, 1.0),
            (math.nan, 10.0),
            (0.0, math.nan),
            (0.0, math.inf),
            (-math.inf, 10.0),
            ("0", 10.0),
            (0.0, "10"),
            (False, 10.0),
            (0.0, True),
        )
        for min_depth_m, max_depth_m in invalid_ranges:
            with self.subTest(
                min_depth_m=min_depth_m,
                max_depth_m=max_depth_m,
            ):
                with self.assertRaises(ObstacleDistanceError) as raised:
                    self.call_distance(
                        [[1.0]],
                        min_depth_m=min_depth_m,
                        max_depth_m=max_depth_m,
                    )
                self.assertEqual(raised.exception.code, ErrorCode.INVALID_DEPTH)

    def test_rejects_invalid_min_valid_pixels(self):
        for min_valid_pixels in (0, -1, 1.0, True, False, "1"):
            with self.subTest(min_valid_pixels=min_valid_pixels):
                with self.assertRaises(ObstacleDistanceError) as raised:
                    self.call_distance(
                        [[1.0]],
                        min_valid_pixels=min_valid_pixels,
                    )
                self.assertEqual(raised.exception.code, ErrorCode.INVALID_DEPTH)

    def test_invalid_source_size_and_roi_propagate_invalid_depth(self):
        invalid_configs = (
            {"source_size": (True, 640)},
            {"roi": (144, 432, 160, False)},
        )
        for changes in invalid_configs:
            with self.subTest(changes=changes):
                with self.assertRaises(ObstacleDistanceError) as raised:
                    self.call_distance([[1.0]], **changes)
                self.assertEqual(raised.exception.code, ErrorCode.INVALID_DEPTH)

    def test_returns_python_float(self):
        result = self.call_distance(
            np.array([[1.25, 2.25]], dtype=np.float32),
            source_size=(1, 2),
            roi=(0, 1, 0, 2),
            percentile=50,
        )

        self.assertIs(type(result), float)


if __name__ == "__main__":
    unittest.main()
