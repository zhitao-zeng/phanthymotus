import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from perception.plugins.obstacle_distance_core.contracts import (
    ErrorCode,
    ObstacleDistanceError,
)
from perception.plugins.obstacle_distance_core import hybrid_tensorrt_backends


class HybridTensorRTBackendTest(unittest.TestCase):
    def test_dav2_preprocessing_matches_fixed_indoor_engine_shape(self):
        image = np.zeros((480, 640, 3), dtype=np.uint8)

        tensor = hybrid_tensorrt_backends._prepare_dav2_image(image)

        self.assertEqual(tensor.shape, (1, 3, 518, 686))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertTrue(tensor.flags.c_contiguous)
        np.testing.assert_allclose(
            tensor[0, :, 0, 0],
            np.array(
                [
                    -0.485 / 0.229,
                    -0.456 / 0.224,
                    -0.406 / 0.225,
                ],
                dtype=np.float32,
            ),
            rtol=1e-6,
        )

    def test_dav2_preprocessing_rejects_incompatible_aspect_ratio(self):
        image = np.zeros((900, 1600, 3), dtype=np.uint8)

        with self.assertRaises(ObstacleDistanceError) as raised:
            hybrid_tensorrt_backends._prepare_dav2_image(image)

        self.assertEqual(raised.exception.code, ErrorCode.MODEL_ERROR)

    def test_factory_requires_three_existing_engine_files(self):
        with self.assertRaises(ObstacleDistanceError) as raised:
            hybrid_tensorrt_backends.create_backends({})

        self.assertEqual(raised.exception.code, ErrorCode.MODEL_ERROR)

    def test_factory_passes_validated_paths_to_both_backends(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            indoor = root / "indoor.engine"
            vehicle = root / "vehicle.engine"
            segmentation = root / "segmentation.engine"
            for path in (indoor, vehicle, segmentation):
                path.write_bytes(b"engine")

            depth_backend = object()
            segmentation_backend = object()
            with mock.patch.object(
                hybrid_tensorrt_backends,
                "HybridTensorRTDepthBackend",
                return_value=depth_backend,
            ) as depth_class, mock.patch.object(
                hybrid_tensorrt_backends,
                "YoloTensorRTSegBackend",
                return_value=segmentation_backend,
            ) as segmentation_class:
                result = hybrid_tensorrt_backends.create_backends(
                    {
                        "indoor_depth_engine": str(indoor),
                        "vehicle_depth_engine": str(vehicle),
                        "segmentation_engine": str(segmentation),
                    }
                )

        self.assertEqual(result, (depth_backend, segmentation_backend))
        depth_class.assert_called_once_with(str(indoor), str(vehicle))
        segmentation_class.assert_called_once_with(str(segmentation))


if __name__ == "__main__":
    unittest.main()
