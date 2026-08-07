import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from perception.plugins.obstacle_distance_core import zipdepth_tensorrt_backends
from perception.plugins.obstacle_distance_core.contracts import SceneDomain


class _Engine:
    def __init__(self, input_shape, output):
        self.input_shape = input_shape
        self.output_names = ["output"]
        self.output = output
        self.inputs = []

    def infer(self, tensor):
        self.inputs.append(tensor)
        return (self.output,)


def _indoor_config():
    return {
        "roi": [0, 4, 0, 4],
        "roi_reference_size": [4, 4],
        "inverse_depth_percentile": 95.0,
        "score_threshold": -0.1,
        "inverse_depth_distance_scale": -10.0,
        "inverse_depth_distance_bias_m": 3.0,
        "classification_margin_m": 0.001,
        "min_output_distance_m": 0.0,
        "max_output_distance_m": 10.0,
        "min_valid_pixels": 1,
    }


class ZipDepthTensorRTBackendTest(unittest.TestCase):
    def _backend(self, inverse_depth):
        indoor = _Engine((1, 3, 384, 512), inverse_depth)
        vehicle = _Engine((1, 3, 768, 768), np.ones((1, 4, 4), np.float32))
        backend = zipdepth_tensorrt_backends.ZipDepthYoloTensorRTDepthBackend(
            "indoor.engine",
            "vehicle.engine",
            indoor_config=_indoor_config(),
        )
        # Runtime engines are lazy-loaded.  Prediction tests inject lightweight
        # fakes directly so they stay independent of the local TensorRT install.
        backend._indoor = indoor
        backend._vehicle = vehicle
        return backend, indoor, vehicle

    def test_zipdepth_preprocessing_is_rgb_chw_and_normalized(self):
        image = np.zeros((4, 8, 3), dtype=np.uint8)
        image[:, :, 2] = 255

        tensor = zipdepth_tensorrt_backends._prepare_zipdepth_image(image)

        self.assertEqual(tensor.shape, (1, 3, 384, 512))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertTrue(tensor.flags.c_contiguous)
        np.testing.assert_array_equal(tensor[0, :, 0, 0], [1.0, 0.0, 0.0])

    def test_direct_indoor_decision_returns_continuous_distances(self):
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        deadline = time.monotonic() + 10.0

        positive, positive_engine, _ = self._backend(
            np.full((1, 1, 2, 2), 0.22, dtype=np.float32)
        )
        negative, _, _ = self._backend(
            np.full((1, 1, 2, 2), 0.05, dtype=np.float32)
        )
        with mock.patch.object(
            zipdepth_tensorrt_backends,
            "_decode_image",
            return_value=image,
        ):
            near = positive.predict_indoor_distance(b"image", deadline)
            far = negative.predict_indoor_distance(b"image", deadline)

        self.assertAlmostEqual(near, 0.8)
        self.assertAlmostEqual(far, 2.5)
        self.assertEqual(positive_engine.inputs[0].shape, (1, 3, 384, 512))

    def test_direct_indoor_decision_preserves_threshold_classification(self):
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        deadline = time.monotonic() + 10.0
        near, _, _ = self._backend(
            np.full((1, 1, 2, 2), 0.2, dtype=np.float32)
        )
        far, _, _ = self._backend(
            np.full((1, 1, 2, 2), 0.09, dtype=np.float32)
        )
        with mock.patch.object(
            zipdepth_tensorrt_backends,
            "_decode_image",
            return_value=image,
        ):
            near_distance = near.predict_indoor_distance(b"image", deadline)
            far_distance = far.predict_indoor_distance(b"image", deadline)

        self.assertEqual(near_distance, 0.999)
        self.assertAlmostEqual(far_distance, 2.1)

    def test_vehicle_prediction_keeps_metric_yolo_path(self):
        backend, _, vehicle = self._backend(
            np.full((1, 1, 2, 2), 0.2, dtype=np.float32)
        )
        image = np.zeros((4, 8, 3), dtype=np.uint8)
        with mock.patch.object(
            zipdepth_tensorrt_backends,
            "_decode_image",
            return_value=image,
        ):
            prediction = backend.predict_depth(
                b"image",
                SceneDomain.VEHICLE,
                time.monotonic() + 10.0,
            )

        self.assertEqual(prediction.depth_m.shape, (4, 8))
        self.assertEqual(vehicle.inputs[0].shape, (1, 3, 768, 768))

    def test_factory_builds_zipdepth_yolo_and_segmentation_backends(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / name for name in ("indoor.engine", "vehicle.engine", "seg.engine")]
            for path in paths:
                path.write_bytes(b"engine")
            depth = object()
            segmentation = object()
            with mock.patch.object(
                zipdepth_tensorrt_backends,
                "ZipDepthYoloTensorRTDepthBackend",
                return_value=depth,
            ) as depth_class, mock.patch.object(
                zipdepth_tensorrt_backends,
                "NativeTensorRTSegBackend",
                return_value=segmentation,
            ) as segmentation_class:
                result = zipdepth_tensorrt_backends.create_backends(
                    {
                        "indoor_depth_engine": str(paths[0]),
                        "vehicle_depth_engine": str(paths[1]),
                        "segmentation_engine": str(paths[2]),
                        "indoor": _indoor_config(),
                        "vehicle": {
                            "allowed_classes": ["person", "car"],
                            "min_confidence": 0.25,
                        },
                    }
                )

        self.assertEqual(result, (depth, segmentation))
        depth_class.assert_called_once_with(
            str(paths[0]),
            str(paths[1]),
            indoor_config=_indoor_config(),
            decision_threshold_m=1.0,
        )
        segmentation_class.assert_called_once_with(
            str(paths[2]),
            allowed_classes=["person", "car"],
            min_confidence=0.25,
        )


if __name__ == "__main__":
    unittest.main()
