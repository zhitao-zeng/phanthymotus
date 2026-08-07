import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from perception.plugins.obstacle_distance_core import native_tensorrt_backends
from perception.plugins.obstacle_distance_core.contracts import (
    ErrorCode,
    ObstacleDistanceError,
)


class NativeTensorRTBackendTest(unittest.TestCase):
    def test_read_engine_accepts_raw_and_ultralytics_formats(self):
        with tempfile.TemporaryDirectory() as directory:
            raw_path = Path(directory) / "raw.engine"
            raw_path.write_bytes(b"raw-tensorrt-engine")
            metadata_path = Path(directory) / "metadata.engine"
            metadata = {"task": "depth", "names": {"0": "depth"}}
            encoded = json.dumps(metadata).encode()
            metadata_path.write_bytes(
                len(encoded).to_bytes(4, "little", signed=True)
                + encoded
                + b"serialized"
            )

            raw_metadata, raw_engine = native_tensorrt_backends._read_engine(
                str(raw_path)
            )
            parsed_metadata, parsed_engine = native_tensorrt_backends._read_engine(
                str(metadata_path)
            )

        self.assertEqual(raw_metadata, {})
        self.assertEqual(raw_engine, b"raw-tensorrt-engine")
        self.assertEqual(parsed_metadata, metadata)
        self.assertEqual(parsed_engine, b"serialized")

    def test_prepare_yolo_image_letterboxes_and_normalizes(self):
        image = np.zeros((4, 8, 3), dtype=np.uint8)
        image[:, :, 2] = 255

        tensor, ratio, dw, dh = native_tensorrt_backends._prepare_yolo_image(
            image,
            8,
        )

        self.assertEqual(tensor.shape, (1, 3, 8, 8))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertTrue(tensor.flags.c_contiguous)
        self.assertEqual((ratio, dw, dh), (1.0, 0, 2))
        np.testing.assert_array_equal(tensor[0, :, 2, 0], [1.0, 0.0, 0.0])
        np.testing.assert_allclose(tensor[0, :, 0, 0], 114 / 255.0)

    def test_scale_depth_removes_letterbox_padding(self):
        depth = np.full((8, 8), 100.0, dtype=np.float32)
        depth[2:6] = 3.5

        scaled = native_tensorrt_backends._scale_depth_to_original(
            depth,
            original_height=4,
            original_width=8,
        )

        self.assertEqual(scaled.shape, (4, 8))
        np.testing.assert_allclose(scaled, 3.5)

    def test_process_yolo_masks_filters_confidence_and_restores_shape(self):
        detections = np.zeros((1, 3, 7), dtype=np.float32)
        detections[0, 0] = [0, 0, 640, 640, 0.9, 2, 1]
        detections[0, 1] = [0, 0, 640, 640, 0.01, 1, 1]
        prototypes = np.ones((1, 1, 160, 160), dtype=np.float32)

        results = native_tensorrt_backends._process_yolo_masks(
            detections,
            prototypes,
            original_height=4,
            original_width=8,
            ratio=1.0,
            dw=0,
            dh=2,
        )

        self.assertEqual(len(results), 1)
        class_id, confidence, mask = results[0]
        self.assertEqual(class_id, 2)
        self.assertAlmostEqual(confidence, 0.9)
        self.assertEqual(mask.shape, (4, 8))
        self.assertEqual(mask.dtype, np.bool_)
        self.assertTrue(mask.all())

    def test_process_yolo_masks_filters_before_allocating_masks(self):
        detections = np.zeros((1, 3, 7), dtype=np.float32)
        detections[0, 0] = [0, 0, 640, 640, 0.9, 2, 1]
        detections[0, 1] = [0, 0, 640, 640, 0.8, 7, 1]
        detections[0, 2] = [0, 0, 640, 640, 0.2, 2, 1]
        prototypes = np.ones((1, 1, 160, 160), dtype=np.float32)

        results = native_tensorrt_backends._process_yolo_masks(
            detections,
            prototypes,
            original_height=4,
            original_width=8,
            ratio=1.0,
            dw=0,
            dh=2,
            confidence_floor=0.25,
            allowed_class_ids=frozenset({2}),
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], 2)

    def test_factory_requires_three_existing_engine_files(self):
        with self.assertRaises(ObstacleDistanceError) as raised:
            native_tensorrt_backends.create_backends({})

        self.assertEqual(raised.exception.code, ErrorCode.MODEL_ERROR)

    def test_factory_passes_paths_to_native_backends(self):
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
                native_tensorrt_backends,
                "NativeTensorRTDepthBackend",
                return_value=depth_backend,
            ) as depth_class, mock.patch.object(
                native_tensorrt_backends,
                "NativeTensorRTSegBackend",
                return_value=segmentation_backend,
            ) as segmentation_class:
                result = native_tensorrt_backends.create_backends(
                    {
                        "indoor_depth_engine": str(indoor),
                        "vehicle_depth_engine": str(vehicle),
                        "segmentation_engine": str(segmentation),
                        "vehicle": {
                            "allowed_classes": ["person", "car"],
                            "min_confidence": 0.25,
                        },
                    }
                )

        self.assertEqual(result, (depth_backend, segmentation_backend))
        depth_class.assert_called_once_with(str(indoor), str(vehicle))
        segmentation_class.assert_called_once_with(
            str(segmentation),
            allowed_classes=["person", "car"],
            min_confidence=0.25,
        )


if __name__ == "__main__":
    unittest.main()
