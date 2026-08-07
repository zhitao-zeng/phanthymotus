import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from perception.plugins.obstacle_distance_core.contracts import SceneDomain
from perception.plugins.obstacle_distance_core.places365_router import (
    Places365SceneRouter,
    _prepare_places365_image,
)


class _TensorInfo:
    name = "input"


class _Session:
    def __init__(self, logits):
        self.logits = logits
        self.inputs = []

    def get_inputs(self):
        return [_TensorInfo()]

    def get_outputs(self):
        return [_TensorInfo()]

    def run(self, _outputs, feed):
        self.inputs.append(feed["input"])
        return [self.logits[None]]


class Places365RouterTest(unittest.TestCase):
    def test_preprocessing_preserves_aspect_and_center_crops(self):
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        image[:, :, 2] = 255

        tensor = _prepare_places365_image(image)

        self.assertEqual(tensor.shape, (1, 3, 224, 224))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertTrue(tensor.flags.c_contiguous)

    def test_top_five_io_vote_routes_from_pixels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels = root / "IO_places365.txt"
            labels.write_text(
                "\n".join(
                    f"class{index} {2 if index in (0, 1, 2) else 1}"
                    for index in range(365)
                ),
                encoding="utf-8",
            )
            logits = np.arange(365, dtype=np.float32) * -1
            session = _Session(logits)
            with mock.patch(
                "onnxruntime.InferenceSession",
                return_value=session,
            ):
                router = Places365SceneRouter(
                    str(root / "model.onnx"),
                    str(labels),
                    top_k=5,
                )
            with mock.patch(
                "perception.plugins.obstacle_distance_core.places365_router._decode_image",
                return_value=np.zeros((240, 320, 3), dtype=np.uint8),
            ):
                scene = router.predict(b"image")

        self.assertEqual(scene, SceneDomain.VEHICLE)
        self.assertEqual(session.inputs[0].shape, (1, 3, 224, 224))


if __name__ == "__main__":
    unittest.main()
