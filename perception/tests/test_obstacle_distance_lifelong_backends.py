"""obstacle_distance 真实模型 backends（lifelong_yolo_backends）单元测试。

真实权重/依赖不在测试环境强制要求;只有模型目录存在时才跑加载冒烟测试。
"""

import unittest
from pathlib import Path
from unittest import mock

from perception.plugins.obstacle_distance_core import ErrorCode
from perception.plugins.obstacle_distance_core.contracts import (
    ObstacleDistanceError,
)
from perception.plugins.obstacle_distance_core.lifelong_yolo_backends import (
    _pick_model_file,
    create_backends,
)


class PickModelFileTest(unittest.TestCase):
    def test_missing_inputs_return_none(self):
        self.assertIsNone(_pick_model_file("", ("model.pt",)))
        self.assertIsNone(_pick_model_file(None, ("model.pt",)))
        self.assertIsNone(_pick_model_file("/no/such/dir", ("model.pt",)))

    def test_named_file_wins_over_fallback(self):
        tmp = Path(tempfile_path())
        (tmp / "NSK_int8.pth").write_bytes(b"x")
        (tmp / "other.pth").write_bytes(b"y")
        result = _pick_model_file(str(tmp), ("NSK_int8.pth", "NSK.pth.tar"))
        self.assertEqual(result, str(tmp / "NSK_int8.pth"))

    def test_fallback_to_any_checkpoint(self):
        tmp = Path(tempfile_path())
        (tmp / "weights.pth").write_bytes(b"y")
        result = _pick_model_file(str(tmp), ("NSK_int8.pth",))
        self.assertEqual(result, str(tmp / "weights.pth"))


class CreateBackendsTest(unittest.TestCase):
    def test_missing_models_raise_model_error(self):
        config = {
            "mode": "model",
            "backend_factory": (
                "plugins.obstacle_distance_core.lifelong_yolo_backends:"
                "create_backends"
            ),
            "depth_model_dir": "/no/such/depth",
            "segmentation_model_dir": "/no/such/seg",
        }
        with self.assertRaises(ObstacleDistanceError) as caught:
            create_backends(config)
        self.assertIs(caught.exception.code, ErrorCode.MODEL_ERROR)

    def test_loads_when_models_present(self):
        depth_dir = Path(tempfile_path("depth"))
        seg_dir = Path(tempfile_path("seg"))
        (depth_dir / "NSK_int8.pth").write_bytes(b"placeholder")
        (seg_dir / "model.pt").write_bytes(b"placeholder")
        config = {
            "depth_model_dir": str(depth_dir),
            "segmentation_model_dir": str(seg_dir),
        }
        with mock.patch(
            "perception.plugins.obstacle_distance_core."
            "lifelong_yolo_backends.LifelongDepthBackend"
        ) as depth_type, mock.patch(
            "perception.plugins.obstacle_distance_core."
            "lifelong_yolo_backends.YoloSegBackend"
        ) as seg_type:
            depth_type.return_value = object()
            seg_type.return_value = object()
            depth, segmentation = create_backends(config)
            self.assertIsNotNone(depth)
            self.assertIsNotNone(segmentation)
            depth_type.assert_called_once_with(str(depth_dir), None)
            seg_type.assert_called_once_with(str(seg_dir), None)


def tempfile_path(suffix: str = "") -> str:
    import tempfile

    directory = Path(tempfile.mkdtemp(prefix="obstacle-backends"))
    if suffix:
        directory = directory / suffix
        directory.mkdir(parents=True, exist_ok=True)
    return str(directory)


if __name__ == "__main__":
    unittest.main()
