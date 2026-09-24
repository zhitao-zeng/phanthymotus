"""The DepthART TensorRT adapter and visual-depth startup contract."""

import sys
import types

import numpy as np

from vision_stubs import _FakeExecutor  # noqa: F401 — installs ROS fakes

import plugins.depthart_runtime as runtime
from plugins.visual_depth import VideoDepthPerceptionPlugin


def test_start_waits_for_model_before_registering_camera(monkeypatch):
    plugin = VideoDepthPerceptionPlugin({}, "test", _FakeExecutor())
    loaded = []

    def load_model():
        loaded.append(True)
        plugin._model = object()

    monkeypatch.setattr(plugin, "_ensure_model", load_model)
    result = plugin.dispatch("visual_depth", {"action": "start", "input_topic": "/cam/rgb"})
    assert loaded == [True]
    assert result["state"] == "running"
    assert len(plugin._nodes) == 1


def test_trt_plugin_loads_before_engine_and_preserves_vision_contract(tmp_path, monkeypatch):
    engine = tmp_path / "depthart.engine"
    plugin = tmp_path / "scan.so"
    engine.write_bytes(b"engine-placeholder")
    plugin.write_bytes(b"plugin-placeholder")
    calls = []

    class Version:
        restype = None

        def __call__(self):
            return b"SelectiveScan-1"

    class Plugin:
        depthart_selective_scan_trt_version = Version()

    def load_plugin(path, mode):
        calls.append(("plugin", path))
        assert mode == runtime.ctypes.RTLD_GLOBAL
        return Plugin()

    class Engine:
        input_shape = (1, 3, 480, 864)
        input_dtype = np.float32
        output_names = ["depth"]

        def __init__(self, path):
            calls.append(("engine", path))

        def infer(self, blob):
            return [np.full((1, 480, 864), 2.0, dtype=np.float32)]

        def close(self):
            calls.append(("close", None))

    monkeypatch.setattr(runtime.ctypes, "CDLL", load_plugin)
    monkeypatch.setitem(sys.modules, "utils.tensorrt_runtime",
                        types.SimpleNamespace(TensorRTEngine=Engine))
    monkeypatch.setattr(runtime, "prepare_image",
                        lambda frame, dtype: np.zeros((1, 3, 480, 864), dtype=dtype))
    monkeypatch.setattr(runtime, "restore_align_corners",
                        lambda depth, size: np.full(size, 2.0, dtype=np.float32))

    session = runtime.DepthARTSession(engine, plugin)
    assert [kind for kind, _ in calls] == ["plugin", "engine"]
    outputs, meta = session.infer(np.zeros((1080, 1920, 3), dtype=np.uint8))
    assert meta is None and outputs[0].shape == (1080, 1920)
    assert outputs[0][0, 0] == 2.0
    session.close()
    assert calls[-1][0] == "close"
