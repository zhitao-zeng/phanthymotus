"""Face MCP/ROS2 lifecycle tests without loading model files."""

from __future__ import annotations

import json

from vision_stubs import _FakeCompressedImage, _FakeExecutor, _wait_until

import plugins.face_id.plugin as face_plugin
from plugins.face import FaceRecognitionPlugin, TOOLS


class _FakeFaceEngine:
    def __init__(self):
        self.inputs = []
        self.closed = False

    def infer_face_identity(self, image_bytes):
        self.inputs.append(image_bytes)
        return {
            "detect_confidence": 0.91,
            "bbox_relative": [0.1, 0.2, 0.3, 0.4],
            "identity": {"person_id": "n000001", "confidence": 0.88},
        }

    def close(self):
        self.closed = True


class _Builder:
    def __init__(self):
        self.calls = 0
        self.engines = []

    def __call__(self, cfg):
        self.calls += 1
        engine = _FakeFaceEngine()
        engine.cfg = dict(cfg)
        self.engines.append(engine)
        return engine


def _plugin(monkeypatch):
    builder = _Builder()
    monkeypatch.setattr(face_plugin, "build_face_engine", builder)
    executor = _FakeExecutor()
    plugin = FaceRecognitionPlugin(
        {"backend": "tensorrt", "recognizer": "lvface"},
        executor,
    )
    return plugin, executor, builder


def test_face_tool_contract_keeps_judge_name():
    assert TOOLS[0]["name"] == "face"
    assert TOOLS[0]["multiInstance"] is True
    assert TOOLS[0]["topic_in"][0]["format"] == "image/jpeg"
    assert TOOLS[0]["topic_out"][0]["format"] == "data/json"


def test_constructor_and_info_do_not_load_models(monkeypatch):
    plugin, _executor, builder = _plugin(monkeypatch)
    info = plugin.dispatch("face", {"action": "info"})
    assert info["state"] == "idle"
    assert builder.calls == 0


def test_first_start_loads_once_and_publishes_schema(monkeypatch):
    plugin, executor, builder = _plugin(monkeypatch)
    result = plugin.dispatch(
        "face", {"action": "start", "input_topic": "/camera/image"}
    )
    assert result == {
        "state": "running",
        "input": "/camera/image",
        "output": "/camera/image/face",
    }
    assert builder.calls == 1
    node = executor.nodes[0]
    assert node.subscriptions[0].topic == "/camera/image"
    node.subscriptions[0].callback(_FakeCompressedImage(b"jpeg"))
    assert _wait_until(lambda: bool(node.publishers[0].messages))
    payload = json.loads(node.publishers[0].messages[-1])
    assert payload["identity"]["person_id"] == "n000001"
    assert builder.engines[0].inputs == [b"jpeg"]
    plugin.dispatch("face", {"action": "stop"})


def test_repeated_case_start_stop_reuses_engine(monkeypatch):
    plugin, executor, builder = _plugin(monkeypatch)
    node = None
    for _ in range(5):
        plugin.dispatch("face", {"action": "start", "input_topic": "/camera/image"})
        assert len(executor.nodes) == 1
        current = executor.nodes[0]
        node = current if node is None else node
        assert current is node
        assert len(current.subscriptions) == 1
        plugin.dispatch("face", {"action": "stop"})
        assert executor.nodes == [node]
        assert not node.worker_alive
    assert builder.calls == 1
    assert not builder.engines[0].closed


def test_repeated_lifecycle_stress_keeps_one_node_and_no_workers(monkeypatch):
    plugin, executor, builder = _plugin(monkeypatch)
    node = None
    for _ in range(500):
        plugin.dispatch("face", {"action": "start", "input_topic": "/camera/image"})
        current = executor.nodes[0]
        node = current if node is None else node
        assert current is node
        plugin.dispatch("face", {"action": "stop"})
        assert not current.worker_alive
    assert executor.nodes == [node]
    assert len(node.subscriptions) == 1
    assert node._workers == []
    assert builder.calls == 1


def test_model_config_change_closes_cached_engine(monkeypatch):
    plugin, executor, builder = _plugin(monkeypatch)
    plugin.dispatch("face", {"action": "start", "input_topic": "/camera/image"})
    first_node = executor.nodes[0]
    plugin.dispatch("face", {"action": "stop"})
    first = builder.engines[0]
    result = plugin.dispatch(
        "face", {"action": "config", "recognizer": "mobilefacenet"}
    )
    assert result == {"status": "configured", "engine_loaded": False, "reused": False}
    assert first.closed
    assert first_node.destroyed
    assert executor.nodes == []
    plugin.dispatch("face", {"action": "start", "input_topic": "/camera/image"})
    assert builder.calls == 2
    assert builder.engines[1].cfg["recognizer"] == "mobilefacenet"
    plugin.dispatch("face", {"action": "stop"})


def test_invalid_frame_publishes_no_face_instead_of_timing_out(monkeypatch):
    plugin, executor, builder = _plugin(monkeypatch)

    def fail(_image_bytes):
        raise ValueError("bad frame")

    plugin.dispatch("face", {"action": "start", "input_topic": "/camera/image"})
    builder.engines[0].infer_face_identity = fail
    node = executor.nodes[0]
    node.subscriptions[0].callback(_FakeCompressedImage(b"bad"))
    assert _wait_until(lambda: bool(node.publishers[0].messages))
    assert json.loads(node.publishers[0].messages[-1]) == {
        "detect_confidence": 0.0,
        "bbox_relative": None,
        "identity": None,
    }
    plugin.dispatch("face", {"action": "stop"})
