import importlib
import queue
import sys
import threading
import types
import unittest
from pathlib import Path
from unittest import mock


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class _FakeNode:
    pass


class _FakeQoSProfile:
    def __init__(self, *, reliability, history, depth, durability):
        self.reliability = reliability
        self.history = history
        self.depth = depth
        self.durability = durability


class _ReliabilityPolicy:
    BEST_EFFORT = "best_effort"
    RELIABLE = "reliable"


class _HistoryPolicy:
    KEEP_LAST = "keep_last"


class _DurabilityPolicy:
    VOLATILE = "volatile"


def _load_asr_module():
    rclpy = types.ModuleType("rclpy")
    rclpy_node = types.ModuleType("rclpy.node")
    rclpy_node.Node = _FakeNode
    rclpy_qos = types.ModuleType("rclpy.qos")
    rclpy_qos.QoSProfile = _FakeQoSProfile
    rclpy_qos.ReliabilityPolicy = _ReliabilityPolicy
    rclpy_qos.HistoryPolicy = _HistoryPolicy
    rclpy_qos.DurabilityPolicy = _DurabilityPolicy
    std_msgs = types.ModuleType("std_msgs")
    std_msgs_msg = types.ModuleType("std_msgs.msg")
    std_msgs_msg.String = type("String", (), {})
    modules = {
        "rclpy": rclpy,
        "rclpy.node": rclpy_node,
        "rclpy.qos": rclpy_qos,
        "std_msgs": std_msgs,
        "std_msgs.msg": std_msgs_msg,
    }
    sys.modules.pop("plugins.asr", None)
    with mock.patch.dict(sys.modules, modules):
        module = importlib.import_module("plugins.asr")
    return module


class _FakeVadSession:
    last_instance = None

    def __init__(self, **_kwargs):
        self.backend = "fake"
        self.reset_count = 0
        _FakeVadSession.last_instance = self

    def diagnostics(self):
        return {}

    def init(self):
        self.reset_count += 1

    def notify_idle(self, _now):
        return None

    def process_chunk(self, _pcm, _timestamp):
        return None

    def flush(self):
        return b""


class _FakeRunningNode:
    def __init__(self):
        self.state = "running"
        self.stop_calls = 0
        self.start_calls = 0

    def stop(self):
        self.stop_calls += 1
        self.state = "idle"

    def start(self):
        self.start_calls += 1
        self.state = "running"
        return {"state": "running"}


class AsrLifecycleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.asr = _load_asr_module()

    @classmethod
    def tearDownClass(cls):
        sys.modules.pop("plugins.asr", None)

    def test_input_qos_matches_reliable_stream_contract(self):
        qos = self.asr._ASR_INPUT_QOS
        self.assertEqual(qos.reliability, _ReliabilityPolicy.RELIABLE)
        self.assertEqual(qos.history, _HistoryPolicy.KEEP_LAST)
        self.assertEqual(qos.depth, 200)
        self.assertEqual(qos.durability, _DurabilityPolicy.VOLATILE)

    def test_vad_worker_acknowledges_pause_reset_and_resume(self):
        pcm_q = queue.Queue()
        result_q = queue.Queue()
        stop_evt = threading.Event()
        pause_evt = threading.Event()
        pause_ack_evt = threading.Event()
        resume_ack_evt = threading.Event()

        with mock.patch.object(self.asr, "VadSession", _FakeVadSession):
            worker = threading.Thread(
                target=self.asr._vad_worker,
                args=(
                    pcm_q,
                    result_q,
                    stop_evt,
                    "fake",
                    0.4,
                    400,
                    500,
                    "/unused",
                    {"trigger_mode": "vad"},
                    False,
                    10,
                    pause_evt,
                    pause_ack_evt,
                    resume_ack_evt,
                    7,
                ),
                daemon=True,
            )
            worker.start()
            self.assertTrue(resume_ack_evt.wait(timeout=1.0))

            pause_evt.set()
            self.assertTrue(pause_ack_evt.wait(timeout=2.0))
            self.assertFalse(resume_ack_evt.is_set())
            self.assertEqual(_FakeVadSession.last_instance.reset_count, 1)

            pause_evt.clear()
            self.assertTrue(resume_ack_evt.wait(timeout=1.0))
            self.assertFalse(pause_ack_evt.is_set())

            stop_evt.set()
            worker.join(timeout=2.0)
            self.assertFalse(worker.is_alive())

    def test_utterance_queue_accepts_legacy_and_kws_aware_items(self):
        legacy = self.asr._unpack_utterance_queue_item((b"pcm", 1.0, 2.0))
        kws_aware = self.asr._unpack_utterance_queue_item(
            (b"pcm", 1.0, 2.0, True)
        )

        self.assertEqual(legacy, (b"pcm", 1.0, 2.0, False))
        self.assertEqual(kws_aware, (b"pcm", 1.0, 2.0, True))

    def test_audio_contract_truncates_only_incomplete_pcm_sample(self):
        node = self.asr._ASRNode.__new__(self.asr._ASRNode)
        node._input_topic = "/audio"
        node._audio_contract_warns = {}

        with self.assertLogs("plugins.asr", level="WARNING"):
            pcm = node._check_audio_contract(
                "unexpected", b"\x01\x00" * 511 + b"\x01"
            )

        self.assertEqual(pcm, b"\x01\x00" * 511)
        self.assertEqual(node._audio_contract_warns, {
            "format": 1,
            "align": 1,
            "size": 1,
        })

    def test_non_model_config_restarts_running_nodes(self):
        plugin = self.asr.ASRPlugin.__new__(self.asr.ASRPlugin)
        node = _FakeRunningNode()
        adapter = object()
        plugin._state_lock = threading.Lock()
        plugin._loading = False
        plugin._load_error = None
        plugin._adapter = adapter
        plugin._plugin_cfg = {"mode": "offline", "language": "zh-CN"}
        plugin._mode = "offline"
        plugin._language = "zh-CN"
        plugin._asr_model = "paraformer-zh-en"
        plugin._vad_backend = "firered"
        plugin._vad_threshold = 0.4
        plugin._vad_silence_ms = 400
        plugin._vad_pre_roll_ms = 500
        plugin._vad_model_dir = "/models/vad"
        plugin._kws_cfg = {"trigger_mode": "vad", "enabled": False}
        plugin._save_vad_segments = False
        plugin._max_saved_segments = 1000
        plugin._nodes = {"case": node}

        result = plugin._dispatch_action(
            "config", {"language": "en-US"}, ""
        )

        self.assertEqual(result["status"], "configured")
        self.assertEqual(node.stop_calls, 1)
        self.assertEqual(node.start_calls, 1)
        self.assertEqual(node.state, "running")
        self.assertIs(node._adapter, adapter)
        self.assertEqual(node._language, "en-US")


if __name__ == "__main__":
    unittest.main()
