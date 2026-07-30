from __future__ import annotations

import importlib
import importlib.util
import json
import math
import sys
import threading
import time
import types
import unittest
from pathlib import Path
from unittest import mock


PERCEPTION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PERCEPTION_ROOT))


def _ros_stubs() -> dict[str, types.ModuleType]:
    class FakeNode:
        def __init__(self, name):
            self.name = name
            self.publishers = []
            self.subscriptions = []
            self.destroy_calls = 0

        def create_publisher(self, message_type, topic, qos):
            publisher = mock.Mock(message_type=message_type, topic=topic, qos=qos)
            self.publishers.append(publisher)
            return publisher

        def create_subscription(self, message_type, topic, callback, qos):
            subscription = types.SimpleNamespace(
                message_type=message_type,
                topic=topic,
                callback=callback,
                qos=qos,
            )
            self.subscriptions.append(subscription)
            return subscription

        def destroy_node(self):
            self.destroy_calls += 1
            return True

    rclpy = types.ModuleType("rclpy")
    rclpy.node = types.ModuleType("rclpy.node")
    rclpy.node.Node = FakeNode
    rclpy.executors = types.ModuleType("rclpy.executors")
    rclpy.qos = types.ModuleType("rclpy.qos")
    rclpy.qos.QoSProfile = lambda **kwargs: kwargs
    rclpy.qos.ReliabilityPolicy = types.SimpleNamespace(RELIABLE="RELIABLE")
    rclpy.qos.HistoryPolicy = types.SimpleNamespace(KEEP_LAST="KEEP_LAST")
    rclpy.qos.DurabilityPolicy = types.SimpleNamespace(VOLATILE="VOLATILE")

    sensor_msgs = types.ModuleType("sensor_msgs")
    sensor_msgs.msg = types.ModuleType("sensor_msgs.msg")
    sensor_msgs.msg.CompressedImage = type("CompressedImage", (), {})
    std_msgs = types.ModuleType("std_msgs")
    std_msgs.msg = types.ModuleType("std_msgs.msg")
    std_msgs.msg.String = type("String", (), {})
    return {
        "rclpy": rclpy,
        "rclpy.node": rclpy.node,
        "rclpy.executors": rclpy.executors,
        "rclpy.qos": rclpy.qos,
        "sensor_msgs": sensor_msgs,
        "sensor_msgs.msg": sensor_msgs.msg,
        "std_msgs": std_msgs,
        "std_msgs.msg": std_msgs.msg,
    }


class _ModuleOverrides:
    _MISSING = object()

    def __init__(self, replacements):
        self.replacements = replacements
        self.originals = {}

    def start(self):
        for name, replacement in self.replacements.items():
            self.originals[name] = sys.modules.get(name, self._MISSING)
            sys.modules[name] = replacement
        return self

    def stop(self):
        for name, original in self.originals.items():
            if original is self._MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original
        self.originals.clear()

    def __enter__(self):
        return self.start()

    def __exit__(self, *_args):
        self.stop()


class ObstacleDistancePluginContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.modules = _ModuleOverrides(_ros_stubs())
        cls.modules.start()
        cls.original_plugin = sys.modules.pop(
            "plugins.obstacle_distance", _ModuleOverrides._MISSING
        )
        cls.plugin_module = importlib.import_module("plugins.obstacle_distance")

    @classmethod
    def tearDownClass(cls):
        sys.modules.pop("plugins.obstacle_distance", None)
        cls.modules.stop()
        if cls.original_plugin is not _ModuleOverrides._MISSING:
            sys.modules["plugins.obstacle_distance"] = cls.original_plugin

    def test_tool_contract_and_qos_match_image_processor_requirements(self):
        tool = self.plugin_module.TOOLS[0]

        self.assertEqual(tool["name"], "obstacle_distance")
        self.assertEqual(tool["type"], "processor")
        self.assertIs(tool["multiInstance"], True)
        self.assertEqual(
            tool["inputSchema"]["properties"]["action"]["enum"],
            ["start", "stop", "info", "config"],
        )
        self.assertEqual(
            tool["inputSchema"]["properties"]["scene_hint"]["enum"],
            ["indoor", "vehicle"],
        )
        self.assertEqual(
            tool["topic_in"],
            [{"format": "image/jpeg", "desc": "front camera image"}],
        )
        self.assertEqual(
            tool["topic_out"],
            [{"format": "data/json", "desc": "nearest obstacle distance"}],
        )
        self.assertEqual(self.plugin_module.ObstacleDistancePlugin.PREFIX, "obstacle_distance")
        self.assertEqual(self.plugin_module._CAMERA_QOS["reliability"], "RELIABLE")
        self.assertEqual(self.plugin_module._CAMERA_QOS["history"], "KEEP_LAST")
        self.assertEqual(self.plugin_module._CAMERA_QOS["depth"], 1)
        self.assertEqual(self.plugin_module._CAMERA_QOS["durability"], "VOLATILE")
        self.assertEqual(self.plugin_module._RESULT_QOS, self.plugin_module._CAMERA_QOS)

    def test_output_topic_is_derived_from_input_topic(self):
        self.assertEqual(
            self.plugin_module._obstacle_distance_output_topic("/camera/front"),
            "/camera/front/obstacle_distance",
        )


class _Estimator:
    def __init__(self, *, distance=2.0, entered=None, release=None, fail_once=False):
        self.distance = distance
        self.entered = entered
        self.release = release
        self.fail_once = fail_once
        self.calls = []

    def estimate(self, image_bytes, scene_hint=None, timestamp=None):
        self.calls.append((image_bytes, scene_hint, timestamp))
        if self.entered is not None:
            self.entered.set()
        if self.release is not None:
            self.release.wait(timeout=1)
        if self.fail_once:
            self.fail_once = False
            raise RuntimeError("private estimator failure")
        from plugins.obstacle_distance_core.estimator import DistanceResult

        distance = (
            float(image_bytes[-1] - ord("0"))
            if image_bytes and ord("0") <= image_bytes[-1] <= ord("9")
            else self.distance
        )
        return DistanceResult(
            distance_m=distance,
            near_obstacle=distance < 1.0,
            decision_threshold_m=1.0,
            scene=str(scene_hint),
            status="ok",
            error_code=None,
            fallback=False,
            approximate_geometry=False,
            latency_ms=0.1,
            timestamp=timestamp,
        )


def _wait_for(predicate, timeout=1.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        threading.Event().wait(0.005)
    return predicate()


class ObstacleDistanceNodeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.modules = _ModuleOverrides(_ros_stubs())
        cls.modules.start()
        cls.original_plugin = sys.modules.pop(
            "plugins.obstacle_distance", _ModuleOverrides._MISSING
        )
        cls.plugin_module = importlib.import_module("plugins.obstacle_distance")

    @classmethod
    def tearDownClass(cls):
        sys.modules.pop("plugins.obstacle_distance", None)
        cls.modules.stop()
        if cls.original_plugin is not _ModuleOverrides._MISSING:
            sys.modules["plugins.obstacle_distance"] = cls.original_plugin

    def _node(self, estimator=None, **kwargs):
        return self.plugin_module._ObstacleDistanceNode(
            "/camera/front",
            estimator or _Estimator(),
            "indoor",
            threading.Lock(),
            node_suffix="test",
            **kwargs,
        )

    def test_queue_is_bounded_and_busy_worker_keeps_latest_frame(self):
        entered = threading.Event()
        release = threading.Event()
        estimator = _Estimator(entered=entered, release=release)
        node = self._node(estimator)
        node.start()
        try:
            node._image_cb(types.SimpleNamespace(data=b"frame1"))
            self.assertTrue(entered.wait(timeout=1))
            node._image_cb(types.SimpleNamespace(data=b"frame2"))
            node._image_cb(types.SimpleNamespace(data=b"frame3"))
            self.assertEqual(node._frame_queue.maxsize, 1)
            queued, timestamp = node._frame_queue.get_nowait()
            self.assertEqual(queued, b"frame3")
            self.assertTrue(math.isfinite(timestamp))
            node._frame_queue.put_nowait((queued, timestamp))
            release.set()
            self.assertTrue(
                _wait_for(lambda: node._pub.publish.call_count == 2)
            )
        finally:
            release.set()
            node.stop()

        published = [
            json.loads(call.args[0].data)["distance_m"]
            for call in node._pub.publish.call_args_list
        ]
        self.assertEqual(published, [1.0, 3.0])

    def test_idle_callback_returns_without_copying_image_data(self):
        class Message:
            reads = 0

            @property
            def data(self):
                self.reads += 1
                return b"large-frame"

        node = self._node()
        message = Message()

        node._image_cb(message)

        self.assertEqual(message.reads, 0)
        self.assertTrue(node._frame_queue.empty())

    def test_shared_inference_lock_serializes_different_nodes(self):
        shared_lock = threading.Lock()
        state_lock = threading.Lock()
        state = {"active": 0, "maximum": 0, "calls": 0}
        both_done = threading.Event()

        class ConcurrentEstimator(_Estimator):
            def estimate(self, *args, **kwargs):
                with state_lock:
                    state["active"] += 1
                    state["maximum"] = max(state["maximum"], state["active"])
                threading.Event().wait(0.03)
                with state_lock:
                    state["active"] -= 1
                    state["calls"] += 1
                    if state["calls"] == 2:
                        both_done.set()
                return super().estimate(*args, **kwargs)

        first = self.plugin_module._ObstacleDistanceNode(
            "/camera/one",
            ConcurrentEstimator(),
            "indoor",
            shared_lock,
            node_suffix="one",
        )
        second = self.plugin_module._ObstacleDistanceNode(
            "/camera/two",
            ConcurrentEstimator(),
            "indoor",
            shared_lock,
            node_suffix="two",
        )
        first.start()
        second.start()
        try:
            first._image_cb(types.SimpleNamespace(data=b"one"))
            second._image_cb(types.SimpleNamespace(data=b"two"))
            self.assertTrue(both_done.wait(timeout=1))
        finally:
            first.stop()
            second.stop()

        self.assertEqual(state["maximum"], 1)

    def test_stop_refuses_restart_until_old_worker_has_exited(self):
        entered = threading.Event()
        release = threading.Event()
        node = self._node(_Estimator(entered=entered, release=release))
        node.start()
        node._image_cb(types.SimpleNamespace(data=b"old"))
        self.assertTrue(entered.wait(timeout=1))
        old_worker = node._worker_thread

        with mock.patch.object(old_worker, "join", return_value=None):
            node.stop()
        with self.assertRaisesRegex(RuntimeError, "worker is still stopping"):
            node.start()
        self.assertIs(node._worker_thread, old_worker)

        release.set()
        self.assertTrue(_wait_for(lambda: not old_worker.is_alive()))
        node.start()
        try:
            self.assertIsNot(node._worker_thread, old_worker)
        finally:
            node.stop()

    def test_stale_generation_never_publishes(self):
        entered = threading.Event()
        release = threading.Event()
        node = self._node(_Estimator(entered=entered, release=release))
        node.start()
        node._image_cb(types.SimpleNamespace(data=b"old"))
        self.assertTrue(entered.wait(timeout=1))
        old_worker = node._worker_thread
        with mock.patch.object(old_worker, "join", return_value=None):
            node.stop()
        release.set()
        self.assertTrue(_wait_for(lambda: not old_worker.is_alive()))
        self.assertEqual(node._pub.publish.call_count, 0)

    def test_estimator_exception_publishes_fallback_and_worker_survives(self):
        node = self._node(_Estimator(fail_once=True))
        node.start()
        try:
            with self.assertLogs(
                "plugins.obstacle_distance", level="ERROR"
            ):
                node._image_cb(types.SimpleNamespace(data=b"bad"))
                self.assertTrue(
                    _wait_for(lambda: node._pub.publish.call_count == 1)
                )
            node._image_cb(types.SimpleNamespace(data=b"good2"))
            self.assertTrue(
                _wait_for(lambda: node._pub.publish.call_count == 2)
            )
        finally:
            node.stop()

        payloads = [
            json.loads(call.args[0].data)
            for call in node._pub.publish.call_args_list
        ]
        self.assertEqual(payloads[0]["error_code"], "model_error")
        self.assertTrue(payloads[0]["fallback"])
        self.assertEqual(payloads[1]["distance_m"], 2.0)

    def test_result_pipeline_failure_drops_one_frame_and_worker_continues(self):
        for failure_point in ("serialize", "publish"):
            with self.subTest(failure_point=failure_point):
                node = self._node()
                if failure_point == "serialize":
                    original = node._serialized_result
                    effects = [
                        RuntimeError("secret serialization failure"),
                        lambda result, timestamp: original(result, timestamp),
                    ]

                    def serialize(result, timestamp):
                        effect = effects.pop(0)
                        if isinstance(effect, Exception):
                            raise effect
                        return effect(result, timestamp)

                    node._serialized_result = mock.Mock(side_effect=serialize)
                else:
                    node._pub.publish.side_effect = [
                        RuntimeError("secret publish failure"),
                        None,
                    ]

                node.start()
                try:
                    with self.assertLogs(
                        "plugins.obstacle_distance", level="ERROR"
                    ) as captured:
                        node._image_cb(types.SimpleNamespace(data=b"frame1"))
                        expected_calls = 0 if failure_point == "serialize" else 1
                        self.assertTrue(
                            _wait_for(
                                lambda: (
                                    node._serialized_result.call_count >= 1
                                    if failure_point == "serialize"
                                    else node._pub.publish.call_count >= expected_calls
                                )
                            )
                        )
                    self.assertNotIn("secret", "\n".join(captured.output))
                    node._image_cb(types.SimpleNamespace(data=b"frame2"))
                    expected_publishes = (
                        1 if failure_point == "serialize" else 2
                    )
                    self.assertTrue(
                        _wait_for(
                            lambda: (
                                node._pub.publish.call_count
                                >= expected_publishes
                            )
                        )
                    )
                    self.assertTrue(node.worker_alive)
                finally:
                    node.stop()

    def test_blocking_publisher_does_not_block_stop_state_transition(self):
        entered = threading.Event()
        release = threading.Event()
        node = self._node()

        def publish(_message):
            entered.set()
            release.wait(timeout=1)

        node._pub.publish.side_effect = publish
        node.start()
        node._image_cb(types.SimpleNamespace(data=b"frame"))
        self.assertTrue(entered.wait(timeout=1))
        worker = node._worker_thread
        try:
            with mock.patch.object(worker, "join", return_value=None):
                started = time.monotonic()
                status = node.stop()
                elapsed = time.monotonic() - started
        finally:
            release.set()
            self.assertTrue(_wait_for(lambda: not worker.is_alive()))

        self.assertLess(elapsed, 0.2)
        self.assertEqual(status["state"], "idle")
        self.assertTrue(node._stop_event.is_set())

    def test_worker_start_failure_rolls_back_node_state_without_secret(self):
        node = self._node()
        worker = mock.Mock()
        worker.start.side_effect = RuntimeError("secret worker failure")
        worker.is_alive.return_value = False

        with mock.patch.object(
            self.plugin_module.threading, "Thread", return_value=worker
        ):
            with self.assertRaises(RuntimeError) as raised:
                node.start()

        self.assertNotIn("secret", str(raised.exception))
        self.assertEqual(node.state, "idle")
        self.assertTrue(node._stop_event.is_set())
        self.assertIsNone(node._worker_thread)
        self.assertEqual(node._worker_threads, [])

    def test_callback_and_json_never_emit_nonfinite_values(self):
        node = self._node()
        node.start()
        try:
            stamp = types.SimpleNamespace(sec=math.inf, nanosec=math.nan)
            message = types.SimpleNamespace(
                data=bytearray(b"frame2"),
                header=types.SimpleNamespace(stamp=stamp),
            )
            with mock.patch.object(self.plugin_module.time, "time", return_value=math.inf):
                node._image_cb(message)
            self.assertTrue(
                _wait_for(lambda: node._pub.publish.call_count == 1)
            )
        finally:
            node.stop()

        raw = node._pub.publish.call_args.args[0].data
        self.assertNotIn("NaN", raw)
        self.assertNotIn("Infinity", raw)
        self.assertTrue(math.isfinite(json.loads(raw)["timestamp"]))

    def test_min_interval_uses_stop_event_wait(self):
        node = self._node(min_interval=0.1)
        original_wait = node._stop_event.wait
        with mock.patch.object(
            node._stop_event, "wait", wraps=original_wait
        ) as waited:
            node.start()
            active_event = node._stop_event
            original_active_wait = active_event.wait
            with mock.patch.object(
                active_event, "wait", wraps=original_active_wait
            ) as active_waited:
                node._image_cb(types.SimpleNamespace(data=b"frame2"))
                self.assertTrue(
                    _wait_for(lambda: node._pub.publish.call_count == 1)
                )
                self.assertTrue(
                    _wait_for(
                        lambda: any(
                            call.args and 0 < call.args[0] <= 0.1
                            for call in active_waited.call_args_list
                        )
                    )
                )
                node.stop()
        waited.assert_not_called()


def _diagnostic_config(**overrides):
    config = {
        "mode": "diagnostic_constant",
        "constant_distance_m": 0.5,
        "decision_threshold_m": 1.0,
        "fallback_distance_m": 3.0,
        "soft_timeout_s": 1.0,
        "scene_mode": "metadata",
        "min_interval_ms": 0,
    }
    config.update(overrides)
    return config


class ObstacleDistancePluginLifecycleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.modules = _ModuleOverrides(_ros_stubs())
        cls.modules.start()
        cls.original_plugin = sys.modules.pop(
            "plugins.obstacle_distance", _ModuleOverrides._MISSING
        )
        cls.plugin_module = importlib.import_module("plugins.obstacle_distance")

    @classmethod
    def tearDownClass(cls):
        sys.modules.pop("plugins.obstacle_distance", None)
        cls.modules.stop()
        if cls.original_plugin is not _ModuleOverrides._MISSING:
            sys.modules["plugins.obstacle_distance"] = cls.original_plugin

    def test_plugin_initialization_copies_config_and_loads_backends_once(self):
        original = _diagnostic_config(nested={"value": "original"})
        executor = mock.Mock()
        with mock.patch.object(
            self.plugin_module.backend_loader,
            "create_model_backends",
            return_value=(None, None),
        ) as create:
            plugin = self.plugin_module.ObstacleDistancePlugin(original, executor)
        original["nested"]["value"] = "mutated"

        self.assertEqual(plugin._plugin_cfg["nested"]["value"], "original")
        create.assert_called_once()

    def test_model_initialization_failure_is_fatal_without_constant_fallback(self):
        with mock.patch.object(
            self.plugin_module.backend_loader,
            "create_model_backends",
            side_effect=RuntimeError("factory failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "factory failed"):
                self.plugin_module.ObstacleDistancePlugin(
                    {"mode": "model", "backend_factory": "x:y"},
                    mock.Mock(),
                )

    def test_diagnostic_mode_without_constant_fails_during_plugin_initialization(self):
        with self.assertRaisesRegex(
            ValueError,
            "diagnostic_constant mode requires constant_distance_m",
        ):
            self.plugin_module.ObstacleDistancePlugin(
                {"mode": "diagnostic_constant"},
                mock.Mock(),
            )

    def test_invalid_top_level_estimator_config_fails_during_plugin_initialization(self):
        invalid = (
            {"decision_threshold_m": 0},
            {"soft_timeout_s": 0},
            {"fallback_distance_m": math.nan},
        )
        for override in invalid:
            with self.subTest(override=override):
                with self.assertRaises(ValueError):
                    self.plugin_module.ObstacleDistancePlugin(
                        _diagnostic_config(**override),
                        mock.Mock(),
                    )

    def test_start_requires_topic_and_explicit_metadata_scene(self):
        plugin = self.plugin_module.ObstacleDistancePlugin(
            _diagnostic_config(), mock.Mock()
        )
        with self.assertRaisesRegex(ValueError, "input_topic is required"):
            plugin.dispatch("obstacle_distance", {"action": "start"})
        with self.assertRaisesRegex(ValueError, "scene_hint is required"):
            plugin.dispatch(
                "obstacle_distance",
                {"action": "start", "input_topic": "/camera"},
            )

    def test_scene_priority_reuse_and_info(self):
        from plugin_dispatch import dispatch_plugin

        executor = mock.Mock()
        plugin = self.plugin_module.ObstacleDistancePlugin(
            _diagnostic_config(fixed_scene="vehicle"), executor
        )
        plugin.dispatch(
            "obstacle_distance",
            {
                "action": "config",
                "instance_id": "case-1",
                "scene_hint": "vehicle",
            },
        )
        first = dispatch_plugin(
            [plugin],
            "obstacle_distance_obstacle_distance",
            {
                "action": "start",
                "instance_id": "case-1",
                "input_topic": "/camera",
                "scene_hint": "indoor",
            },
        )
        second = plugin.dispatch(
            "obstacle_distance",
            {
                "action": "start",
                "instance_id": "case-1",
                "input_topic": "/camera",
                "scene_hint": "indoor",
            },
        )
        info = plugin.dispatch("info", {"instance_id": "case-1"})
        plugin.dispatch("stop", {"instance_id": "case-1"})

        self.assertEqual(first, second)
        self.assertEqual(info["state"], "running")
        self.assertEqual(info["scene"], "indoor")
        self.assertEqual(info["topic_in"][0]["topic"], "/camera")
        self.assertEqual(
            info["topic_out"][0]["topic"], "/camera/obstacle_distance"
        )
        executor.add_node.assert_called_once()

    def test_instance_scene_is_used_and_fixed_scene_only_outside_metadata_mode(self):
        metadata = self.plugin_module.ObstacleDistancePlugin(
            _diagnostic_config(fixed_scene="indoor"), mock.Mock()
        )
        with self.assertRaisesRegex(ValueError, "scene_hint is required"):
            metadata.dispatch(
                "start", {"input_topic": "/metadata-camera"}
            )

        fixed = self.plugin_module.ObstacleDistancePlugin(
            _diagnostic_config(
                scene_mode="fixed",
                fixed_scene="vehicle",
            ),
            mock.Mock(),
        )
        status = fixed.dispatch("start", {"input_topic": "/fixed-camera"})
        try:
            self.assertEqual(status["scene"], "vehicle")
        finally:
            fixed.dispatch("stop", {})

    def test_topic_change_retires_node_and_final_cleanup_is_exactly_once(self):
        executor = mock.Mock()
        plugin = self.plugin_module.ObstacleDistancePlugin(
            _diagnostic_config(), executor
        )
        plugin.dispatch(
            "start",
            {
                "instance_id": "case-1",
                "input_topic": "/old",
                "scene_hint": "indoor",
            },
        )
        old = plugin._nodes["case-1"]
        plugin.dispatch(
            "start",
            {
                "instance_id": "case-1",
                "input_topic": "/new",
                "scene_hint": "vehicle",
            },
        )
        new = plugin._nodes["case-1"]

        self.assertIn(old, plugin._retired_nodes)
        self.assertEqual(old.destroy_calls, 0)
        executor.remove_node.assert_called_once_with(old)
        plugin.prepare_shutdown()
        plugin.destroy_nodes()
        plugin.destroy_nodes()

        self.assertEqual(old.destroy_calls, 1)
        self.assertEqual(new.destroy_calls, 1)
        self.assertEqual(executor.remove_node.call_count, 2)
        self.assertEqual(plugin._nodes, {})
        self.assertEqual(plugin._retired_nodes, [])

    def test_cleanup_retries_transient_remove_and_destroy_failures(self):
        plugin = object.__new__(self.plugin_module.ObstacleDistancePlugin)
        plugin._lifecycle_lock = threading.RLock()
        plugin._executor = mock.Mock()
        remove_retry = mock.Mock(name="remove_retry")
        destroy_retry = mock.Mock(name="destroy_retry")
        healthy = mock.Mock(name="healthy")
        nodes = [remove_retry, destroy_retry, healthy]
        plugin._nodes = {str(index): node for index, node in enumerate(nodes)}
        plugin._retired_nodes = []
        plugin._executor_nodes = {id(node): node for node in nodes}
        plugin._destroyed_node_ids = set()
        remove_attempts = {id(node): 0 for node in nodes}

        def remove(node):
            remove_attempts[id(node)] += 1
            if node is remove_retry and remove_attempts[id(node)] == 1:
                raise RuntimeError("secret remove failure")
            return True

        plugin._executor.remove_node.side_effect = remove
        destroy_retry.destroy_node.side_effect = [
            RuntimeError("secret destroy failure"),
            True,
        ]

        with self.assertLogs(
            "plugins.obstacle_distance", level="ERROR"
        ) as captured:
            plugin.destroy_nodes()

        self.assertNotIn("secret", "\n".join(captured.output))
        self.assertIn(id(remove_retry), plugin._executor_nodes)
        remove_retry.destroy_node.assert_not_called()
        self.assertNotIn(id(destroy_retry), plugin._executor_nodes)
        self.assertIn(remove_retry, plugin._retired_nodes)
        self.assertIn(destroy_retry, plugin._retired_nodes)
        healthy.destroy_node.assert_called_once_with()

        plugin.destroy_nodes()

        self.assertEqual(plugin._executor_nodes, {})
        self.assertEqual(plugin._retired_nodes, [])
        remove_retry.destroy_node.assert_called_once_with()
        self.assertEqual(destroy_retry.destroy_node.call_count, 2)

    def test_executor_add_failure_destroys_new_node_without_leaking_secret(self):
        executor = mock.Mock()
        executor.add_node.side_effect = RuntimeError("secret add failure")
        plugin = self.plugin_module.ObstacleDistancePlugin(
            _diagnostic_config(), executor
        )
        node = mock.Mock()
        node.destroy_node.return_value = True

        with mock.patch.object(
            self.plugin_module, "_ObstacleDistanceNode", return_value=node
        ):
            with self.assertRaises(RuntimeError) as raised:
                plugin.dispatch(
                    "start",
                    {"input_topic": "/camera", "scene_hint": "indoor"},
                )

        self.assertNotIn("secret", str(raised.exception))
        node.destroy_node.assert_called_once_with()
        self.assertEqual(plugin._nodes, {})
        self.assertEqual(plugin._executor_nodes, {})
        self.assertEqual(plugin._retired_nodes, [])

    def test_new_node_start_failure_is_removed_and_retired_for_cleanup(self):
        executor = mock.Mock()
        plugin = self.plugin_module.ObstacleDistancePlugin(
            _diagnostic_config(), executor
        )
        node = mock.Mock(
            _input_topic="/camera",
            _scene_hint="indoor",
            state="idle",
        )
        node.start.side_effect = RuntimeError("secret start failure")

        with mock.patch.object(
            self.plugin_module, "_ObstacleDistanceNode", return_value=node
        ):
            with self.assertRaises(RuntimeError) as raised:
                plugin.dispatch(
                    "start",
                    {"input_topic": "/camera", "scene_hint": "indoor"},
                )

        self.assertNotIn("secret", str(raised.exception))
        executor.remove_node.assert_called_once_with(node)
        self.assertEqual(plugin._nodes, {})
        self.assertEqual(plugin._executor_nodes, {})
        self.assertEqual(plugin._retired_nodes, [node])
        plugin.destroy_nodes()
        node.destroy_node.assert_called_once_with()

    def test_config_validates_fields_stops_running_node_and_changes_threshold(self):
        plugin = self.plugin_module.ObstacleDistancePlugin(
            _diagnostic_config(), mock.Mock()
        )
        plugin.dispatch(
            "start",
            {
                "instance_id": "case-1",
                "input_topic": "/camera",
                "scene_hint": "indoor",
            },
        )
        old_estimator = plugin._nodes["case-1"]._estimator
        result = plugin.dispatch(
            "config",
            {
                "instance_id": "case-1",
                "decision_threshold_m": 0.4,
                "min_interval_ms": 5,
                "scene_hint": "vehicle",
                "input_topic": "/ignored",
            },
        )
        node = plugin._nodes["case-1"]

        self.assertEqual(result["status"], "configured")
        self.assertEqual(node.state, "idle")
        self.assertIsNot(node._estimator, old_estimator)
        self.assertEqual(node._scene_hint, "vehicle")
        self.assertEqual(node._min_interval, 0.005)
        node.start()
        try:
            node._image_cb(types.SimpleNamespace(data=b"frame"))
            self.assertTrue(
                _wait_for(lambda: node._pub.publish.call_count == 1)
            )
        finally:
            node.stop()
        payload = json.loads(node._pub.publish.call_args.args[0].data)
        self.assertEqual(payload["decision_threshold_m"], 0.4)
        self.assertFalse(payload["near_obstacle"])

        invalid = (
            {"scene_hint": "road"},
            {"decision_threshold_m": 0},
            {"min_interval_ms": -1},
        )
        for config in invalid:
            with self.subTest(config=config):
                with self.assertRaises(ValueError):
                    plugin.dispatch(
                        "config", {"instance_id": "case-2", **config}
                    )

    def test_live_backend_mutation_is_rejected_and_errors_hide_arguments(self):
        plugin = self.plugin_module.ObstacleDistancePlugin(
            _diagnostic_config(), mock.Mock()
        )
        secret = "secret.module:factory"
        with self.assertRaises(ValueError) as raised:
            plugin.dispatch(
                "config",
                {"backend_factory": secret, "mode": "model"},
            )
        self.assertNotIn(secret, str(raised.exception))
        self.assertIn("backend configuration cannot be changed live", str(raised.exception))

    def test_stop_all_is_idempotent_and_cleanup_continues_after_node_errors(self):
        plugin = object.__new__(self.plugin_module.ObstacleDistancePlugin)
        plugin._lifecycle_lock = threading.RLock()
        plugin._executor = mock.Mock()
        healthy = mock.Mock()
        broken = mock.Mock()
        broken.stop.side_effect = RuntimeError("stop failed")
        broken.destroy_node.side_effect = RuntimeError("destroy failed")
        plugin._nodes = {"healthy": healthy, "broken": broken}
        plugin._retired_nodes = [healthy]
        plugin._executor_nodes = {id(healthy): healthy, id(broken): broken}

        with self.assertLogs("plugins.obstacle_distance", level="ERROR"):
            plugin.prepare_shutdown()
            plugin.destroy_nodes()

        healthy.stop.assert_called()
        healthy.destroy_node.assert_called_once_with()
        broken.destroy_node.assert_not_called()
        self.assertEqual(plugin._retired_nodes, [broken])

        broken.stop.side_effect = None
        broken.destroy_node.side_effect = None
        plugin.destroy_nodes()

        broken.destroy_node.assert_called_once_with()
        self.assertEqual(plugin._nodes, {})
        self.assertEqual(plugin._retired_nodes, [])


class ObstacleDistanceMainIntegrationTest(unittest.TestCase):
    def test_bundle_registers_only_when_explicitly_enabled(self):
        modules = _ros_stubs()
        yaml = types.ModuleType("yaml")
        yaml.safe_load = mock.Mock()
        fake_plugin_module = types.ModuleType("plugins.obstacle_distance")
        fake_plugin_module.ObstacleDistancePlugin = mock.Mock()
        modules.update(
            {
                "yaml": yaml,
                "plugins.obstacle_distance": fake_plugin_module,
            }
        )
        with _ModuleOverrides(modules):
            spec = importlib.util.spec_from_file_location(
                "perception_main_obstacle_distance_test",
                PERCEPTION_ROOT / "main.py",
            )
            main = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(main)
            executor = object()
            disabled = main.PerceptionBundle({"plugins": {}}, executor)
            enabled_config = {"enabled": True, "mode": "diagnostic_constant"}
            enabled = main.PerceptionBundle(
                {"plugins": {"obstacle_distance": enabled_config}},
                executor,
            )

        self.assertEqual(disabled._plugins, [])
        fake_plugin_module.ObstacleDistancePlugin.assert_called_once_with(
            enabled_config, executor
        )
        self.assertEqual(
            enabled._plugins,
            [fake_plugin_module.ObstacleDistancePlugin.return_value],
        )


if __name__ == "__main__":
    unittest.main()
