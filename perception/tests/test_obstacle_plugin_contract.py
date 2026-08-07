import importlib
import sys
import types
import unittest
from array import array
from pathlib import Path
from unittest import mock


PERCEPTION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PERCEPTION_ROOT))


def _install_ros_stubs():
    class FakeNode:
        def __init__(self, name):
            self.name = name

        def create_publisher(self, *args, **kwargs):
            return mock.Mock()

        def create_subscription(self, *args, **kwargs):
            return mock.Mock()

        def destroy_subscription(self, *args, **kwargs):
            return True

    rclpy = types.ModuleType("rclpy")
    rclpy.node = types.ModuleType("rclpy.node")
    rclpy.node.Node = FakeNode
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

    sys.modules.update(
        {
            "rclpy": rclpy,
            "rclpy.node": rclpy.node,
            "rclpy.qos": rclpy.qos,
            "sensor_msgs": sensor_msgs,
            "sensor_msgs.msg": sensor_msgs.msg,
            "std_msgs": std_msgs,
            "std_msgs.msg": std_msgs.msg,
        }
    )


class ObstaclePluginContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _install_ros_stubs()
        cls.obstacle = importlib.import_module("plugins.obstacle")

    def test_local_model_initialization_failure_is_fatal(self):
        with mock.patch.object(
            self.obstacle,
            "create_model_backends",
            side_effect=RuntimeError("model load failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "model load failed"):
                self.obstacle.LocalDistanceAdapter(
                    {
                        "provider": "local",
                        "mode": "model",
                        "fixed_scene": "indoor",
                    }
                )

    def test_diagnostic_constant_remains_an_explicit_supported_mode(self):
        adapter = self.obstacle.LocalDistanceAdapter(
            {
                "provider": "local",
                "mode": "diagnostic_constant",
                "constant_distance_m": 0.5,
                "fixed_scene": "indoor",
            }
        )

        result = adapter.estimate(b"diagnostic-image")

        self.assertEqual(result["pred_distance"], 0.5)
        self.assertEqual(result["status"], "diagnostic_constant")
        self.assertFalse(result["fallback"])

    def test_scene_must_be_explicit_and_fixed_for_ros_input(self):
        with self.assertRaisesRegex(ValueError, "fixed_scene"):
            self.obstacle.LocalDistanceAdapter(
                {
                    "provider": "local",
                    "mode": "diagnostic_constant",
                    "constant_distance_m": 0.5,
                    "scene_mode": "fixed",
                }
            )
        with self.assertRaisesRegex(ValueError, "scene_mode must be fixed"):
            self.obstacle.LocalDistanceAdapter(
                {
                    "provider": "local",
                    "mode": "diagnostic_constant",
                    "constant_distance_m": 0.5,
                    "scene_mode": "metadata",
                    "fixed_scene": "indoor",
                }
            )

    def test_scene_can_be_overridden_by_environment_for_separate_images(self):
        with mock.patch.dict(
            "os.environ", {"OBSTACLE_FIXED_SCENE": "vehicle"}
        ):
            adapter = self.obstacle.LocalDistanceAdapter(
                {
                    "provider": "local",
                    "mode": "diagnostic_constant",
                    "constant_distance_m": 2.0,
                    "scene_mode": "fixed",
                    "fixed_scene": "indoor",
                }
            )

        result = adapter.estimate(b"diagnostic-image")
        self.assertEqual(result["scene"], "vehicle")

    def test_scene_can_be_selected_from_configured_image_resolution(self):
        adapter = self.obstacle.LocalDistanceAdapter(
            {
                "provider": "local",
                "mode": "diagnostic_constant",
                "constant_distance_m": 2.0,
                "scene_mode": "resolution",
                "resolution_scene_map": {
                    "640x480": "indoor",
                    "1600x900": "vehicle",
                },
            }
        )

        with mock.patch(
            "cv2.imdecode",
            side_effect=(
                self.obstacle.np.zeros((480, 640, 3), dtype="uint8"),
                self.obstacle.np.zeros((900, 1600, 3), dtype="uint8"),
            ),
        ):
            indoor = adapter.estimate(b"indoor-image")
            vehicle = adapter.estimate(b"vehicle-image")

        self.assertEqual(indoor["scene"], "indoor")
        self.assertEqual(vehicle["scene"], "vehicle")

    def test_scene_can_be_selected_from_image_content(self):
        router = mock.Mock()
        router.predict.side_effect = (
            self.obstacle.SceneDomain.INDOOR,
            self.obstacle.SceneDomain.VEHICLE,
        )
        with mock.patch.object(
            self.obstacle,
            "create_scene_router",
            return_value=router,
        ):
            adapter = self.obstacle.LocalDistanceAdapter(
                {
                    "provider": "local",
                    "mode": "diagnostic_constant",
                    "constant_distance_m": 2.0,
                    "scene_mode": "content",
                }
            )

        indoor = adapter.estimate(b"first-image")
        vehicle = adapter.estimate(b"second-image")

        self.assertEqual(indoor["scene"], "indoor")
        self.assertEqual(vehicle["scene"], "vehicle")
        self.assertEqual(router.predict.call_count, 2)

    def test_unknown_resolution_uses_structured_missing_scene_fallback(self):
        adapter = self.obstacle.LocalDistanceAdapter(
            {
                "provider": "local",
                "mode": "diagnostic_constant",
                "constant_distance_m": 2.0,
                "scene_mode": "resolution",
                "resolution_scene_map": {"640x480": "indoor"},
            }
        )

        with mock.patch(
            "cv2.imdecode",
            return_value=self.obstacle.np.zeros(
                (720, 1280, 3), dtype="uint8"
            ),
        ):
            result = adapter.estimate(b"unknown-resolution")

        self.assertTrue(result["fallback"])
        self.assertEqual(result["error_code"], "missing_scene")

    def test_invalid_resolution_image_uses_invalid_image_fallback(self):
        adapter = self.obstacle.LocalDistanceAdapter(
            {
                "provider": "local",
                "mode": "diagnostic_constant",
                "constant_distance_m": 2.0,
                "scene_mode": "resolution",
                "resolution_scene_map": {"640x480": "indoor"},
            }
        )

        with mock.patch("cv2.imdecode", return_value=None):
            result = adapter.estimate(b"not-an-image")

        self.assertTrue(result["fallback"])
        self.assertEqual(result["error_code"], "invalid_image")

    def test_ros_compressed_image_data_is_normalized_to_bytes(self):
        node = self.obstacle._ObstacleNode(
            "/camera/image",
            mock.Mock(),
            "bytes_contract",
        )
        node.state = "running"
        message = types.SimpleNamespace(
            data=array("B", [0, 1, 2, 255]),
            format="jpeg",
        )

        node._image_cb(message)

        queued = node._frame_queue.get_nowait()
        self.assertIsInstance(queued, bytes)
        self.assertEqual(queued, b"\x00\x01\x02\xff")

    def test_idle_node_ignores_frames(self):
        node = self.obstacle._ObstacleNode(
            "/camera/image",
            mock.Mock(),
            "idle_frames",
        )
        message = types.SimpleNamespace(data=b"ignored", format="jpeg")

        node._image_cb(message)

        self.assertTrue(node._frame_queue.empty())

    def test_node_restart_reuses_subscription(self):
        node = self.obstacle._ObstacleNode(
            "/camera/image",
            mock.Mock(),
            "subscription_reuse",
        )
        workers = [mock.Mock(), mock.Mock()]
        workers[0].is_alive.side_effect = (True, False)
        workers[1].is_alive.return_value = True

        with mock.patch.object(
            node,
            "create_subscription",
            wraps=node.create_subscription,
        ) as create_subscription, mock.patch.object(
            node,
            "destroy_subscription",
            wraps=node.destroy_subscription,
        ) as destroy_subscription, mock.patch.object(
            self.obstacle.threading,
            "Thread",
            side_effect=workers,
        ):
            node.start()
            subscription = node._sub
            node.stop()
            node.start()

        self.assertIs(node._sub, subscription)
        create_subscription.assert_called_once()
        destroy_subscription.assert_not_called()
        workers[0].start.assert_called_once()
        workers[0].join.assert_called_once_with(timeout=3.0)
        workers[1].start.assert_called_once()

    def test_node_does_not_restart_while_previous_worker_is_alive(self):
        node = self.obstacle._ObstacleNode(
            "/camera/image",
            mock.Mock(),
            "slow_worker",
        )
        worker = mock.Mock()
        worker.is_alive.return_value = True

        with mock.patch.object(
            self.obstacle.threading,
            "Thread",
            return_value=worker,
        ) as thread_factory:
            node.start()
            first_stop_event = node._stop_event
            node.stop()

            with self.assertRaisesRegex(RuntimeError, "still stopping"):
                node.start()

        self.assertTrue(first_stop_event.is_set())
        thread_factory.assert_called_once()
        worker.join.assert_called_once_with(timeout=3.0)
        self.assertIs(node._worker, worker)

    def test_global_config_update_preserves_model_configuration(self):
        initial_adapter = object()
        updated_adapter = object()
        base_config = {
            "provider": "local",
            "mode": "model",
            "backend_factory": "plugins.backends:create",
            "depth_model_dir": "/models/depth",
            "segmentation_model_dir": "/models/segmentation",
            "fixed_scene": "indoor",
        }

        with mock.patch.object(
            self.obstacle,
            "_build_distance_adapter",
            side_effect=(initial_adapter, updated_adapter),
        ) as build:
            plugin = self.obstacle.ObstacleDistancePlugin(
                base_config,
                mock.Mock(),
            )
            result = plugin.dispatch(
                "obstacle",
                {"action": "config", "decision_threshold_m": 1.25},
            )

        self.assertEqual(result["status"], "configured")
        self.assertIs(plugin._adapter, updated_adapter)
        updated_config = build.call_args_list[1].args[0]
        self.assertEqual(
            updated_config["backend_factory"],
            "plugins.backends:create",
        )
        self.assertEqual(updated_config["depth_model_dir"], "/models/depth")
        self.assertEqual(updated_config["decision_threshold_m"], 1.25)

    def test_instance_config_is_merged_with_base_model_configuration(self):
        initial_adapter = object()
        instance_adapter = object()
        fake_node = mock.Mock()
        fake_node.start.return_value = {"state": "running"}
        executor = mock.Mock()
        base_config = {
            "provider": "local",
            "mode": "model",
            "backend_factory": "plugins.backends:create",
            "depth_model_dir": "/models/depth",
            "segmentation_model_dir": "/models/segmentation",
            "fixed_scene": "indoor",
        }

        with mock.patch.object(
            self.obstacle,
            "_build_distance_adapter",
            side_effect=(initial_adapter, instance_adapter),
        ) as build, mock.patch.object(
            self.obstacle,
            "_ObstacleNode",
            return_value=fake_node,
        ):
            plugin = self.obstacle.ObstacleDistancePlugin(base_config, executor)
            plugin.dispatch(
                "obstacle",
                {
                    "action": "config",
                    "instance_id": "judge-1",
                    "decision_threshold_m": 0.9,
                },
            )
            plugin.dispatch(
                "obstacle",
                {
                    "action": "start",
                    "instance_id": "judge-1",
                    "input_topic": "/benchmark/camera",
                },
            )

        instance_config = build.call_args_list[1].args[0]
        self.assertEqual(
            instance_config["backend_factory"],
            "plugins.backends:create",
        )
        self.assertEqual(instance_config["decision_threshold_m"], 0.9)
        executor.add_node.assert_called_once_with(fake_node)

    def test_repeated_judge_lifecycle_reuses_one_executor_node(self):
        adapter = object()
        executor = mock.Mock()
        fake_node = mock.Mock()
        fake_node.state = "idle"

        def start():
            fake_node.state = "running"
            return {"state": "running"}

        def stop():
            fake_node.state = "idle"
            return {"state": "idle"}

        fake_node.start.side_effect = start
        fake_node.stop.side_effect = stop

        with mock.patch.object(
            self.obstacle,
            "_build_distance_adapter",
            return_value=adapter,
        ) as build, mock.patch.object(
            self.obstacle,
            "_ObstacleNode",
            return_value=fake_node,
        ) as node_factory:
            plugin = self.obstacle.ObstacleDistancePlugin(
                {"provider": "local"},
                executor,
            )
            start_args = {
                "action": "start",
                "input_topic": "/benchmark/camera",
            }
            stop_args = {"action": "stop"}

            for _ in range(100):
                plugin.dispatch("obstacle", {"action": "config"})
                plugin.dispatch("obstacle", start_args)
                plugin.dispatch("obstacle", stop_args)

        node_factory.assert_called_once_with(
            "/benchmark/camera",
            adapter,
            "benchmark_camera",
        )
        build.assert_called_once_with({"provider": "local"})
        executor.add_node.assert_called_once_with(fake_node)
        executor.remove_node.assert_not_called()
        self.assertIs(plugin._nodes["/benchmark/camera"], fake_node)
        self.assertEqual(fake_node.start.call_count, 100)
        self.assertEqual(fake_node.stop.call_count, 100)

    def test_existing_instance_is_reconfigured_without_node_replacement(self):
        initial_adapter = object()
        updated_adapter = object()
        executor = mock.Mock()
        fake_node = mock.Mock()
        fake_node.state = "running"
        fake_node.start.return_value = {"state": "running"}

        with mock.patch.object(
            self.obstacle,
            "_build_distance_adapter",
            side_effect=(initial_adapter, updated_adapter),
        ), mock.patch.object(
            self.obstacle,
            "_ObstacleNode",
            return_value=fake_node,
        ) as node_factory:
            plugin = self.obstacle.ObstacleDistancePlugin(
                {"provider": "local", "fixed_scene": "indoor"},
                executor,
            )
            plugin.dispatch(
                "obstacle",
                {
                    "action": "start",
                    "instance_id": "judge-1",
                    "input_topic": "/benchmark/camera",
                },
            )
            result = plugin.dispatch(
                "obstacle",
                {
                    "action": "config",
                    "instance_id": "judge-1",
                    "decision_threshold_m": 0.9,
                },
            )

        self.assertEqual(result["status"], "configured")
        node_factory.assert_called_once()
        executor.add_node.assert_called_once_with(fake_node)
        executor.remove_node.assert_not_called()
        fake_node.stop.assert_called_once()
        fake_node.set_adapter.assert_called_once_with(updated_adapter)


if __name__ == "__main__":
    unittest.main()
