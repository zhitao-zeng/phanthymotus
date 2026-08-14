import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


PERCEPTION_ROOT = Path(__file__).resolve().parents[1]


def _load_main_module():
    rclpy = types.ModuleType("rclpy")
    rclpy.executors = types.ModuleType("rclpy.executors")
    yaml = types.ModuleType("yaml")
    yaml.safe_load = mock.Mock()
    with mock.patch.dict(
        sys.modules,
        {"rclpy": rclpy, "rclpy.executors": rclpy.executors, "yaml": yaml},
    ):
        spec = importlib.util.spec_from_file_location(
            "perception_main_lifecycle", PERCEPTION_ROOT / "main.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


class MainLifecycleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main = _load_main_module()

    def test_bundle_runs_optional_lifecycle_hooks(self):
        lifecycle_plugin = mock.Mock()
        legacy_plugin = types.SimpleNamespace(PREFIX="legacy")
        bundle = object.__new__(self.main.PerceptionBundle)
        bundle._plugins = [lifecycle_plugin, legacy_plugin]

        bundle.run_lifecycle_hook("prepare_shutdown")
        bundle.run_lifecycle_hook("destroy_nodes")

        lifecycle_plugin.prepare_shutdown.assert_called_once_with()
        lifecycle_plugin.destroy_nodes.assert_called_once_with()

    def test_environment_ports_override_yaml(self):
        source = (PERCEPTION_ROOT / "main.py").read_text(encoding="utf-8")

        self.assertIn('os.environ.get("MCP_PORT") or cfg.get("mcp_port", 15720)', source)
        self.assertIn('os.environ.get("WS_PORT") or cfg.get("ws_port", 15721)', source)


if __name__ == "__main__":
    unittest.main()
