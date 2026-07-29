import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


PERCEPTION_ROOT = Path(__file__).resolve().parents[1]


def _load_plugin_dispatch_module():
    spec = importlib.util.spec_from_file_location(
        "perception_plugin_dispatch_test", PERCEPTION_ROOT / "plugin_dispatch.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_main_module():
    rclpy = types.ModuleType("rclpy")
    rclpy.executors = types.ModuleType("rclpy.executors")
    yaml = types.ModuleType("yaml")
    yaml.safe_load = mock.Mock()
    perception = types.ModuleType("perception")
    perception.__path__ = [str(PERCEPTION_ROOT)]
    original_path = list(sys.path)

    with (
        mock.patch.object(sys, "path", original_path.copy()),
        mock.patch.dict(
            sys.modules,
            {
                "perception": perception,
                "rclpy": rclpy,
                "rclpy.executors": rclpy.executors,
                "yaml": yaml,
            },
        ),
    ):
        spec = importlib.util.spec_from_file_location(
            "perception.main_plugin_dispatch_test", PERCEPTION_ROOT / "main.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        loaded_path = list(sys.path)

    return module, original_path, loaded_path


plugin_dispatch = _load_plugin_dispatch_module()
dispatch_plugin = plugin_dispatch.dispatch_plugin
full_tool_name = plugin_dispatch.full_tool_name


class FakePlugin:
    def __init__(self, prefix, tools=None):
        self.PREFIX = prefix
        self.calls = []
        self.tools = tools or []

    def get_tools(self):
        return self.tools

    def dispatch(self, name, args):
        self.calls.append((name, args))
        return {"prefix": self.PREFIX, "name": name, "args": args}


class PluginDispatchTest(unittest.TestCase):
    def test_dispatch_prefers_longest_underscored_prefix(self):
        for prefixes in (
            ("obstacle", "obstacle_distance"),
            ("obstacle_distance", "obstacle"),
        ):
            with self.subTest(prefixes=prefixes):
                plugins = {prefix: FakePlugin(prefix) for prefix in prefixes}
                result = dispatch_plugin(
                    [plugins[prefix] for prefix in prefixes],
                    "obstacle_distance_info",
                    {},
                )

                self.assertEqual(result["prefix"], "obstacle_distance")
                self.assertEqual(plugins["obstacle"].calls, [])
                self.assertEqual(plugins["obstacle_distance"].calls, [("info", {})])

    def test_same_name_round_trip_preserves_internal_name(self):
        obstacle_distance = FakePlugin("obstacle_distance")
        full_name = full_tool_name("obstacle_distance", "obstacle_distance")

        result = dispatch_plugin([obstacle_distance], full_name, {})

        self.assertEqual(
            result,
            {
                "prefix": "obstacle_distance",
                "name": "obstacle_distance",
                "args": {},
            },
        )

    def test_regular_tool_round_trip_strips_prefix(self):
        obstacle_distance = FakePlugin("obstacle_distance")
        full_name = full_tool_name("obstacle_distance", "info")

        result = dispatch_plugin([obstacle_distance], full_name, {})

        self.assertEqual(
            result,
            {"prefix": "obstacle_distance", "name": "info", "args": {}},
        )

    def test_dispatch_returns_none_without_matching_plugin(self):
        self.assertIsNone(dispatch_plugin([FakePlugin("obstacle")], "camera_info", {}))

    def test_bundle_uses_dispatch_helpers_without_mutating_import_path(self):
        main, original_path, loaded_path = _load_main_module()
        plugin = FakePlugin(
            "obstacle_distance",
            [{"name": "obstacle_distance"}, {"name": "info"}],
        )
        bundle = object.__new__(main.PerceptionBundle)
        bundle._plugins = [plugin]

        tools = bundle.get_all_tools()
        result = bundle.dispatch(tools[1]["name"], {"action": "info"})

        self.assertEqual(loaded_path, original_path)
        self.assertEqual(
            [tool["name"] for tool in tools],
            ["obstacle_distance", "obstacle_distance_info"],
        )
        self.assertEqual(result["name"], "info")


if __name__ == "__main__":
    unittest.main()
