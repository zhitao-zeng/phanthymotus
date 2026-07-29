import sys
import unittest
from pathlib import Path


PERCEPTION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PERCEPTION_ROOT))

from plugin_dispatch import dispatch_plugin, full_tool_name


class FakePlugin:
    def __init__(self, prefix):
        self.PREFIX = prefix
        self.calls = []

    def dispatch(self, name, args):
        self.calls.append((name, args))
        return {"prefix": self.PREFIX, "name": name, "args": args}


class PluginDispatchTest(unittest.TestCase):
    def test_dispatch_prefers_longest_underscored_prefix(self):
        obstacle = FakePlugin("obstacle")
        obstacle_distance = FakePlugin("obstacle_distance")
        args = {"action": "info"}

        result = dispatch_plugin(
            [obstacle, obstacle_distance],
            "obstacle_distance_obstacle_distance",
            args,
        )

        self.assertEqual(
            result,
            {
                "prefix": "obstacle_distance",
                "name": "obstacle_distance",
                "args": args,
            },
        )
        self.assertEqual(obstacle.calls, [])
        self.assertEqual(obstacle_distance.calls, [("obstacle_distance", args)])

    def test_full_tool_name_does_not_duplicate_matching_prefix(self):
        self.assertEqual(
            full_tool_name("obstacle_distance", "obstacle_distance"),
            "obstacle_distance",
        )

    def test_full_tool_name_adds_prefix_to_other_tool_name(self):
        self.assertEqual(
            full_tool_name("obstacle_distance", "info"),
            "obstacle_distance_info",
        )


if __name__ == "__main__":
    unittest.main()
