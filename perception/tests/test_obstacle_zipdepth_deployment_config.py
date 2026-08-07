import json
import unittest
from pathlib import Path

import yaml


PERCEPTION_ROOT = Path(__file__).resolve().parents[1]


class ZipDepthDeploymentConfigTest(unittest.TestCase):
    def test_default_is_independent_resolution_routed_zipdepth(self):
        config = yaml.safe_load((PERCEPTION_ROOT / "config.yaml").read_text())
        obstacle = config["plugins"]["obstacle"]

        self.assertNotIn("shared_inference_port", obstacle)
        self.assertEqual(obstacle["scene_mode"], "resolution")
        self.assertEqual(
            obstacle["resolution_scene_map"],
            {"640x480": "indoor", "1600x900": "vehicle"},
        )
        self.assertNotIn("scene_router_model", obstacle)
        self.assertNotIn("scene_router_io_labels", obstacle)
        self.assertNotIn("scene_router_top_k", obstacle)
        self.assertEqual(
            obstacle["backend_factory"],
            "plugins.obstacle_distance_core.zipdepth_tensorrt_backends:create_backends",
        )
        self.assertEqual(obstacle["decision_threshold_m"], 2.0)
        self.assertEqual(obstacle["indoor"]["inverse_depth_percentile"], 95.0)
        self.assertAlmostEqual(
            obstacle["indoor"]["score_threshold"],
            -0.06,
        )
        self.assertAlmostEqual(
            obstacle["indoor"]["inverse_depth_distance_scale"],
            -27.0,
        )
        self.assertAlmostEqual(
            obstacle["indoor"]["inverse_depth_distance_bias_m"],
            3.7,
        )
        calibration = obstacle["vehicle"]["distance_calibration"]
        self.assertEqual(calibration["mode"], "affine")
        self.assertAlmostEqual(calibration["scale"], 1.2)
        self.assertAlmostEqual(calibration["bias"], -2.4)
        self.assertNotIn("power", calibration)

    def test_manifest_contains_only_the_five_selected_runtime_artifacts(self):
        manifest = json.loads(
            (PERCEPTION_ROOT / "models/obstacle-artifacts.json").read_text()
        )
        artifacts = manifest["artifacts"]

        self.assertEqual(len(artifacts), 5)
        self.assertEqual(
            sum(item["size_bytes"] for item in artifacts),
            32_857_046,
        )
        self.assertTrue(
            all("zipdepth-int8" in item["destination"] for item in artifacts)
        )


if __name__ == "__main__":
    unittest.main()
