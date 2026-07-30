import subprocess
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PERCEPTION_ROOT = REPO_ROOT / "perception"
MODEL_ROOT = PERCEPTION_ROOT / "models" / "obstacle-distance"
ONE_MIB = 1024 * 1024
FORBIDDEN_MODEL_SUFFIXES = (
    ".pth.tar",
    ".pth",
    ".pt",
    ".onnx",
    ".engine",
    ".plan",
    ".safetensors",
    ".mnn",
)


def _tracked_files() -> tuple[str, ...]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )
    return tuple(
        path
        for path in result.stdout.decode(
            "utf-8", errors="surrogateescape"
        ).split("\0")
        if path
    )


def _is_forbidden_model_artifact(path: str) -> bool:
    return path.lower().endswith(FORBIDDEN_MODEL_SUFFIXES)


class ObstacleDistancePackagingTest(unittest.TestCase):
    def test_default_config_keeps_complete_model_handoff_disabled(self):
        config = (PERCEPTION_ROOT / "config.yaml").read_text(encoding="utf-8")
        expected = """  obstacle_distance:
    enabled: false
    mode: model
    backend_factory: ""
    scene_mode: metadata
    fixed_scene: ""
    depth_backend: lifelong_nk
    segmentation_backend: yolo26n_seg
    depth_model_dir: /models/obstacle-distance/lifelong-nk
    segmentation_model_dir: /models/obstacle-distance/yolo26n-seg
    decision_threshold_m: 1.0
    fallback_distance_m: 3.0
    soft_timeout_s: 2.5
    min_interval_ms: 0
    indoor:
      roi: [0, 300, 213, 426]
      min_depth_m: 0.3
      max_depth_m: 10.0
      percentile: 1.0
      min_valid_pixels: 64
    vehicle:
      allowed_classes: [person, car, truck, bus, motorcycle, bicycle]
      min_confidence: 0.25
      percentile: 1.0
      min_depth_m: 0.3
      max_depth_m: 80.0
      camera_to_bumper_offset_m: 1.0
      allow_approximate_geometry: false
      calibration: {}
"""
        self.assertIn(expected, config)

    def test_bundle_registers_obstacle_distance_plugin(self):
        source = (PERCEPTION_ROOT / "main.py").read_text(encoding="utf-8")
        self.assertIn(
            "from plugins.obstacle_distance import ObstacleDistancePlugin",
            source,
        )
        self.assertIn(
            'plugins_cfg.get("obstacle_distance", {}).get("enabled", False)',
            source,
        )

    def test_readme_links_formal_handoff_document(self):
        readme = (PERCEPTION_ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn(
            "[障碍物距离模型交接](docs/obstacle_distance.md)",
            readme,
        )

    def test_handoff_document_covers_runtime_and_model_contracts(self):
        document = (
            PERCEPTION_ROOT / "docs" / "obstacle_distance.md"
        ).read_text(encoding="utf-8")
        required = (
            "Lifelong-MonoDepth",
            "NK.pth.tar",
            "NYU",
            "index 0",
            "KITTI",
            "index 1",
            "YOLO26n-seg",
            "diagnostic_constant",
            "model",
            "class DepthBackend(Protocol):",
            "domain: SceneDomain",
            "deadline_monotonic: float",
            ") -> DepthPrediction:",
            "class InstanceSegmentationBackend(Protocol):",
            ") -> Sequence[InstanceMask]:",
            "module.path:function_name",
            "tuple[DepthBackend, InstanceSegmentationBackend]",
            "[right, down, forward]",
            "camera_to_ego",
            "row-major",
            "bumper_xy",
            "missing_calibration",
            "approximate_geometry",
            '"action": "start"',
            '"action": "stop"',
            '"action": "info"',
            '"action": "config"',
            "{input_topic}/obstacle_distance",
            "data/json",
            "scene_mode",
            "metadata",
            "fixed_scene",
            "deadline",
            "publish",
            "evaluate_obstacle_distance.py",
            "image_path,scene,gt_distance_m",
            "RMSE",
            "O(n log n)",
            "/models/obstacle-distance/lifelong-nk",
            "/models/obstacle-distance/yolo26n-seg",
            "24.9M",
            "30 MB",
            "INT8",
            ".pth.tar",
            ".safetensors",
            "Git",
            "test_ocr_packaging.py",
            "test_ocr_model_downloader.py",
            "plugins.ocr_preprocess",
            "Python 3.14",
            "3 failures",
            "1 error",
            "codex-primary-runtime",
        )
        for text in required:
            with self.subTest(text=text):
                self.assertIn(text, document)

    def test_forbidden_model_suffix_guard_covers_compound_and_plain_paths(self):
        for path in (
            "perception/models/lifelong-nk/NK.pth.tar",
            "third_party/model.onnx",
            "runtime/optimized.engine",
        ):
            with self.subTest(path=path):
                self.assertTrue(_is_forbidden_model_artifact(path))
        self.assertFalse(
            _is_forbidden_model_artifact(
                "perception/docs/obstacle_distance.md"
            )
        )

    def test_repository_has_no_obstacle_model_directory(self):
        self.assertFalse(MODEL_ROOT.exists())

    def test_git_tracks_no_forbidden_model_artifacts(self):
        for relative in _tracked_files():
            self.assertFalse(
                _is_forbidden_model_artifact(relative),
                relative,
            )

    def test_tracked_obstacle_files_are_small_and_not_lfs_pointers(self):
        for relative in _tracked_files():
            normalized = relative.lower()
            if "obstacle" not in normalized:
                continue
            path = REPO_ROOT / relative
            self.assertTrue(path.is_file(), relative)
            self.assertLess(path.stat().st_size, ONE_MIB, relative)
            self.assertFalse(
                path.read_bytes()[:200].startswith(
                    b"version https://git-lfs.github.com/spec/v1"
                ),
                relative,
            )

    def test_dockerfiles_do_not_copy_obstacle_model_directory(self):
        dockerfiles = [
            REPO_ROOT / relative
            for relative in _tracked_files()
            if (REPO_ROOT / relative).name.startswith("Dockerfile")
        ]
        self.assertTrue(dockerfiles)
        for path in dockerfiles:
            source = path.read_text(encoding="utf-8").lower()
            with self.subTest(path=path.relative_to(REPO_ROOT)):
                self.assertNotIn(
                    "copy perception/models/obstacle-distance",
                    source,
                )
                self.assertNotIn("copy models/obstacle-distance", source)


if __name__ == "__main__":
    unittest.main()
