import ast
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


class OCRPackagingTest(unittest.TestCase):
    def test_bundle_registers_ocr_plugin(self):
        source = (REPO_ROOT / "perception" / "main.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("from plugins.ocr import OCRPlugin", source)
        self.assertIn(
            'plugins_cfg.get("ocr", {}).get("enabled", False)', source
        )

    def test_dynamic_ocr_memory_rejection_module_is_removed(self):
        self.assertFalse(
            (REPO_ROOT / "perception" / "plugins" / "ocr_memory_guard.py").exists()
        )

    def test_ocr_preprocess_does_not_eagerly_import_numeric_runtime(self):
        source = (
            REPO_ROOT / "perception" / "plugins" / "ocr_preprocess.py"
        ).read_text(encoding="utf-8")
        module = ast.parse(source)
        eager_imports = {
            alias.name
            for node in module.body
            if isinstance(node, ast.Import)
            for alias in node.names
        }

        self.assertTrue({"cv2", "numpy"}.isdisjoint(eager_imports))

    def test_default_config_is_bounded_for_ocr_leaderboard(self):
        config = (REPO_ROOT / "perception" / "config.yaml").read_text(
            encoding="utf-8"
        )
        self.assertIn("  asr:\n    enabled: false", config)
        self.assertIn("  tts:\n    enabled: false", config)
        self.assertIn("  htmsg:\n    enabled: false", config)
        self.assertIn("  vop:\n    enabled: false", config)
        self.assertIn("  ocr:\n    enabled: true\n    provider: rapidocr", config)
        self.assertIn(
            "model_dir: /models/ocr/ppocrv6-small-trt",
            config,
        )
        self.assertIn("    use_angle_cls: true", config)
        self.assertIn("    max_side_len: 1600", config)
        self.assertRegex(
            config,
            r"empty_result_retry:\n"
            r"(?:      #.*\n)*"
            r"      enabled: true\n"
            r"      det_thresh: 0.1\n"
            r"      det_box_thresh: 0.1",
        )
        self.assertNotIn("    max_input_mb:", config)
        self.assertNotIn("    max_decode_mb:", config)
        self.assertNotIn("    memory_guard:", config)
        self.assertNotIn("fallback_backend:", config)
        self.assertNotIn("fallback_model_dir:", config)
        self.assertNotIn("large_image_strategy:", config)

    def test_jetson_image_downloads_ocr_models_on_first_start(self):
        dockerfile = (REPO_ROOT / "perception" / "Dockerfile.jetson").read_text(
            encoding="utf-8"
        )
        downloader = (
            REPO_ROOT / "perception" / "utils" / "model_downloader.py"
        ).read_text(encoding="utf-8")

        self.assertIn("rapidocr==3.9.1", dockerfile)
        self.assertNotIn("MNN", dockerfile)
        self.assertIn('"pyvips==3.1.0"', dockerfile)
        self.assertIn("libvips42", dockerfile)
        self.assertIn("AS jp61-compat", dockerfile)
        self.assertIn(
            "COPY --from=jp61-compat /usr/lib/aarch64-linux-gnu/libffi.so.8.1.0",
            dockerfile,
        )
        self.assertIn("NUMPY_VERSION=1.26.4", dockerfile)
        self.assertIn("NUMPY_VERSION=1.24.4", dockerfile)
        self.assertIn('[ "${JP_VERSION}" = "61" ]', dockerfile)
        self.assertNotIn("nvidia.box.com", dockerfile)
        self.assertNotIn("onnxruntime_gpu-1.17.0", dockerfile)
        self.assertNotIn("ORT_WHEEL_URL", dockerfile)
        self.assertIn("version('tensorrt')", dockerfile)
        self.assertIn("from importlib.metadata import version", dockerfile)
        self.assertIn("--no-deps", dockerfile)
        self.assertNotIn("onnxruntime", dockerfile)
        self.assertIn("rapidocr.__file__", dockerfile)
        self.assertIn("-name '*.onnx' -delete", dockerfile)
        self.assertIn("ARG JP_VERSION=61", dockerfile)
        self.assertNotIn("OCR_MODEL_REVISION", dockerfile)
        self.assertNotIn("model-seed/ocr", dockerfile)
        self.assertNotIn("seed_ocr_models.sh", dockerfile)
        self.assertFalse(
            (REPO_ROOT / "perception" / "deploy" / "seed_ocr_models.sh").exists()
        )
        self.assertNotIn("resolve/master", dockerfile)

        self.assertIn("OCR_MODEL_BASE_URL", downloader)
        self.assertIn("www.modelscope.cn", downloader)
        self.assertIn("0301e9299b3abe09c6a60796d7bed74c23fcc525", downloader)
        self.assertNotIn("resolve/master", downloader)
        self.assertIn('"size": 11194324', downloader)
        self.assertIn('"size": 12334256', downloader)
        self.assertIn('"size": 23303292', downloader)
        self.assertIn('"size": 19915466', downloader)
        self.assertIn('"size": 1046484', downloader)
        self.assertIn('"size": 1038858', downloader)
        self.assertIn(
            "tensorrt-jp6-trt10.4-orin-batch8-cls8",
            downloader,
        )
        self.assertIn(
            "tensorrt-jp511-trt8.5-orin-batch8-cls8",
            downloader,
        )
        self.assertIn("3b36aae43b2cc4a1", downloader)
        self.assertIn("1bb32a027e93b06d", downloader)
        self.assertIn("148a6895260d3b6b", downloader)
        self.assertIn("02c722e56e621b56", downloader)
        self.assertIn("tempfile.TemporaryDirectory", downloader)
        self.assertIn("for attempt in range(1, 4)", downloader)
        self.assertIn("urlopen(url, timeout=120)", downloader)
        self.assertIn("_verify_download(destination, metadata)", downloader)
        self.assertIn("os.replace(", downloader)
        self.assertNotIn("COPY perception/models", dockerfile)

        service = (REPO_ROOT / "perception" / "deploy" / "service.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("runtime: nvidia", service)

    def test_jetson_image_caps_cpu_threads_and_allocator_arenas(self):
        dockerfile = (REPO_ROOT / "perception" / "Dockerfile.jetson").read_text(
            encoding="utf-8"
        )

        self.assertIn("OMP_NUM_THREADS=1", dockerfile)
        self.assertIn("OPENBLAS_NUM_THREADS=1", dockerfile)
        self.assertIn("MKL_NUM_THREADS=1", dockerfile)
        self.assertIn("NUMEXPR_NUM_THREADS=1", dockerfile)
        self.assertIn("OPENCV_FOR_THREADS_NUM=1", dockerfile)
        self.assertIn("VIPS_CONCURRENCY=1", dockerfile)
        self.assertIn("MALLOC_ARENA_MAX=2", dockerfile)

    def test_jetson_image_loads_large_message_fastdds_profile(self):
        dockerfile = (REPO_ROOT / "perception" / "Dockerfile.jetson").read_text(
            encoding="utf-8"
        )
        service = (REPO_ROOT / "perception" / "deploy" / "service.yml").read_text(
            encoding="utf-8"
        )
        profile_path = (
            REPO_ROOT
            / "perception"
            / "config"
            / "fastdds_large_message.xml"
        )

        self.assertTrue(profile_path.is_file())
        profile = profile_path.read_text(encoding="utf-8")
        self.assertIn("<maxMessageSize>65000</maxMessageSize>", profile)
        self.assertIn("<sendBufferSize>8388608</sendBufferSize>", profile)
        self.assertIn("<receiveBufferSize>8388608</receiveBufferSize>", profile)
        self.assertIn("<useBuiltinTransports>true</useBuiltinTransports>", profile)
        self.assertIn(
            "COPY perception/config/fastdds_large_message.xml "
            "/opt/phanthy-motus/config/fastdds_large_message.xml",
            dockerfile,
        )
        self.assertIn(
            "FASTRTPS_DEFAULT_PROFILES_FILE="
            "/opt/phanthy-motus/config/fastdds_large_message.xml",
            service,
        )
        self.assertNotIn("FASTDDS_BUILTIN_TRANSPORTS", service)


if __name__ == "__main__":
    unittest.main()
