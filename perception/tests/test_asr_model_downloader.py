import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class ASRModelDownloaderTest(unittest.TestCase):
    def test_jetson_model_destinations_match_product_config(self):
        perception_dir = Path(__file__).resolve().parents[1]
        dockerfile = (perception_dir / "Dockerfile.jetson").read_text()
        config = (perception_dir / "config.yaml").read_text()

        self.assertIn(
            "--output-dir /models/sherpa-onnx/x_asr_punct_int8",
            dockerfile,
        )
        self.assertIn("--output-dir /models/firered_vad", dockerfile)
        self.assertIn(
            "src/silero_vad/data/silero_vad.onnx",
            dockerfile,
        )
        self.assertIn("model_path: /models/sherpa-onnx/x_asr_punct_int8", config)
        self.assertIn("model_dir: /models/firered_vad", config)

    def test_downloads_complete_bundle_from_staging(self):
        from utils.asr_model_downloader import download_model

        with tempfile.TemporaryDirectory() as source_tmp:
            with tempfile.TemporaryDirectory() as output_tmp:
                source = Path(source_tmp)
                payloads = {
                    "config.json": b'{"model": "paraformer"}',
                    "model.int8.onnx": b"test-onnx-payload",
                    "tokens.txt": b"0 <blank>\n",
                }
                for filename, payload in payloads.items():
                    (source / filename).write_bytes(payload)

                download_model(
                    source.as_uri(),
                    output_tmp,
                    tuple(payloads),
                    retries=1,
                )

                for filename, payload in payloads.items():
                    self.assertEqual(
                        (Path(output_tmp) / filename).read_bytes(), payload
                    )

    def test_empty_download_preserves_existing_bundle(self):
        from utils.asr_model_downloader import download_model

        with tempfile.TemporaryDirectory() as source_tmp:
            with tempfile.TemporaryDirectory() as output_tmp:
                source = Path(source_tmp)
                (source / "model.onnx").write_bytes(b"")
                destination = Path(output_tmp) / "model.onnx"
                destination.write_bytes(b"known-good")

                with self.assertRaisesRegex(ValueError, "empty"):
                    download_model(
                        source.as_uri(),
                        output_tmp,
                        ("model.onnx",),
                        retries=1,
                    )

                self.assertEqual(destination.read_bytes(), b"known-good")

    def test_retry_recovers_from_transient_failure(self):
        from utils import asr_model_downloader

        response = mock.MagicMock()
        response.__enter__.return_value.read.side_effect = [b"model", b""]
        with tempfile.TemporaryDirectory() as output_tmp:
            with mock.patch.object(
                asr_model_downloader,
                "urlopen",
                side_effect=[OSError("temporary"), response],
            ), mock.patch.object(asr_model_downloader.time, "sleep"):
                asr_model_downloader.download_model(
                    "https://models.example.test",
                    output_tmp,
                    ("model.onnx",),
                    retries=2,
                )

            self.assertEqual(
                (Path(output_tmp) / "model.onnx").read_bytes(), b"model"
            )

    def test_rejects_path_traversal_filename(self):
        from utils.asr_model_downloader import download_model

        with tempfile.TemporaryDirectory() as output_tmp:
            with self.assertRaisesRegex(ValueError, "plain file names"):
                download_model(
                    "https://models.example.test",
                    output_tmp,
                    ("../secret",),
                    retries=1,
                )


if __name__ == "__main__":
    unittest.main()
