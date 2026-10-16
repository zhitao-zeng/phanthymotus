"""Offline contracts for automatic language routing and recognizer ownership."""

import gc
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import wave
import weakref

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plugins import icefall_asr
from utils import install_icefall_models


def wav():
    output = io.BytesIO()
    with wave.open(output, "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(b"\x01\x00" * 16000)
    return output.getvalue()


class FakeIdentifier:
    selected = "en"

    def __init__(self, *args, **kwargs):
        self.closed = False

    def identify(self, pcm):
        return self.selected, 0.9

    def close(self):
        self.closed = True


class FakeRecognizer:
    def __init__(self, language):
        self.language = language

    def create_stream(self):
        return mock.Mock(result=mock.Mock(text=self.language))

    def decode_stream(self, stream):
        pass


class RoutingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        models = {}
        for language in ("en", "ja", "zh"):
            models[language] = {"type": "paraformer" if language == "zh" else "transducer",
                                "model": language + "/model.onnx",
                                "tokens": language + "/tokens.txt"}
            if language != "zh":
                models[language].update({key: language + "/" + key + ".onnx"
                                         for key in ("encoder", "decoder", "joiner")})
            for key in (key for key in models[language] if key != "type"):
                path = self.root / models[language][key]
                path.parent.mkdir(exist_ok=True)
                path.write_bytes(b"test")
        (self.root / "lid.bin").write_bytes(b"test")
        (self.root / "manifest.json").write_text(json.dumps({
            "languages": models, "language_id": {"model": "lid.bin"}}))
        FakeIdentifier.selected = "en"
        self.worker = mock.patch.object(icefall_asr, "_LanguageIdentifier", FakeIdentifier)
        self.worker.start()
        self.adapter = icefall_asr.IcefallMultilingualASRAdapter(str(self.root))
        self.loaded = []
        def load(language):
            self.loaded.append(language)
            return FakeRecognizer(language)
        self.loader = mock.patch.object(self.adapter, "_load_recognizer", side_effect=load)
        self.loader.start()

    def tearDown(self):
        self.adapter.close()
        self.loader.stop()
        self.worker.stop()
        self.temp.cleanup()

    def test_construction_does_not_start_worker_or_load_recognizer(self):
        self.assertIsNone(self.adapter._identifier)
        self.assertEqual(self.loaded, [])

    def test_audio_language_wins_over_a_fixed_legacy_config(self):
        FakeIdentifier.selected = "ja"
        self.assertEqual(self.adapter.transcribe(wav(), "zh-CN"), "ja")
        self.assertEqual(self.loaded, ["ja"])

    def test_switch_releases_previous_recognizer(self):
        self.adapter.transcribe(wav(), "auto")
        previous = weakref.ref(self.adapter._recognizer)
        FakeIdentifier.selected = "ja"
        self.adapter.transcribe(wav(), "auto")
        gc.collect()
        self.assertIsNone(previous())
        self.assertEqual(self.loaded, ["en", "ja"])

    def test_same_language_reuses_recognizer(self):
        self.adapter.transcribe(wav(), "auto")
        self.adapter.transcribe(wav(), "auto")
        self.assertEqual(self.loaded, ["en"])

    def test_prepare_checks_assets_before_input(self):
        (self.root / "ja/encoder.onnx").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "ja/encoder.onnx"):
            self.adapter.prepare()
        self.assertEqual(self.loaded, [])

    def test_identifier_failure_does_not_use_previous_language(self):
        self.adapter.transcribe(wav(), "auto")
        with mock.patch.object(self.adapter._identifier, "identify", side_effect=RuntimeError("worker exited")):
            with self.assertRaisesRegex(RuntimeError, "worker exited"):
                self.adapter.transcribe(wav(), "auto")
        self.assertEqual(self.loaded, ["en"])

    def test_close_allows_a_clean_restart(self):
        self.adapter.prepare()
        identifier = self.adapter._identifier
        self.adapter.close()
        self.assertTrue(identifier.closed)
        self.assertIsNone(self.adapter._recognizer)
        self.adapter.prepare()
        self.assertIsNot(identifier, self.adapter._identifier)

    def test_wrong_audio_format_fails_before_language_selection(self):
        output = io.BytesIO()
        with wave.open(output, "wb") as stream:
            stream.setnchannels(2)
            stream.setsampwidth(2)
            stream.setframerate(16000)
            stream.writeframes(b"\0" * 160)
        with self.assertRaisesRegex(ValueError, "mono"):
            self.adapter.transcribe(output.getvalue(), "auto")
        self.assertIsNone(self.adapter._identifier)

    def test_plugin_config_then_start_defers_runtime_preparation(self):
        import vision_stubs  # noqa: F401
        from plugins import asr

        config = {"asr_model": "icefall-multilingual", "model_dir": str(self.root),
                  "device": "cpu", "warmup": False}
        node = mock.Mock(state="idle")
        node.start.return_value = {"state": "running"}
        with mock.patch.object(icefall_asr.IcefallMultilingualASRAdapter, "prepare") as prepare:
            plugin = asr.ASRPlugin(config, mock.Mock())
            plugin.dispatch("asr", {"action": "info"})
            plugin.dispatch("asr", {"action": "config", "language": "zh-CN"})
            prepare.assert_not_called()
            with mock.patch.object(asr, "_ASRNode", return_value=node):
                result = plugin.dispatch("asr", {"action": "start", "input_topic": "/test/audio"})
            self.assertEqual(result["state"], "running")
            prepare.assert_called_once()


class ModelReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.data = b"runtime model"
        self.manifest = json.dumps({"files": [{"path": "en/model.onnx", "size": len(self.data),
            "sha256": hashlib.sha256(self.data).hexdigest()}]}).encode()
        self.digest = hashlib.sha256(self.manifest).hexdigest()

    def tearDown(self):
        self.temp.cleanup()

    def test_invalid_manifest_does_not_install_any_file(self):
        opener = mock.Mock()
        opener.open.return_value = io.BytesIO(self.manifest)
        with mock.patch.object(install_icefall_models.urllib.request, "build_opener", return_value=opener):
            with self.assertRaisesRegex(RuntimeError, "manifest checksum"):
                install_icefall_models.install("http://models.invalid", "wrong", self.root)
        self.assertFalse((self.root / "en/model.onnx").exists())

    def test_invalid_artifact_is_not_published_as_a_complete_model(self):
        opener = mock.Mock()
        opener.open.side_effect = lambda url, **kwargs: io.BytesIO(self.manifest if url.endswith("release.json") else b"bad")
        with mock.patch.object(install_icefall_models.urllib.request, "build_opener", return_value=opener), \
                mock.patch.object(install_icefall_models.time, "sleep"):
            with self.assertRaisesRegex(RuntimeError, "artifact verification"):
                install_icefall_models.install("http://models.invalid", self.digest, self.root)
        self.assertFalse((self.root / "en/model.onnx").exists())
        self.assertFalse((self.root / "en/model.onnx.part").exists())

    def test_verified_existing_artifact_is_reused(self):
        path = self.root / "en/model.onnx"
        path.parent.mkdir()
        path.write_bytes(self.data)
        opener = mock.Mock()
        opener.open.return_value = io.BytesIO(self.manifest)
        with mock.patch.object(install_icefall_models.urllib.request, "build_opener", return_value=opener):
            install_icefall_models.install("http://models.invalid", self.digest, self.root)
        self.assertEqual(opener.open.call_count, 1)


if __name__ == "__main__":
    unittest.main()
