"""Audio language selection followed by a single resident offline recognizer."""

from __future__ import annotations

import gc
import io
import json
import logging
import os
from pathlib import Path
import selectors
import struct
import subprocess
import threading
import time
import wave

log = logging.getLogger(__name__)


class _LanguageIdentifier:
    REQUEST_TIMEOUT_S = 30.0

    def __init__(self, model, num_threads, languages):
        binary = os.environ.get("ASR_LANGUAGE_ID_BINARY", "/opt/asr-language-id/bin/whisper-lid")
        self._process = subprocess.Popen(
            [binary, str(model), str(num_threads), ",".join(languages)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0,
        )
        self._selector = selectors.DefaultSelector()
        self._selector.register(self._process.stdout, selectors.EVENT_READ)
        os.set_blocking(self._process.stdin.fileno(), False)
        self._write_selector = selectors.DefaultSelector()
        self._write_selector.register(self._process.stdin, selectors.EVENT_WRITE)
        try:
            if not self._read_result().get("ready"):
                raise RuntimeError("Language identifier did not initialize")
        except Exception:
            self.close()
            raise

    def _read_result(self, timeout=30.0):
        deadline = time.monotonic() + timeout
        line = bytearray()
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not self._selector.select(remaining):
                raise RuntimeError("Language identifier timed out")
            char = self._process.stdout.read(1)
            if not char:
                raise RuntimeError("Language identifier exited before returning a result")
            if char == b"\n":
                result = json.loads(line)
                if "error" in result:
                    raise RuntimeError(result["error"])
                return result
            line.extend(char)

    def identify(self, pcm):
        deadline = time.monotonic() + self.REQUEST_TIMEOUT_S
        try:
            payload = memoryview(struct.pack("<I", len(pcm) // 2) + pcm)
            while payload:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not self._write_selector.select(remaining):
                    raise RuntimeError("Language identifier timed out while accepting audio")
                try:
                    written = os.write(self._process.stdin.fileno(), payload)
                except BlockingIOError:
                    continue
                payload = payload[written:]
            result = self._read_result(timeout=max(0, deadline - time.monotonic()))
            return result["language"], float(result["probability"])
        except Exception as error:
            # A late response must never become the next utterance's language.
            self.close()
            if isinstance(error, OSError):
                raise RuntimeError("Language identifier is unavailable") from error
            raise

    def close(self):
        process = getattr(self, "_process", None)
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            for stream in (process.stdin, process.stdout):
                if stream is not None:
                    stream.close()
        selector = getattr(self, "_selector", None)
        if selector is not None:
            selector.close()
        write_selector = getattr(self, "_write_selector", None)
        if write_selector is not None:
            write_selector.close()


class IcefallMultilingualASRAdapter:
    """Keep language identification separate from the user's ASR weights.

    The legacy language argument is intentionally not a routing hint: callers
    that always send zh-CN must still recognize the audio's actual language.
    """

    def __init__(self, model_dir, device="cpu", num_threads=2):
        if device != "cpu":
            raise ValueError("Icefall multilingual ASR requires CPU INT8 weights")
        if int(num_threads) < 1:
            raise ValueError("num_threads must be positive")
        self._root = Path(model_dir)
        self._num_threads = int(num_threads)
        self._lock = threading.RLock()
        self._identifier = None
        self._recognizer = None
        self._current_language = None
        self._models = {}
        self.last_language = None
        self.last_language_probability = None

    def prepare(self):
        with self._lock:
            if self._identifier is not None:
                return
            manifest = json.loads((self._root / "manifest.json").read_text())
            models = manifest["languages"]
            for language, model in models.items():
                keys = ("encoder", "decoder", "joiner", "tokens") if model["type"] in ("transducer", "x-asr") else ("model", "tokens")
                if model["type"] == "x-asr":
                    keys += ("bpe_model", "bpe_vocab", "hotwords", "hotwords_bpe", "prefix_lm")
                if model["type"] not in ("transducer", "paraformer", "x-asr"):
                    raise ValueError("Unsupported recognizer type for " + language)
                for key in keys:
                    path = self._root / model[key]
                    if not path.is_file():
                        raise FileNotFoundError("Missing ASR artifact: " + str(path))
            lid_model = self._root / manifest["language_id"]["model"]
            if not lid_model.is_file():
                raise FileNotFoundError("Missing language identifier model: " + str(lid_model))
            self._models = models
            self._identifier = _LanguageIdentifier(lid_model, self._num_threads, tuple(models))

    def _load_recognizer(self, language):
        import sherpa_onnx

        spec = self._models[language]
        if spec["type"] == "x-asr":
            from plugins.x_asr import XASRAdapter

            return XASRAdapter(
                str((self._root / spec["tokens"]).parent), "cpu", self._num_threads,
                max_active_paths=spec["max_active_paths"],
                tail_padding_seconds=spec.get("tail_pad_ms", 300) / 1000.0,
                prefix_lm_path=str(self._root / spec["prefix_lm"]),
                prefix_lm_scale=spec["prefix_lm_scale"],
                entity_boost=spec["entity_boost"],
            )
        options = dict(num_threads=self._num_threads, provider="cpu")
        tokens = str(self._root / spec["tokens"])
        if spec["type"] == "paraformer":
            return sherpa_onnx.OfflineRecognizer.from_paraformer(
                paraformer=str(self._root / spec["model"]), tokens=tokens, **options)
        return sherpa_onnx.OfflineRecognizer.from_transducer(
            **{key: str(self._root / spec[key]) for key in ("encoder", "decoder", "joiner")},
            tokens=tokens, decoding_method="greedy_search", **options)

    def transcribe(self, wav_bytes, language="auto"):
        import numpy as np

        with wave.open(io.BytesIO(wav_bytes), "rb") as audio:
            if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) != (1, 2, 16000):
                raise ValueError("ASR expects mono 16kHz PCM16 WAV audio")
            pcm = audio.readframes(audio.getnframes())
        if not pcm:
            return ""
        with self._lock:
            self.prepare()
            try:
                detected, probability = self._identifier.identify(pcm)
            except Exception:
                self._identifier.close()
                self._identifier = None
                raise
            if detected not in self._models:
                raise RuntimeError("Unsupported detected language: " + detected)
            if detected != self._current_language:
                # Release the previous language before allocating the next one.
                self._recognizer = None
                self._current_language = None
                gc.collect()
                self._recognizer = self._load_recognizer(detected)
                self._current_language = detected
                log.info("[asr] active language=%s, provider=cpu", detected)
            self.last_language = detected
            self.last_language_probability = probability
            if self._models[detected]["type"] == "x-asr":
                return self._recognizer.transcribe(wav_bytes, "zh-CN")
            samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
            padding = int(self._models[detected].get("tail_pad_ms", 300)) * 16
            if padding:
                samples = np.pad(samples, (0, padding))
            stream = self._recognizer.create_stream()
            stream.accept_waveform(16000, samples)
            self._recognizer.decode_stream(stream)
            return stream.result.text.strip()

    def transcribe_rich(self, wav_bytes, language="auto", include_emotion=False):
        return self.transcribe(wav_bytes, language), {}

    def close(self):
        with self._lock:
            if self._identifier is not None:
                self._identifier.close()
            self._identifier = None
            self._recognizer = None
            self._current_language = None

    def __del__(self):
        if hasattr(self, "_lock"):
            self.close()
