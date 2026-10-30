"""Whisper large-v3-turbo Q5 adapter for the ASR plugin.

One persistent whisper.cpp CUDA worker holds the model; the plugin sends each
VAD segment as a temporary WAV and receives text plus whisper's own language
decision. No separate LID model is loaded for this profile.
"""
from __future__ import annotations

import atexit
import ctypes.util
import io
import json
import logging
import os
import select
import subprocess
import tempfile
import threading
import time
import wave
from pathlib import Path

from plugins.asr_numeric import NumericOutputNormalizer
from plugins.numeric_dictation import normalize_long_numeric_dictation

log = logging.getLogger(__name__)

DEFAULT_WORKER = "/opt/whisper-turbo/bin/whisper_turbo_worker"
DEFAULT_NUMERIC_MANIFEST = "/opt/asr-models/icefall/manifest.json"
LANGUAGES = ("en", "zh", "ja", "fr", "de", "es", "ru", "th")
CALL_TIMEOUT_S = 130.0
LOAD_TIMEOUT_S = 60.0


class WhisperTurboAdapter:
    """Offline multilingual ASR with Whisper's built-in language identification."""

    def __init__(self, model_dir: str, device: str = "gpu", num_threads: int = 2):
        if device != "gpu":
            raise ValueError("Whisper Turbo Q5 requires the CUDA worker")
        self._root = Path(model_dir)
        self._model = self._root / "model.bin"
        self._worker = Path(os.environ.get("WHISPER_TURBO_WORKER", DEFAULT_WORKER))
        self._num_threads = int(num_threads)
        self._lock = threading.RLock()
        self._process = None
        self._numeric = None
        self.last_language = None
        self.last_language_probability = None
        self._load_numeric()
        atexit.register(self.close)

    def _load_numeric(self):
        manifest = self._root / "numeric.json"
        source = manifest if manifest.is_file() else Path(DEFAULT_NUMERIC_MANIFEST)
        if not source.is_file():
            log.warning("[asr] whisper-turbo: no numeric manifest at %s; "
                        "using the legacy numeric dictation normalizer", source)
            return
        try:
            data = json.loads(source.read_text())
            config = data.get("numeric_output") if "numeric_output" in data else data
            if not isinstance(config, dict):
                return
            grammar_dir = config.get("grammar_dir")
            if not isinstance(grammar_dir, str):
                return
            base = source.parent
            path = (base / grammar_dir) if not os.path.isabs(grammar_dir) else Path(grammar_dir)
            self._numeric = NumericOutputNormalizer(path, config.get("model_names", ()))
            log.info("[asr] whisper-turbo numeric normalizer: %s", path)
        except Exception as exc:
            log.warning("[asr] whisper-turbo numeric normalizer unavailable: %s", exc)
            self._numeric = None

    def _render(self, text: str, language: str) -> str:
        if self._numeric is not None:
            return self._numeric.normalize(text, language)
        return normalize_long_numeric_dictation(text, language)

    def _start(self):
        if self._process is not None and self._process.poll() is None:
            return
        if not self._worker.is_file():
            raise FileNotFoundError("Whisper Turbo worker not found: %s" % self._worker)
        if not self._model.is_file():
            raise FileNotFoundError("Whisper Turbo model not found: %s" % self._model)
        env = dict(os.environ)
        env.setdefault("CUDA_CACHE_PATH", "/tmp/whisper-turbo")
        # libggml-cuda.so is linked without a DT_NEEDED on the driver API; the
        # Jetson runtime provides libcuda.so.1 but the loader will not pull it
        # in unless something asks. The product CMD already preloads libgomp,
        # so append rather than replace.
        preload = [item for item in env.get("LD_PRELOAD", "").split() if item]
        libcuda = ctypes.util.find_library("cuda")
        if libcuda and libcuda not in preload:
            preload.append(libcuda)
        if preload:
            env["LD_PRELOAD"] = " ".join(preload)
        self._process = subprocess.Popen(
            [str(self._worker), str(self._model)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, bufsize=1, env=env)
        ready = self._read(LOAD_TIMEOUT_S)
        if not ready.get("ready") or not ready.get("gpu"):
            raise RuntimeError("Whisper Turbo worker did not open on CUDA: %r" % ready)
        log.info("[asr] whisper-turbo worker ready: load=%.2fs model=%s",
                 ready.get("load_s", 0.0), self._model)

    def _read(self, timeout: float) -> dict:
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("whisper-turbo worker response timeout")
            ready, _, _ = select.select([self._process.stdout], [], [], remaining)
            if not ready:
                raise TimeoutError("whisper-turbo worker response timeout")
            line = self._process.stdout.readline()
            if not line:
                raise RuntimeError("whisper-turbo worker exited")
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                log.warning("[asr] whisper-turbo ignored non-JSON worker line: %r", line[:200])

    def _transcribe(self, wav_bytes: bytes, language: str) -> tuple[str, str]:
        with wave.open(io.BytesIO(wav_bytes), "rb") as audio:
            if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) != (1, 2, 16000):
                raise ValueError("ASR expects mono 16kHz PCM16 WAV audio")
            pcm = audio.readframes(audio.getnframes())
        if not pcm:
            return "", ""
        with self._lock:
            self._start()
            handle, path = tempfile.mkstemp(prefix="whisper-turbo-", suffix=".wav")
            os.close(handle)
            try:
                with wave.open(path, "wb") as out:
                    out.setnchannels(1)
                    out.setsampwidth(2)
                    out.setframerate(16000)
                    out.writeframes(pcm)
                self._process.stdin.write(json.dumps({"audio": path}) + "\n")
                self._process.stdin.flush()
                reply = self._read(CALL_TIMEOUT_S)
            finally:
                try:
                    os.unlink(path)
                except OSError:
                    pass
        if reply.get("error"):
            raise RuntimeError("whisper-turbo worker error: " + str(reply["error"]))
        detected = str(reply.get("predicted_language") or "").strip().lower()
        if detected not in LANGUAGES:
            detected = ""
        self.last_language = detected or None
        self.last_language_probability = None
        text = str(reply.get("raw_text") or "").strip()
        return self._render(text, detected), detected

    def transcribe(self, wav_bytes: bytes, language: str = "auto") -> str:
        return self._transcribe(wav_bytes, language)[0]

    def transcribe_rich(self, wav_bytes: bytes, language: str = "auto",
                        include_emotion: bool = False) -> tuple[str, dict]:
        text, detected = self._transcribe(wav_bytes, language)
        return text, ({"lang": detected} if detected else {})

    def transcribe_aligned(self, wav_bytes: bytes, language: str = "auto",
                           include_emotion: bool = False):
        text, detected = self._transcribe(wav_bytes, language)
        tags = {"lang": detected} if detected else {}
        return text, tags, None

    def close(self):
        with self._lock:
            process, self._process = self._process, None
        if process is None:
            return
        try:
            if process.stdin:
                process.stdin.close()
        except OSError:
            pass
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)

    def __del__(self):
        if hasattr(self, "_lock"):
            self.close()
