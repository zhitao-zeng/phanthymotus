"""Protocol and numeric behaviour of the CUDA Whisper Turbo adapter.

The real worker needs CUDA and the 548 MB GGML model; these tests replace it
with a tiny executable that speaks the same JSON lines, so the adapter's
process lifecycle, language tag and numeric rendering are covered without a
Jetson or a model.
"""
from __future__ import annotations

import io
import json
import sys
import wave
from pathlib import Path

PERCEPTION_ROOT = Path(__file__).resolve().parents[1]
if str(PERCEPTION_ROOT) not in sys.path:
    sys.path.insert(0, str(PERCEPTION_ROOT))

from plugins.whisper_turbo import WhisperTurboAdapter  # noqa: E402


def _wav_bytes(seconds: float = 0.1) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(b"\x00\x00" * int(16000 * seconds))
    return buffer.getvalue()


def _fake_worker(path: Path, text: str = "hello 2024", language: str = "en") -> None:
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import json, sys\n"
        "print(json.dumps({'ready': True, 'gpu': True, 'load_s': 0.01}), flush=True)\n"
        "for line in sys.stdin:\n"
        f"    print(json.dumps({{'raw_text': {text!r}, 'predicted_language': {language!r}}}), flush=True)\n")
    path.chmod(0o755)


def test_transcribe_rich_returns_text_and_language(tmp_path, monkeypatch):
    worker = tmp_path / "worker"
    _fake_worker(worker)
    (tmp_path / "model.bin").write_bytes(b"model")
    monkeypatch.setenv("WHISPER_TURBO_WORKER", str(worker))
    adapter = WhisperTurboAdapter(str(tmp_path), "gpu", 2)
    try:
        text, tags = adapter.transcribe_rich(_wav_bytes(), "auto")
    finally:
        adapter.close()
    assert text == "hello 2024"
    assert tags == {"lang": "en"}
    assert adapter.last_language == "en"


def test_unknown_language_is_not_published(tmp_path, monkeypatch):
    worker = tmp_path / "worker"
    _fake_worker(worker, text="plain", language="xx")
    (tmp_path / "model.bin").write_bytes(b"model")
    monkeypatch.setenv("WHISPER_TURBO_WORKER", str(worker))
    adapter = WhisperTurboAdapter(str(tmp_path), "gpu", 2)
    try:
        text, tags = adapter.transcribe_rich(_wav_bytes(), "auto")
    finally:
        adapter.close()
    assert text == "plain"
    assert tags == {}


def test_cpu_is_rejected():
    import pytest
    with pytest.raises(ValueError):
        WhisperTurboAdapter("/does/not/exist", "cpu", 2)
