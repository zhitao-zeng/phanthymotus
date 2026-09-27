"""
Which weight files the X-ASR adapter hands to sherpa-onnx, per device.

Run from the repo root:
    python -m pytest perception/tests -q

The cpu bundle is int8 and the gpu bundle fp32, with the same file names apart
from the `.int8` infix. Getting the infix wrong on cpu breaks the product path;
getting it wrong on gpu loads int8 weights under CUDA, which provider_for_device
catches only by silently falling back to cpu. `sherpa_onnx` is a stub that
records the recognizer arguments, as in test_onnx_provider.py.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

PERCEPTION_ROOT = Path(__file__).resolve().parents[1]
if str(PERCEPTION_ROOT) not in sys.path:
    sys.path.insert(0, str(PERCEPTION_ROOT))

from plugins.x_asr import XASRAdapter  # noqa: E402
from utils import onnx_provider  # noqa: E402

SHARED = ("decoder-epoch-99-avg-1.onnx", "tokens.txt", "bpe.model", "bpe.vocab",
          "hotwords.txt", "hotwords.bpe.txt")


@pytest.fixture(autouse=True)
def _clear_cache():
    onnx_provider.cuda_available.cache_clear()
    yield
    onnx_provider.cuda_available.cache_clear()


def _fake_sherpa(monkeypatch, tmp_path, *, with_cuda: bool) -> dict:
    pkg_dir = tmp_path / "sherpa_onnx"
    (pkg_dir / "lib").mkdir(parents=True)
    if with_cuda:
        (pkg_dir / "lib" / "libonnxruntime_providers_cuda.so").write_bytes(b"")
    calls = {}

    class OfflineRecognizer:
        @staticmethod
        def from_transducer(**kwargs):
            calls.update(kwargs)
            return object()

    module = types.ModuleType("sherpa_onnx")
    module.__file__ = str(pkg_dir / "__init__.py")
    module.OfflineRecognizer = OfflineRecognizer
    monkeypatch.setitem(sys.modules, "sherpa_onnx", module)
    return calls


def _bundle(root: Path, infix: str) -> Path:
    root.mkdir()
    for name in SHARED + (f"encoder-epoch-99-avg-1{infix}.onnx",
                          f"joiner-epoch-99-avg-1{infix}.onnx"):
        (root / name).write_bytes(b"")
    return root


def test_cpu_loads_int8_on_the_cpu(monkeypatch, tmp_path):
    calls = _fake_sherpa(monkeypatch, tmp_path, with_cuda=True)
    XASRAdapter(str(_bundle(tmp_path / "int8", ".int8")), "cpu")
    assert Path(calls["encoder"]).name == "encoder-epoch-99-avg-1.int8.onnx"
    assert Path(calls["joiner"]).name == "joiner-epoch-99-avg-1.int8.onnx"
    assert calls["provider"] == "cpu"


def test_gpu_loads_fp32_under_cuda(monkeypatch, tmp_path):
    calls = _fake_sherpa(monkeypatch, tmp_path, with_cuda=True)
    XASRAdapter(str(_bundle(tmp_path / "fp32", "")), "gpu")
    assert Path(calls["encoder"]).name == "encoder-epoch-99-avg-1.onnx"
    assert Path(calls["joiner"]).name == "joiner-epoch-99-avg-1.onnx"
    assert calls["provider"] == "cuda"


def test_gpu_without_the_cuda_wheel_falls_back_to_cpu(monkeypatch, tmp_path):
    calls = _fake_sherpa(monkeypatch, tmp_path, with_cuda=False)
    XASRAdapter(str(_bundle(tmp_path / "fp32", "")), "gpu")
    assert calls["provider"] == "cpu"


def test_gpu_on_the_int8_bundle_reports_what_is_missing(monkeypatch, tmp_path):
    _fake_sherpa(monkeypatch, tmp_path, with_cuda=True)
    with pytest.raises(FileNotFoundError, match="encoder-epoch-99-avg-1.onnx"):
        XASRAdapter(str(_bundle(tmp_path / "int8", ".int8")), "gpu")
