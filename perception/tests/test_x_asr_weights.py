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


# ── entity boost ─────────────────────────────────────────────────────────────
#
# At score 2.5 the first token of 万神殿 ranked 31st-41st against beam 16, so the
# hotword bonus, added only to survivors, never reached it. The patched decoder
# ranks a hotword's next token early when its per-token score reaches
# SHERPA_ONNX_EARLY_HOTWORD_MIN_SCORE; only the boosted phrases may qualify.

HOTWORDS = "小 范 :2.5\n万 神 殿 :2.5\nPhanthyMovie :2.5\n"


def _boosted_bundle(root: Path) -> Path:
    _bundle(root, ".int8")
    (root / "hotwords.bpe.txt").write_text(HOTWORDS, encoding="utf-8")
    (root / "lm.onnx").write_bytes(b"")
    return root


def _prefix_runtime(monkeypatch, tmp_path):
    calls = _fake_sherpa(monkeypatch, tmp_path, with_cuda=False)
    monkeypatch.setattr(sys.modules["sherpa_onnx"], "XASR_PREFIX_LM_VERSION", 1, raising=False)
    monkeypatch.delenv("SHERPA_ONNX_EARLY_HOTWORD_MIN_SCORE", raising=False)
    return calls


def test_boost_raises_only_the_named_phrase_and_sets_the_early_threshold(monkeypatch, tmp_path):
    import os
    calls = _prefix_runtime(monkeypatch, tmp_path)
    root = _boosted_bundle(tmp_path / "b")
    XASRAdapter(str(root), "cpu", prefix_lm_path=str(root / "lm.onnx"), prefix_lm_scale=0.05,
                entity_boost={"万神殿": 4.0})
    boosted = Path(calls["hotwords_file"]).read_text(encoding="utf-8")
    assert boosted == "小 范 :2.5\n万 神 殿 :4.0\nPhanthyMovie :2.5\n"
    assert float(os.environ["SHERPA_ONNX_EARLY_HOTWORD_MIN_SCORE"]) == 3.9


def test_boost_of_a_phrase_that_is_not_a_hotword_fails(monkeypatch, tmp_path):
    _prefix_runtime(monkeypatch, tmp_path)
    root = _boosted_bundle(tmp_path / "b")
    with pytest.raises(ValueError, match="万神店"):
        XASRAdapter(str(root), "cpu", prefix_lm_path=str(root / "lm.onnx"), prefix_lm_scale=0.05,
                    entity_boost={"万神店": 4.0})


def test_boost_without_the_prefix_lm_fails(monkeypatch, tmp_path):
    _prefix_runtime(monkeypatch, tmp_path)
    root = _boosted_bundle(tmp_path / "b")
    with pytest.raises(ValueError, match="prefix LM"):
        XASRAdapter(str(root), "cpu", entity_boost={"万神殿": 4.0})


def test_no_boost_leaves_the_hotwords_and_threshold_alone(monkeypatch, tmp_path):
    import os
    calls = _prefix_runtime(monkeypatch, tmp_path)
    root = _boosted_bundle(tmp_path / "b")
    XASRAdapter(str(root), "cpu", prefix_lm_path=str(root / "lm.onnx"), prefix_lm_scale=0.05)
    assert Path(calls["hotwords_file"]) == root / "hotwords.bpe.txt"
    assert "SHERPA_ONNX_EARLY_HOTWORD_MIN_SCORE" not in os.environ
