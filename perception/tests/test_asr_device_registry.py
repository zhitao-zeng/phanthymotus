"""
Invariants for the ASR (model, device) registry — no hardware or models needed.

Run from the repo root:
    python -m pytest perception/tests -q

These exist because the registry encodes measurements, and a plausible-looking
edit can silently undo them. Two mistakes in particular are cheap to make and
expensive to notice:

- pointing a `gpu` entry at int8 weights, which runs 1.25x-3.3x *slower* than the
  CPU because ONNX Runtime's CUDA provider has no int8 kernels, and
- pointing a `cpu` entry at fp16 weights, which measured 42890 ms against int8's
  3295 ms because ONNX Runtime has no fp16 CPU kernels.

Neither raises; both just make the product slow. Also asserted: the configSchema's
`device` visibility list matches the models that actually have gpu weights, so the
dashboard never offers a choice the plugin would reject.

sherpa_onnx is stubbed out because importing plugins.asr pulls it in transitively.
"""

from __future__ import annotations

import io
import struct
import sys
import types
import wave
from pathlib import Path

import pytest

PERCEPTION_ROOT = Path(__file__).resolve().parents[1]
if str(PERCEPTION_ROOT) not in sys.path:
    sys.path.insert(0, str(PERCEPTION_ROOT))

sys.modules.setdefault("sherpa_onnx", types.ModuleType("sherpa_onnx"))

from plugins import asr  # noqa: E402
from utils import model_downloader  # noqa: E402

VALID_DTYPES = {"int8", "fp32", "fp16"}


def _device_specs():
    for model, info in asr.ASR_MODELS.items():
        for device, spec in info["devices"].items():
            yield model, device, spec


def test_every_model_has_a_cpu_entry():
    """cpu is the fallback for an unsupported device request, so it must exist."""
    for model, info in asr.ASR_MODELS.items():
        assert "cpu" in info["devices"], f"{model} has no cpu weights"


def test_device_keys_are_known():
    for model, device, _ in _device_specs():
        assert device in ("cpu", "gpu"), f"{model} declares unknown device {device!r}"


def test_dtypes_are_declared_and_known():
    for model, device, spec in _device_specs():
        assert spec.get("dtype") in VALID_DTYPES, \
            f"{model}/{device} has dtype {spec.get('dtype')!r}"


def test_gpu_entries_are_never_int8():
    """ONNX Runtime's CUDA provider has no int8 kernels: it partitions the graph,
    falls back to CPU node by node, and adds a copy at every boundary. Measured
    1.25x-3.3x slower than the CPU, and it also perturbs the output (3 of 4
    SenseVoice transcripts changed versus the same model on CPU)."""
    for model, device, spec in _device_specs():
        if device == "gpu":
            assert spec["dtype"] != "int8", \
                f"{model}/gpu points at int8 weights, which is slower than cpu"


def test_cpu_entries_are_never_fp16():
    """ONNX Runtime has no fp16 CPU kernels and casts everything: 42890 ms where
    int8 took 3295 ms on the same audio."""
    for model, device, spec in _device_specs():
        if device == "cpu":
            assert spec["dtype"] != "fp16", \
                f"{model}/cpu points at fp16 weights, which is ~10x slower than int8"


def test_download_keys_resolve():
    """A gpu entry must name a pinned bundle; a cpu entry a legacy archive."""
    for model, device, spec in _device_specs():
        key = spec["download"]
        if device == "gpu":
            assert key in model_downloader.SHERPA_GPU_BUNDLES, \
                f"{model}/gpu download key {key!r} is not a SHERPA_GPU_BUNDLES entry"
        else:
            assert key in model_downloader.MODELS, \
                f"{model}/cpu download key {key!r} is not a MODELS entry"


def test_gpu_cache_hit_does_not_rehash_large_model(monkeypatch, tmp_path):
    payload = b"already-downloaded"
    bundle = {
        "base_url": "https://unused.invalid",
        "files": {
            "model.onnx": {
                "size": len(payload),
                # Deliberately not the payload digest: a cache hit must not read
                # the whole model merely to recompute SHA on every process start.
                "sha256": "0" * 64,
            },
        },
    }
    (tmp_path / "model.onnx").write_bytes(payload)
    monkeypatch.setitem(model_downloader.SHERPA_GPU_BUNDLES, "test_fast_cache", bundle)
    monkeypatch.setattr(model_downloader, "ensure_verified_bundle",
                        lambda *args, **kwargs: pytest.fail("cache was rehashed"))

    paths = model_downloader.ensure_gpu_model("test_fast_cache", str(tmp_path))
    assert paths == {"model.onnx": str(tmp_path / "model.onnx")}


def test_model_dirs_are_unique():
    """Two entries sharing a directory would download over each other's weights."""
    dirs = [spec["dir"] for _, _, spec in _device_specs()]
    assert len(dirs) == len(set(dirs)), "duplicate model_dir in ASR_MODELS"


def test_default_model_exists_and_is_the_schema_default():
    assert asr.DEFAULT_ASR_MODEL in asr.ASR_MODELS
    schema = asr.TOOLS[0]["configSchema"]["properties"]
    assert schema["asr_model"]["default"] == asr.DEFAULT_ASR_MODEL


def test_schema_enum_matches_the_registry():
    schema = asr.TOOLS[0]["configSchema"]["properties"]
    assert sorted(schema["asr_model"]["enum"]) == sorted(asr.ASR_MODELS)


def test_device_field_is_shown_exactly_for_models_with_gpu_weights():
    """Otherwise the dashboard offers gpu on a model whose config action rejects
    it, or hides it on a model that supports it."""
    schema = asr.TOOLS[0]["configSchema"]["properties"]
    shown_for = schema["device"]["x-show-when"]["asr_model"]
    assert sorted(shown_for) == asr.asr_models_supporting("gpu")


def test_device_schema_default_is_auto():
    schema = asr.TOOLS[0]["configSchema"]["properties"]
    assert schema["device"]["default"] == "auto"
    assert sorted(schema["device"]["enum"]) == ["auto", "cpu", "gpu"]


@pytest.mark.parametrize(("jp_version", "expected"), [
    ("61", "gpu"),
    ("6.1", "gpu"),
    ("511", "cpu"),
    ("5.1.1", "cpu"),
])
def test_auto_device_uses_measured_jetpack_split(monkeypatch, jp_version, expected):
    monkeypatch.setenv("JP_VERSION", jp_version)
    monkeypatch.delenv("JETPACK_VERSION", raising=False)
    assert asr._resolve_asr_device("auto") == expected


def test_auto_device_reads_image_record(monkeypatch, tmp_path):
    record = tmp_path / "jetpack.env"
    record.write_text("JP_VERSION=61\n", encoding="utf-8")
    monkeypatch.delenv("JP_VERSION", raising=False)
    monkeypatch.delenv("JETPACK_VERSION", raising=False)
    monkeypatch.setattr(asr, "_JETPACK_ENV_PATH", record)
    assert asr._resolve_asr_device("auto") == "gpu"


def test_auto_device_is_cpu_on_unknown_host(monkeypatch, tmp_path):
    monkeypatch.delenv("JP_VERSION", raising=False)
    monkeypatch.delenv("JETPACK_VERSION", raising=False)
    monkeypatch.setattr(asr, "_JETPACK_ENV_PATH", tmp_path / "missing")
    assert asr._resolve_asr_device("auto") == "cpu"


@pytest.mark.parametrize(("requested", "expected"), [
    ("cpu", "cpu"),
    ("gpu", "cpu"),
    ("cuda", "cpu"),
])
def test_jp511_never_selects_fp32_weights_for_its_cpu_only_wheel(
        monkeypatch, requested, expected):
    monkeypatch.setenv("JP_VERSION", "511")
    assert asr._resolve_asr_device(requested) == expected


def test_jp61_allows_explicit_gpu(monkeypatch):
    monkeypatch.setenv("JP_VERSION", "61")
    assert asr._resolve_asr_device("gpu") == "gpu"


def test_jp511_runtime_config_rejects_explicit_gpu(monkeypatch):
    monkeypatch.setenv("JP_VERSION", "511")
    plugin = asr.ASRPlugin.__new__(asr.ASRPlugin)
    plugin._language = "zh-CN"
    plugin._asr_model = "sensevoice-small"
    plugin._device = "cpu"

    result = plugin.dispatch("asr", {"action": "config", "device": "gpu"})
    assert result["status"] == "error"
    assert result["device"] == "cpu"
    assert "no CUDA sherpa-onnx wheel" in result["message"]


@pytest.mark.parametrize(("jp_version", "expected_device", "download_kind"), [
    ("61", "gpu", "gpu"),
    ("511", "cpu", "cpu"),
])
def test_build_auto_selects_the_matching_sensevoice_bundle(
        monkeypatch, jp_version, expected_device, download_kind):
    captured = {}
    downloads = []

    def adapter(model_dir, device, num_threads, **kwargs):
        captured.update(model_dir=model_dir, device=device,
                        num_threads=num_threads, **kwargs)
        return object()

    monkeypatch.setenv("JP_VERSION", jp_version)
    monkeypatch.setattr("utils.model_downloader.ensure_model",
                        lambda name, path: downloads.append(("cpu", name, path)))
    monkeypatch.setattr("utils.model_downloader.ensure_gpu_model",
                        lambda name, path: downloads.append(("gpu", name, path)))
    monkeypatch.setitem(asr.ASR_MODELS["sensevoice-small"], "adapter", adapter)

    asr._build_asr_adapter({
        "asr_model": "sensevoice-small",
        "device": "auto",
        "warmup": False,
    })

    spec = asr.ASR_MODELS["sensevoice-small"]["devices"][expected_device]
    assert captured["device"] == expected_device
    assert captured["model_dir"] == spec["dir"]
    assert downloads == [(download_kind, spec["download"], spec["dir"])]


def test_asr_models_supporting():
    assert "sensevoice-small" in asr.asr_models_supporting("gpu")
    assert "paraformer-zh-en" in asr.asr_models_supporting("gpu")
    # Measured 0.80x on CUDA — deliberately absent.
    assert "x-asr-zh-en" not in asr.asr_models_supporting("gpu")
    assert asr.asr_models_supporting("cpu") == sorted(asr.ASR_MODELS)


@pytest.mark.parametrize("cfg, expected_dir", [
    ({}, None),                                        # registry default
    ({"model_dir": "/custom/place"}, "/custom/place"),  # honoured
])
def test_model_dir_override(cfg, expected_dir):
    spec = asr.ASR_MODELS["sensevoice-small"]["devices"]["cpu"]
    got = asr._model_dir_for(cfg, spec)
    assert got == (expected_dir or spec["dir"])


def test_model_dir_from_another_entry_is_ignored():
    """config.yaml ships a model_dir for one bundle; reusing it for a different
    model or device would download the wrong weights into it."""
    spec = asr.ASR_MODELS["sensevoice-small"]["devices"]["gpu"]
    other = asr.ASR_MODELS["paraformer-zh-en"]["devices"]["cpu"]["dir"]
    assert asr._model_dir_for({"model_dir": other}, spec) == spec["dir"]


# ── warmup ───────────────────────────────────────────────────────────────────
#
# The first CUDA inference cost 1777 ms against a 77 ms steady state (lazy kernel
# loading, cuDNN autotuning, memory pool). Untouched, that lands on the operator's
# first utterance, right after the model finished loading.

def test_silence_wav_is_a_decodable_16k_mono_wav():
    import io
    import wave
    data = asr._silence_wav(0.5)
    with wave.open(io.BytesIO(data)) as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == asr.SAMPLE_RATE
        assert wf.getnframes() == int(asr.SAMPLE_RATE * 0.5)


def test_warmup_decodes_one_clip():
    calls = []

    class _Adapter:
        def transcribe(self, wav_bytes, language):
            calls.append((len(wav_bytes), language))
            return ""

    asr._warmup_adapter(_Adapter(), "sensevoice-small", "gpu")
    assert len(calls) == 1
    assert calls[0][0] > 0


@pytest.mark.parametrize("cfg, expect_warm", [
    ({}, True),                      # on by default — the gpu first-call cost
    ({"warmup": True}, True),
    ({"warmup": False}, False),      # opt out
])
def test_build_warms_up_unless_disabled(monkeypatch, cfg, expect_warm):
    warmed = []
    monkeypatch.setattr(asr, "_warmup_adapter",
                        lambda adapter, model, device: warmed.append((model, device)))
    monkeypatch.setattr("utils.model_downloader.ensure_model",
                        lambda *a, **k: None)
    monkeypatch.setitem(asr.ASR_MODELS["sensevoice-small"], "adapter",
                        lambda model_dir, device, num_threads, **kwargs: object())

    asr._build_asr_adapter({"asr_model": "sensevoice-small", "device": "cpu", **cfg})
    assert bool(warmed) is expect_warm


def test_warmup_failure_does_not_propagate():
    """A model that cannot decode silence still fails loudly on real audio; it must
    not stop the plugin from coming up."""
    class _Broken:
        def transcribe(self, wav_bytes, language):
            raise RuntimeError("no session")

    asr._warmup_adapter(_Broken(), "sensevoice-small", "gpu")  # must not raise


# ── SenseVoice phrase-level homophone replacement ───────────────────────────

def test_sensevoice_homophone_replacer_is_opt_in():
    assert asr._sensevoice_homophone_assets({}, "/models/sensevoice") == ("", "")


@pytest.mark.parametrize(("raw", "expected"), [
    ("小白的PM2 .5浓度是多少？", "小白的PM2.5浓度是多少？"),
    ("D 电影。", "D电影。"),
    ("Fancy ,fancy .Please visit .", "Fancy, fancy. Please visit."),
    ("hello ,你面前有什么", "hello,你面前有什么"),
])
def test_sensevoice_hr_spacing_preserves_product_format(raw, expected):
    assert asr._normalize_sensevoice_hr_spacing(raw) == expected


def test_sensevoice_homophone_assets_resolve_under_model_dir(tmp_path):
    hr_dir = tmp_path / "hr"
    hr_dir.mkdir()
    lexicon = hr_dir / "lexicon.txt"
    rule_fst = hr_dir / "replace.fst"
    lexicon.write_text("范 fan4\n", encoding="utf-8")
    rule_fst.write_bytes(b"fst")

    got = asr._sensevoice_homophone_assets(
        {"sensevoice_homophone_replacer": {"enabled": True}},
        str(tmp_path),
    )
    assert got == (str(lexicon), str(rule_fst))


def test_sensevoice_homophone_assets_fail_if_incomplete(tmp_path):
    with pytest.raises(FileNotFoundError, match="homophone-replacer assets"):
        asr._sensevoice_homophone_assets(
            {"sensevoice_homophone_replacer": {"enabled": True}},
            str(tmp_path),
        )


def test_build_passes_sensevoice_homophone_assets(monkeypatch, tmp_path):
    hr_dir = tmp_path / "hr"
    hr_dir.mkdir()
    lexicon = hr_dir / "lexicon.txt"
    rule_fst = hr_dir / "replace.fst"
    lexicon.write_text("范 fan4\n", encoding="utf-8")
    rule_fst.write_bytes(b"fst")
    captured = {}

    def adapter(model_dir, device, num_threads, **kwargs):
        captured.update(model_dir=model_dir, device=device,
                        num_threads=num_threads, **kwargs)
        return object()

    monkeypatch.setattr("utils.model_downloader.ensure_model",
                        lambda *args, **kwargs: None)
    monkeypatch.setitem(asr.ASR_MODELS["sensevoice-small"], "adapter", adapter)
    asr._build_asr_adapter({
        "asr_model": "sensevoice-small",
        "device": "cpu",
        "model_dir": str(tmp_path),
        "warmup": False,
        "sensevoice_homophone_replacer": {"enabled": True},
    })

    assert captured["hr_lexicon"] == str(lexicon)
    assert captured["hr_rule_fst"] == str(rule_fst)


def test_sensevoice_offline_decode_does_not_append_streaming_tail():
    accepted = {}

    class _Stream:
        result = types.SimpleNamespace(text="ok")

        def accept_waveform(self, sample_rate, samples):
            accepted["sample_rate"] = sample_rate
            accepted["samples"] = list(samples)

    class _Recognizer:
        def create_stream(self):
            return _Stream()

        def decode_streams(self, streams):
            assert len(streams) == 1

    pcm_samples = (1024, -2048, 4096)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(asr.SAMPLE_RATE)
        wav_file.writeframes(struct.pack("<3h", *pcm_samples))

    adapter = asr.SherpaOnnxSenseVoiceAdapter.__new__(
        asr.SherpaOnnxSenseVoiceAdapter
    )
    adapter._recognizer = _Recognizer()
    adapter._homophone_replacer_enabled = False

    assert adapter.transcribe(buffer.getvalue(), "zh-CN") == "ok"
    assert accepted["sample_rate"] == asr.SAMPLE_RATE
    assert accepted["samples"] == pytest.approx(
        [sample / 32768.0 for sample in pcm_samples]
    )
