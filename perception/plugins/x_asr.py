"""Offline X-ASR transducer adapter for the product ASR plugin."""

from __future__ import annotations

import hashlib
import io
import logging
import os
import struct
import tempfile
import threading
import wave
from pathlib import Path


SAMPLE_RATE = 16000
# Defaults for the bundled model. Deployments can override both values through
# asr_beam_paths and asr_tail_pad_ms.
TAIL_PADDING_SECONDS = 0.3
MAX_ACTIVE_PATHS = 3
HOTWORDS_SCORE = 2.5

log = logging.getLogger(__name__)


def _prepare_hotwords_file(
    source: Path,
    score: float = HOTWORDS_SCORE,
    output_dir: Path = Path("/tmp/asr_x_asr_hotwords"),
) -> Path:
    """Convert one phrase per line to sherpa's character-separated BPE form."""
    source_bytes = source.read_bytes()
    digest = hashlib.sha256(
        source_bytes + b"\0" + str(float(score)).encode("ascii")
    ).hexdigest()[:16]
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"hotwords-char-bpe-{digest}.txt"
    if output.is_file():
        return output

    seen: set[str] = set()
    lines: list[str] = []
    for raw_line in source_bytes.decode("utf-8").splitlines():
        phrase = raw_line.strip()
        if not phrase or phrase.startswith("#"):
            continue
        compact = "".join(phrase.split())
        if not compact or compact in seen:
            continue
        seen.add(compact)
        lines.append(f"{' '.join(compact)} :{float(score)}\n")

    if not lines:
        raise ValueError(f"No usable hotwords in {source}")
    return _write_atomically(output, lines)


def _boost_hotwords(
    encoded: Path,
    boosts: dict,
    output_dir: Path = Path("/tmp/asr_x_asr_hotwords"),
) -> Path:
    """Raise the score of the named phrases in an encoded hotwords file.

    Keys are compact phrases ("万神殿"); every key must already be a hotword,
    so a typo fails here instead of silently boosting nothing.
    """
    boosts = {"".join(str(k).split()): float(v) for k, v in boosts.items()}
    source_bytes = encoded.read_bytes()
    digest = hashlib.sha256(
        source_bytes + b"\0" + repr(sorted(boosts.items())).encode("utf-8")
    ).hexdigest()[:16]
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"hotwords-boosted-{digest}.txt"
    if output.is_file():
        return output

    found: set[str] = set()
    lines: list[str] = []
    for raw_line in source_bytes.decode("utf-8").splitlines():
        phrase, sep, _ = raw_line.rpartition(" :")
        compact = "".join(phrase.split())
        if sep and compact in boosts:
            raw_line = f"{phrase} :{boosts[compact]}"
            found.add(compact)
        lines.append(raw_line + "\n")
    missing = sorted(set(boosts) - found)
    if missing:
        raise ValueError(f"Boosted phrases are not hotwords in {encoded}: {', '.join(missing)}")
    return _write_atomically(output, lines)


def _write_atomically(output: Path, lines: list[str]) -> Path:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output.parent,
            prefix=f".{output.name}.",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.writelines(lines)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return output


class XASRAdapter:
    """Decode complete utterances with X-ASR and the packaged hotword list."""

    def __init__(
        self,
        model_dir: str,
        device: str = "cpu",
        num_threads: int = 2,
        max_active_paths: int = None,
        tail_padding_seconds: float = None,
        prefix_lm_path: str = "",
        prefix_lm_scale: float = 0.0,
        entity_boost: dict = None,
    ):
        from utils.onnx_provider import provider_for_device

        self._max_active_paths = (
            MAX_ACTIVE_PATHS if max_active_paths is None else int(max_active_paths)
        )
        self._tail_padding_seconds = (
            TAIL_PADDING_SECONDS
            if tail_padding_seconds is None
            else float(tail_padding_seconds)
        )
        root = Path(model_dir)
        # The cpu bundle is int8 (encoder+joiner) and the gpu bundle fp32; the
        # decoder is the same fp32 file in both. ASR_MODELS picks the directory.
        dtype = "" if device == "gpu" else ".int8"
        encoder = root / f"encoder-epoch-99-avg-1{dtype}.onnx"
        decoder = root / "decoder-epoch-99-avg-1.onnx"
        joiner = root / f"joiner-epoch-99-avg-1{dtype}.onnx"
        tokens = root / "tokens.txt"
        bpe_model = root / "bpe.model"
        bpe_vocab = root / "bpe.vocab"
        hotwords = root / "hotwords.txt"
        required = (
            encoder,
            decoder,
            joiner,
            tokens,
            bpe_model,
            bpe_vocab,
            hotwords,
        )
        missing = [path.name for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"X-ASR model bundle is incomplete at {root}: {', '.join(missing)}"
            )

        import sherpa_onnx

        lm_scale = float(prefix_lm_scale)
        if not 0.0 <= lm_scale <= 1.0:
            raise ValueError("prefix_lm_scale must be between 0 and 1")
        lm_options = {}
        if lm_scale > 0:
            if getattr(sherpa_onnx, "XASR_PREFIX_LM_VERSION", 0) != 1:
                raise RuntimeError("X-ASR prefix LM requires the prefix-enabled sherpa-onnx runtime")
            if not prefix_lm_path or not Path(prefix_lm_path).is_file():
                raise FileNotFoundError("X-ASR prefix LM model is missing")
            lm_options = {"lm": prefix_lm_path, "lm_scale": lm_scale}

        # Refuses int8 on the GPU and falls back to cpu when the installed wheel
        # has no CUDA provider, so a mismatched device/bundle is reported.
        provider = provider_for_device(device,
                                       (str(encoder), str(decoder), str(joiner)))
        encoded_hotwords = root / "hotwords.bpe.txt"
        if not encoded_hotwords.is_file():
            encoded_hotwords = _prepare_hotwords_file(hotwords)
        early_min = None
        if entity_boost:
            # The patched decoder adds a hotword's next-token score to the pruning
            # rank, not only to the survivors, for hotwords whose per-token score
            # reaches SHERPA_ONNX_EARLY_HOTWORD_MIN_SCORE. It reads the variable at
            # every decode and only ranks early alongside the prefix LM. The
            # threshold sits just under the boosted scores so the plain 2.5
            # hotwords keep their usual, after-pruning effect.
            if not lm_options:
                raise ValueError("entity_boost needs the prefix LM (prefix_lm_scale > 0)")
            early_min = min(float(v) for v in entity_boost.values()) - 0.1
            if early_min <= HOTWORDS_SCORE:
                raise ValueError(f"entity_boost scores must exceed {HOTWORDS_SCORE + 0.1}")
            encoded_hotwords = _boost_hotwords(encoded_hotwords, entity_boost)
            os.environ["SHERPA_ONNX_EARLY_HOTWORD_MIN_SCORE"] = f"{early_min:g}"
        self._recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(encoder),
            decoder=str(decoder),
            joiner=str(joiner),
            tokens=str(tokens),
            num_threads=int(num_threads),
            provider=provider,
            sample_rate=SAMPLE_RATE,
            feature_dim=80,
            decoding_method="modified_beam_search",
            max_active_paths=self._max_active_paths,
            hotwords_file=str(encoded_hotwords),
            hotwords_score=HOTWORDS_SCORE,
            modeling_unit="bpe",
            bpe_vocab=str(bpe_vocab),
            **lm_options,
        )
        self._decode_lock = threading.Lock()
        log.info(
            "[asr] X-ASR adapter loaded: encoder=%s, device=%s, provider=%s, "
            "max_active_paths=%d, tail_padding=%.2fs, hotwords_score=%.1f, prefix_lm_scale=%.3f, "
            "entity_boost=%s early_min=%s",
            encoder,
            device,
            provider,
            self._max_active_paths,
            self._tail_padding_seconds,
            HOTWORDS_SCORE,
            lm_scale,
            entity_boost or {},
            early_min,
        )

    def transcribe(self, wav_bytes: bytes, language: str) -> str:
        del language
        with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
            if wav_file.getnchannels() != 1 or wav_file.getsampwidth() != 2:
                raise ValueError("X-ASR expects mono 16-bit PCM WAV")
            sample_rate = wav_file.getframerate()
            pcm = wav_file.readframes(wav_file.getnframes())

        sample_count = len(pcm) // 2
        samples = [
            sample / 32768.0
            for sample in struct.unpack(f"<{sample_count}h", pcm)
        ]
        samples.extend([0.0] * int(sample_rate * self._tail_padding_seconds))

        with self._decode_lock:
            stream = self._recognizer.create_stream()
            stream.accept_waveform(sample_rate, samples)
            self._recognizer.decode_streams([stream])
            result = stream.result
        return str(getattr(result, "text", result or "")).strip()
