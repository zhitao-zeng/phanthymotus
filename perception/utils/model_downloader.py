"""Download perception models into the shared runtime cache when missing."""

from __future__ import annotations

import fcntl
import hashlib
import logging
import os
import shutil
import tarfile
import tempfile
import time
import zipfile
from urllib.request import urlopen, urlretrieve

log = logging.getLogger(__name__)

COS_BASE = "https://agi-phanthy-dev-1252788780.cos.ap-beijing.myqcloud.com/public"
OBSTACLE_MODEL_REVISION = "b8ba6d69a819b5ed6f0c1c5723b37c8775fa737b"
OBSTACLE_MODEL_BASE = (
    "https://www.modelscope.cn/models/Flame4pd/"
    "obstacle-distance-jetson-int8/resolve"
)
OBSTACLE_ENGINE_BUNDLES = {
    "jp61": {
        "zipdepth-base-npu-512x384-int8.engine": (
            7935428,
            "aa34296bcaeed28a5176b423f074da3923c996e7be06702a3952d475000a8887",
        ),
        "yolo26n-depth-int8.engine": (
            7778158,
            "8174652d6ba72af15c10caccf95629d585d33245e5242aa1f1734317d5a23f7c",
        ),
        "yolo26n-seg-int8.engine": (
            5641961,
            "7cb85598bc50b82ab5835102dab9214f6e58a0061c6a1891ee018387346bae30",
        ),
    },
    "jp511": {
        "zipdepth-base-npu-512x384-int8.engine": (
            7936960,
            "61d9b81c81bcd26660d3647bfb86fd133f865ad5b73b4177efcad2884f7a2d1c",
        ),
        "yolo26n-depth-int8.engine": (
            6746230,
            "816ca14c23af37ee2961ec09db51a54888462c5e4bb296bbaba2a569e6f2bb64",
        ),
        "yolo26n-seg-int8.engine": (
            4920200,
            "ed5e0f8dcb968440866f5b0433f7b813d14f2e89f810be6e8910945e0af42635",
        ),
    },
}


def _progress_hook(name: str):
    """Create a reporthook for urlretrieve that logs download progress."""
    last_pct = [0]
    def hook(block_num, block_size, total_size):
        if total_size > 0:
            pct = min(int(block_num * block_size * 100 / total_size), 100)
            if pct >= last_pct[0] + 10:
                last_pct[0] = pct
                mb_done = block_num * block_size / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                log.info(f"[model_downloader] {name}: {pct}% ({mb_done:.1f}/{mb_total:.1f} MB)")
    return hook

MODELS = {
    "asr": {
        "url": f"{COS_BASE}/sherpa-onnx-streaming-paraformer-bilingual-zh-en.zip",
        "check_file": "tokens.txt",
    },
    "asr_en": {
        "url": f"{COS_BASE}/sherpa-onnx-streaming-zipformer-en-2023-06-26.zip",
        "check_file": "tokens.txt",
    },
    "asr_sensevoice": {
        "url": f"{COS_BASE}/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.zip",
        "check_file": "tokens.txt",
    },
    "asr_paraformer_offline": {
        "url": f"{COS_BASE}/sherpa-onnx-paraformer-zh-small-2024-03-09.tar.bz2",
        "check_file": "tokens.txt",
    },
    "tts": {
        "url": f"{COS_BASE}/matcha-icefall-zh-en.tar.bz2",
        "check_file": "model-steps-3.onnx",
    },
    "tts_vocoder": {
        "url": f"{COS_BASE}/vocos-16khz-univ.onnx",
        "check_file": "vocos-16khz-univ.onnx",
        "single_file": True,
    },
    "kws": {
        "url": f"{COS_BASE}/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20.tar.bz2",
        "check_file": "tokens.txt",
    },
    "kws_zh": {
        "url": f"{COS_BASE}/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01.zip",
        "check_file": "tokens.txt",
    },
    "kws_en": {
        "url": f"{COS_BASE}/sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01.zip",
        "check_file": "tokens.txt",
    },
    "vad": {
        "url": f"{COS_BASE}/silero_vad.onnx",
        "check_file": "silero_vad.onnx",
        "single_file": True,  # Not an archive, just a single file download
    },
    "denoise": {
        "url": f"{COS_BASE}/gtcrn_simple.onnx",
        "check_file": "gtcrn_simple.onnx",
        "single_file": True,
    },
}


def _obstacle_bundle_name(bundle: str | None = None) -> str:
    value = bundle or os.environ.get("OBSTACLE_MODEL_BUNDLE")
    if not value:
        value = os.environ.get("PHANTHY_JP_VERSION")
    aliases = {
        "61": "jp61",
        "jp61": "jp61",
        "511": "jp511",
        "jp511": "jp511",
    }
    normalized = aliases.get(str(value or "").strip().lower())
    if normalized:
        return normalized

    try:
        import tensorrt as trt

        major = int(str(trt.__version__).split(".", 1)[0])
    except Exception:
        major = 0
    if major >= 10:
        return "jp61"
    if major == 8:
        return "jp511"
    raise RuntimeError(
        "Cannot select obstacle engine bundle; set PHANTHY_JP_VERSION to "
        "61 or 511"
    )


def _file_matches(path: str, expected_size: int, expected_sha: str) -> bool:
    try:
        if os.path.getsize(path) != expected_size:
            return False
        digest = hashlib.sha256()
        with open(path, "rb") as model_file:
            for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest() == expected_sha
    except OSError:
        return False


def _download_with_retry(url: str, destination: str, name: str) -> None:
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            with urlopen(url, timeout=60) as response:
                with open(destination, "wb") as output:
                    shutil.copyfileobj(response, output, 1024 * 1024)
            return
        except Exception as error:
            last_error = error
            if attempt < 3:
                log.warning(
                    "[model_downloader] %s: download attempt %d failed; "
                    "retrying",
                    name,
                    attempt,
                )
                time.sleep(attempt)
    raise RuntimeError(
        f"[model_downloader] {name}: download failed after 3 attempts"
    ) from last_error


def ensure_obstacle_models(
    model_dir: str,
    bundle: str | None = None,
) -> dict[str, str]:
    """Download the matching obstacle TensorRT engines when absent or invalid."""
    bundle_name = _obstacle_bundle_name(bundle)
    manifest = OBSTACLE_ENGINE_BUNDLES[bundle_name]
    os.makedirs(model_dir, exist_ok=True)
    paths: dict[str, str] = {}

    for filename, (expected_size, expected_sha) in manifest.items():
        destination = os.path.join(model_dir, filename)
        paths[filename] = destination
        if _file_matches(destination, expected_size, expected_sha):
            log.info(
                "[model_downloader] obstacle/%s: already exists at %s",
                bundle_name,
                destination,
            )
            continue

        # Platform instances share /models. Serialize each missing file so a
        # cold multi-instance launch downloads one copy instead of one per process.
        with open(f"{destination}.lock", "a+b") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            if _file_matches(destination, expected_size, expected_sha):
                continue
            url = (
                f"{OBSTACLE_MODEL_BASE}/{OBSTACLE_MODEL_REVISION}/"
                f"{bundle_name}/{filename}"
            )
            fd, partial_path = tempfile.mkstemp(
                prefix=f".{filename}.",
                suffix=".partial",
                dir=model_dir,
            )
            os.close(fd)
            try:
                log.info(
                    "[model_downloader] obstacle/%s: downloading %s",
                    bundle_name,
                    filename,
                )
                _download_with_retry(
                    url,
                    partial_path,
                    f"obstacle/{filename}",
                )
                if not _file_matches(partial_path, expected_size, expected_sha):
                    raise RuntimeError(
                        f"[model_downloader] obstacle/{filename}: size or "
                        "SHA256 verification failed"
                    )
                os.replace(partial_path, destination)
            finally:
                if os.path.exists(partial_path):
                    os.unlink(partial_path)

    return paths


def ensure_model(name: str, model_dir: str) -> None:
    """Ensure model files exist in model_dir. Download from COS if missing."""
    info = MODELS.get(name)
    if not info:
        raise ValueError(f"Unknown model name: {name}")

    check_path = os.path.join(model_dir, info["check_file"])
    if os.path.exists(check_path):
        log.info(f"[model_downloader] {name}: already exists at {model_dir}")
        return

    url = info["url"]
    os.makedirs(model_dir, exist_ok=True)
    log.info(f"[model_downloader] {name}: downloading from {url} ...")

    if info.get("single_file"):
        # Direct file download (not an archive)
        dest = os.path.join(model_dir, info["check_file"])
        urlretrieve(url, dest, reporthook=_progress_hook(name))
        log.info(f"[model_downloader] {name}: done.")
        return

    # Determine suffix from URL
    if url.endswith(".zip"):
        suffix = ".zip"
    else:
        suffix = ".tar.bz2"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name

    try:
        urlretrieve(url, tmp_path, reporthook=_progress_hook(name))
        log.info(f"[model_downloader] {name}: extracting to {model_dir} ...")

        if suffix == ".zip":
            _extract_zip(tmp_path, model_dir)
        else:
            _extract_tar(tmp_path, model_dir)

        log.info(f"[model_downloader] {name}: done.")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    # Verify
    if not os.path.exists(check_path):
        raise RuntimeError(
            f"[model_downloader] {name}: download completed but {info['check_file']} "
            f"not found in {model_dir}"
        )


def _extract_zip(zip_path: str, model_dir: str) -> None:
    """Extract zip, stripping common top-level directory prefix."""
    with zipfile.ZipFile(zip_path, 'r') as zf:
        # Filter out __MACOSX and directory entries
        names = [n for n in zf.namelist()
                 if not n.endswith('/') and not n.startswith('__MACOSX')]
        if not names:
            raise RuntimeError(f"Empty archive: {zip_path}")

        prefix = _common_prefix_from_names(names)
        for name in names:
            stripped = name[len(prefix):] if prefix else name
            if not stripped:
                continue
            dest = os.path.join(model_dir, stripped)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with zf.open(name) as src, open(dest, 'wb') as dst:
                dst.write(src.read())


def _extract_tar(tar_path: str, model_dir: str) -> None:
    """Extract tar.bz2, stripping common top-level directory prefix."""
    with tarfile.open(tar_path, "r:bz2") as tf:
        members = tf.getmembers()
        if not members:
            raise RuntimeError(f"Empty archive: {tar_path}")

        names = [m.name for m in members if not m.isdir()]
        prefix = _common_prefix_from_names(names)
        for m in members:
            if m.isdir():
                continue
            if prefix:
                m.name = m.name[len(prefix):]
            if not m.name:
                continue
            m.name = m.name.lstrip("/")
            tf.extract(m, model_dir)


def _common_prefix_from_names(names: list[str]) -> str:
    """Find common top-level directory prefix from file name list."""
    dirs_with_slash = [n.split("/", 1) for n in names if "/" in n]
    if not dirs_with_slash:
        return ""
    first_parts = set(parts[0] for parts in dirs_with_slash)
    if len(first_parts) == 1:
        return first_parts.pop() + "/"
    return ""
