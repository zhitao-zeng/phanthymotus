"""Install the pinned DepthART TensorRT runtime during image construction."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile
import time
from urllib.request import ProxyHandler, build_opener


BASE = (
    "https://modelscope.cn/models/Flame4pd/depthart-metric-s-jp61-trt/resolve/"
    "bf3cb0416f78a8f2d493a832b1efbf8ee091c9cd/jp61/"
)
DEST = Path("/opt/vision-depth")
FILES = {
    "depthart-metric-s-fp16.engine": (
        "depthart-metric-s-fp16.engine", 50936900,
        "2535f34517beeb685691952af75ab6285beaf6b7f7a0ab175ab1ecbb7aa4a547",
    ),
    "libdepthart_selective_scan_trt.so": (
        "libdepthart_selective_scan_trt.so", 1259096,
        "6d76307585ffe615db8e620f1826550f086197e9e834236e03565043eb413c35",
    ),
}


def install() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    opener = build_opener(ProxyHandler({}))
    for target_name, (source_name, size, digest) in FILES.items():
        target = DEST / target_name
        for attempt in range(3):
            with tempfile.NamedTemporaryFile(dir=DEST, suffix=".part", delete=False) as temporary:
                temporary_path = Path(temporary.name)
            try:
                hasher = hashlib.sha256()
                count = 0
                with opener.open(BASE + source_name, timeout=40) as response:
                    with temporary_path.open("wb") as output:
                        while chunk := response.read(1024 * 1024):
                            output.write(chunk)
                            hasher.update(chunk)
                            count += len(chunk)
                if count != size or hasher.hexdigest() != digest:
                    raise ValueError(f"Invalid DepthART runtime artifact: {target_name}")
                os.replace(temporary_path, target)
                break
            except Exception:
                temporary_path.unlink(missing_ok=True)
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)


if __name__ == "__main__":
    install()
