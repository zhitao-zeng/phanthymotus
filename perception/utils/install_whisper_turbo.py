#!/usr/bin/env python3
"""Install the pinned Whisper Turbo Q5 release (worker, libs, GGML model).

The release is a ModelScope model repo containing a release.json manifest with
per-file sizes and sha256. Only the files for the requested JetPack line and the
Q5 model are downloaded; runtime symlinks are created from the versioned SO
names because Git releases do not carry reliable symlinks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

DEFAULT_BASE = ("https://www.modelscope.cn/models/"
                "Flame4pd/asr-whisper-large-v3-turbo-q5-v1/resolve")


def fetch(url: str, destination: Path, size: int | None = None,
          sha256: str | None = None, retries: int = 4) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    for attempt in range(1, retries + 1):
        try:
            digest = hashlib.sha256()
            written = 0
            request = urllib.request.Request(url, headers={"User-Agent": "phanthymotus-asr-installer"})
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
                while True:
                    block = response.read(1 << 20)
                    if not block:
                        break
                    output.write(block)
                    digest.update(block)
                    written += len(block)
                    if size and written % (64 << 20) < (1 << 20):
                        print(f"[whisper-turbo] {destination.name}: {written}/{size}", flush=True)
            if size is not None and written != size:
                raise RuntimeError(f"size mismatch for {url}: {written} != {size}")
            if sha256 is not None and digest.hexdigest() != sha256:
                raise RuntimeError(f"sha256 mismatch for {url}")
            temporary.replace(destination)
            return
        except Exception as error:
            temporary.unlink(missing_ok=True)
            if attempt == retries:
                raise
            print(f"[whisper-turbo] retry {attempt}/{retries} {url}: {error}", flush=True)
            time.sleep(2 * attempt)


def make_symlinks(lib_dir: Path) -> None:
    for real in sorted(lib_dir.glob("*.so.*.*")):
        name = real.name
        major = name.split(".so.", 1)[1].split(".", 1)[0]
        for link in (f"{name.split('.so.', 1)[0]}.so.{major}", f"{name.split('.so.', 1)[0]}.so"):
            path = lib_dir / link
            path.unlink(missing_ok=True)
            path.symlink_to(real.name)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=DEFAULT_BASE)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--jp", required=True, choices=["61", "511"])
    parser.add_argument("--destination", default="/opt/whisper-turbo")
    parser.add_argument("--model", default="/opt/asr-models/turbo/model.bin")
    args = parser.parse_args()

    base = args.base_url.rstrip("/") + "/" + urllib.parse.quote(args.revision)
    print(f"[whisper-turbo] release base {base}", flush=True)
    manifest_path = Path("/tmp/whisper-turbo-release.json")
    fetch(base + "/release.json", manifest_path)
    manifest = json.loads(manifest_path.read_text())
    files = {item["path"]: item for item in manifest["files"]}

    destination = Path(args.destination)
    wanted = [path for path in files if path.startswith(f"jp{args.jp}/")]
    wanted.append("model/ggml-large-v3-turbo-q5_0.bin")
    for relative in wanted:
        item = files.get(relative)
        if item is None:
            raise RuntimeError(f"release has no {relative}")
        if relative.startswith(f"jp{args.jp}/"):
            target = destination / relative.split("/", 1)[1]
        else:
            target = Path(args.model)
        print(f"[whisper-turbo] {relative} -> {target}", flush=True)
        fetch(base + "/" + urllib.parse.quote(relative), target,
              item["size"], item["sha256"])
        if relative.endswith("whisper_turbo_worker"):
            target.chmod(0o755)
    make_symlinks(destination / "lib")
    print("[whisper-turbo] installed", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
