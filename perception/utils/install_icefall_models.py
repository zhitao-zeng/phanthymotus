"""Install a fixed model release into an image-owned directory."""

import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.request


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def install(base_url, manifest_sha256, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    # Distribution is reached directly, independently of build proxy variables.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(base_url.rstrip("/") + "/release.json", timeout=60) as response:
        manifest_bytes = response.read()
    if hashlib.sha256(manifest_bytes).hexdigest() != manifest_sha256:
        raise RuntimeError("Model release manifest checksum mismatch")
    manifest = json.loads(manifest_bytes)
    for item in manifest["files"]:
        path = destination / item["path"]
        if path.is_file() and path.stat().st_size == item["size"] and file_sha256(path) == item["sha256"]:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name + ".part")
        for attempt in range(3):
            try:
                with opener.open(base_url.rstrip("/") + "/" + item["path"], timeout=60) as response:
                    with partial.open("wb") as output:
                        for block in iter(lambda: response.read(4 * 1024 * 1024), b""):
                            output.write(block)
                if partial.stat().st_size != item["size"] or file_sha256(partial) != item["sha256"]:
                    raise RuntimeError("Model artifact verification failed: " + item["path"])
                partial.replace(path)
                print("[asr-models] installed " + item["path"], flush=True)
                break
            except Exception:
                partial.unlink(missing_ok=True)
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)
    (destination / "release.json").write_bytes(manifest_bytes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--destination", required=True)
    args = parser.parse_args()
    install(args.base_url, args.manifest_sha256, args.destination)
