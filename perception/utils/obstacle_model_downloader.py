#!/usr/bin/env python3
"""Download and verify the obstacle model artifacts declared in a manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def load_manifest(path: str | Path) -> list[dict[str, object]]:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported obstacle artifact manifest schema")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("artifact manifest must contain a nonempty list")

    validated = []
    destinations = set()
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise ValueError("each artifact must be an object")
        name = artifact.get("name")
        url = artifact.get("url")
        size_bytes = artifact.get("size_bytes")
        sha256 = artifact.get("sha256")
        destination = artifact.get("destination")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("artifact name must be a nonempty string")
        if not isinstance(url, str) or not url.startswith(("http://", "https://", "file://")):
            raise ValueError(f"artifact {name} has an invalid URL")
        if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes <= 0:
            raise ValueError(f"artifact {name} has an invalid size")
        if not isinstance(sha256, str) or not _SHA256_RE.fullmatch(sha256):
            raise ValueError(f"artifact {name} has an invalid sha256")
        if not isinstance(destination, str) or not Path(destination).is_absolute():
            raise ValueError(f"artifact {name} destination must be absolute")
        if destination in destinations:
            raise ValueError("artifact destinations must be unique")
        destinations.add(destination)
        validated.append(dict(artifact))
    return validated


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_is_valid(artifact: dict[str, object]) -> bool:
    destination = Path(str(artifact["destination"]))
    return (
        destination.is_file()
        and destination.stat().st_size == artifact["size_bytes"]
        and sha256_file(destination) == artifact["sha256"]
    )


def download_artifact(
    artifact: dict[str, object],
    *,
    timeout_s: float = 120.0,
    retries: int = 3,
) -> None:
    name = str(artifact["name"])
    destination = Path(str(artifact["destination"]))
    expected_size = int(artifact["size_bytes"])
    expected_sha256 = str(artifact["sha256"])
    if artifact_is_valid(artifact):
        print(f"{name}: already verified at {destination}", flush=True)
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    last_error = None
    for attempt in range(1, retries + 1):
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix=f".{destination.name}.",
                dir=destination.parent,
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                digest = hashlib.sha256()
                downloaded = 0
                with urlopen(str(artifact["url"]), timeout=timeout_s) as response:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        downloaded += len(chunk)
                        if downloaded > expected_size:
                            raise ValueError(
                                f"{name} exceeds expected size {expected_size}"
                            )
                        digest.update(chunk)
                        temporary.write(chunk)
                temporary.flush()
                os.fsync(temporary.fileno())

            if downloaded != expected_size:
                raise ValueError(
                    f"{name} size mismatch: {downloaded} != {expected_size}"
                )
            actual_sha256 = digest.hexdigest()
            if actual_sha256 != expected_sha256:
                raise ValueError(
                    f"{name} sha256 mismatch: {actual_sha256}"
                )
            os.replace(temporary_path, destination)
            temporary_path = None
            print(f"{name}: verified {downloaded} bytes", flush=True)
            return
        except (OSError, TimeoutError, URLError, ValueError) as error:
            last_error = error
            if attempt < retries:
                print(
                    f"{name}: attempt {attempt}/{retries} failed; retrying",
                    flush=True,
                )
                time.sleep(min(attempt, 3))
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
    raise RuntimeError(f"failed to download verified artifact {name}") from last_error


def download_manifest(
    manifest_path: str | Path,
    *,
    timeout_s: float = 120.0,
    retries: int = 3,
    destination_root: str | Path = "/",
) -> None:
    if retries < 1:
        raise ValueError("retries must be positive")
    if timeout_s <= 0:
        raise ValueError("timeout must be positive")
    root = Path(destination_root)
    if not root.is_absolute():
        raise ValueError("destination root must be absolute")
    for artifact in load_manifest(manifest_path):
        if root != Path("/"):
            artifact = dict(artifact)
            destination = Path(str(artifact["destination"]))
            artifact["destination"] = str(root / destination.relative_to("/"))
        download_artifact(
            artifact,
            timeout_s=timeout_s,
            retries=retries,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--destination-root",
        default="/",
        help="prefix absolute manifest destinations (useful for local verification)",
    )
    args = parser.parse_args()
    download_manifest(
        args.manifest,
        timeout_s=args.timeout,
        retries=args.retries,
        destination_root=args.destination_root,
    )


if __name__ == "__main__":
    main()
