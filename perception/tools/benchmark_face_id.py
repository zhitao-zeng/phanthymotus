#!/usr/bin/env python3
"""Run the pure face-identification core over local images as JSON Lines."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import yaml

PERCEPTION_ROOT = Path(__file__).resolve().parents[1]
if str(PERCEPTION_ROOT) not in sys.path:
    sys.path.insert(0, str(PERCEPTION_ROOT))

from plugins.face_id.engine import build_face_engine  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--gallery", type=Path)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--expected-person-id")
    parser.add_argument("images", nargs="+", type=Path)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    with args.config.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    cfg = dict(loaded.get("plugins", {}).get("face", loaded))
    if args.gallery:
        cfg["face_db_dir"] = str(args.gallery.resolve())
    if args.warmup < 0 or args.repeat < 1:
        parser.error("--warmup must be >= 0 and --repeat must be >= 1")
    build_started = time.perf_counter()
    engine = build_face_engine(cfg)
    build_ms = (time.perf_counter() - build_started) * 1000.0
    image_bytes = [(path, path.read_bytes()) for path in args.images]
    latencies: list[float] = []
    successes = 0
    failures = 0
    correct = 0
    try:
        for _ in range(args.warmup):
            engine.infer_face_identity(image_bytes[0][1])
        for _ in range(args.repeat):
            for image_path, encoded in image_bytes:
                started = time.perf_counter()
                try:
                    payload = engine.infer_face_identity(encoded)
                except Exception as error:  # noqa: BLE001 - benchmark records failures
                    failures += 1
                    if not args.summary_only:
                        print(
                            json.dumps(
                                {"image_file": str(image_path), "error": str(error)},
                                ensure_ascii=False,
                            ),
                            flush=True,
                        )
                    continue
                latency_ms = (time.perf_counter() - started) * 1000.0
                latencies.append(latency_ms)
                successes += 1
                identity = payload.get("identity") or {}
                if (
                    args.expected_person_id is not None
                    and identity.get("person_id") == args.expected_person_id
                ):
                    correct += 1
                if not args.summary_only:
                    print(
                        json.dumps(
                            {
                                "image_file": str(image_path),
                                **payload,
                                "latency_ms": round(latency_ms, 3),
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
    finally:
        engine.close()
    summary = {
        "type": "summary",
        "engine_build_ms": round(build_ms, 3),
        "warmup": args.warmup,
        "attempts": args.repeat * len(image_bytes),
        "successes": successes,
        "failures": failures,
        "latency_mean_ms": round(float(np.mean(latencies)), 3) if latencies else None,
        "latency_p50_ms": round(float(np.percentile(latencies, 50)), 3) if latencies else None,
        "latency_p95_ms": round(float(np.percentile(latencies, 95)), 3) if latencies else None,
        "latency_max_ms": round(float(np.max(latencies)), 3) if latencies else None,
    }
    if args.expected_person_id is not None:
        summary["expected_person_id"] = args.expected_person_id
        summary["correct"] = correct
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    if failures:
        return 1
    if args.expected_person_id is not None and correct != successes:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
