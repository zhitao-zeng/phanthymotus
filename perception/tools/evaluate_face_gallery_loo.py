#!/usr/bin/env python3
"""Evaluate a mounted identity gallery with leakage-aware leave-one-out matching."""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

PERCEPTION_ROOT = Path(__file__).resolve().parents[1]
if str(PERCEPTION_ROOT) not in sys.path:
    sys.path.insert(0, str(PERCEPTION_ROOT))

from plugins.face_id.engine import _apply_runtime_profile, build_face_engine  # noqa: E402
from plugins.face_id.alignment import align_face  # noqa: E402
from plugins.face_id.gallery import select_primary_face  # noqa: E402
from plugins.face_id.loo import evaluate_gallery_leave_one_out  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--gallery", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--centroid-weights",
        default="0,0.25,0.5,0.6,0.75,1",
        help="comma-separated grid",
    )
    parser.add_argument("--latency-runs", type=int, default=20)
    parser.add_argument("--all-face-queries", action="store_true")
    args = parser.parse_args()
    weights = [float(item) for item in args.centroid_weights.split(",")]
    if not weights or any(item < 0 or item > 1 for item in weights):
        parser.error("centroid weights must be between 0 and 1")

    loaded = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    cfg = _apply_runtime_profile(dict(loaded.get("plugins", {}).get("face", loaded)))
    cfg["face_db_dir"] = str(args.gallery.resolve())
    quality = dict(cfg.get("gallery_quality") or {})
    build_started = time.perf_counter()
    engine = build_face_engine(cfg)
    build_ms = (time.perf_counter() - build_started) * 1000.0
    if args.latency_runs < 1:
        parser.error("--latency-runs must be at least 1")
    sample_path = next(
        path
        for path in sorted(args.gallery.rglob("*"))
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )
    sample = cv2.imread(str(sample_path), cv2.IMREAD_COLOR)
    detection = select_primary_face(engine.detector.detect(sample), sample.shape)
    if detection is None:
        raise RuntimeError(f"no face detected in latency sample: {sample_path}")
    aligned = align_face(sample, detection.landmarks)
    engine.recognizer.embed(aligned, flip_tta=False)
    embedding_latencies = []
    for _ in range(args.latency_runs):
        started = time.perf_counter()
        engine.recognizer.embed(aligned, flip_tta=False)
        embedding_latencies.append((time.perf_counter() - started) * 1000.0)
    candidate_maps = {False: None, True: None}
    candidate_build_ms = 0.0
    if args.all_face_queries:
        candidate_started = time.perf_counter()
        primary_candidates = {}
        tta_candidates = {}
        for template in engine.gallery.templates:
            if template.source_paths is None or len(template.source_paths) != len(
                template.exemplars
            ):
                raise RuntimeError(
                    f"source paths unavailable for identity {template.person_id}"
                )
            for exemplar_index, source_path in enumerate(template.source_paths):
                query_image = cv2.imread(source_path, cv2.IMREAD_COLOR)
                detections = engine.detector.detect(query_image)
                if not detections:
                    raise RuntimeError(f"no query faces detected: {source_path}")
                primary_rows = []
                tta_rows = []
                for query_detection in detections:
                    query_aligned = align_face(query_image, query_detection.landmarks)
                    primary = engine.recognizer.embed(query_aligned, flip_tta=False)
                    flipped = engine.recognizer.embed(
                        cv2.flip(query_aligned, 1), flip_tta=False
                    )
                    primary_rows.append(primary)
                    tta_rows.append(primary + flipped)
                key = (template.person_id, exemplar_index)
                primary_candidates[key] = np.vstack(primary_rows)
                tta_candidates[key] = np.vstack(tta_rows)
        candidate_maps = {False: primary_candidates, True: tta_candidates}
        candidate_build_ms = (time.perf_counter() - candidate_started) * 1000.0
    runs = []
    try:
        for query_flip_tta in (False, True):
            for centroid_weight in weights:
                started = time.perf_counter()
                result = evaluate_gallery_leave_one_out(
                    engine.gallery,
                    centroid_weight=centroid_weight,
                    query_flip_tta=query_flip_tta,
                    candidate_queries=candidate_maps[query_flip_tta],
                    max_subcenters=int(quality.get("max_subcenters", 3)),
                    subcenter_min_images=int(quality.get("subcenter_min_images", 4)),
                )
                result["matching_ms"] = round(
                    (time.perf_counter() - started) * 1000.0, 3
                )
                runs.append(result)
    finally:
        engine.close()

    summary = {
        "config": str(args.config.resolve()),
        "gallery": str(args.gallery.resolve()),
        "recognizer": cfg.get("recognizer"),
        "backend": cfg.get("backend"),
        "detector_backend": cfg.get("detector_backend", cfg.get("backend")),
        "recognizer_backend": cfg.get("recognizer_backend", cfg.get("backend")),
        "engine_build_ms": round(build_ms, 3),
        "all_face_queries": bool(args.all_face_queries),
        "candidate_build_ms": round(candidate_build_ms, 3),
        "embedding_latency_p50_ms": round(
            float(np.percentile(embedding_latencies, 50)), 3
        ),
        "embedding_latency_p95_ms": round(
            float(np.percentile(embedding_latencies, 95)), 3
        ),
        "max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "runs"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
