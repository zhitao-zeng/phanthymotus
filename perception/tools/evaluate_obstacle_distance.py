#!/usr/bin/env python3
"""Offline obstacle-distance evaluation without model or dataset downloads."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Mapping

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from perception.plugins.obstacle_distance_core import backend_loader
from perception.plugins.obstacle_distance_core.estimator import (
    ObstacleDistanceEstimator,
)
from perception.plugins.obstacle_distance_core.metrics import (
    evaluate_grouped_predictions,
    evaluate_predictions,
    scan_thresholds,
)


_REQUIRED_MANIFEST_FIELDS = {"image_path", "scene", "gt_distance_m"}
_SCENES = {"indoor", "vehicle"}

_DEFAULT_CONFIG = {
    "mode": "model",
    "decision_threshold_m": 1.0,
    "fallback_distance_m": 3.0,
    "soft_timeout_s": 2.5,
    "indoor": {
        "roi": [0, 300, 213, 426],
        "min_depth_m": 0.3,
        "max_depth_m": 10.0,
        "percentile": 1.0,
        "min_valid_pixels": 64,
    },
    "vehicle": {
        "allowed_classes": [
            "person",
            "car",
            "truck",
            "bus",
            "motorcycle",
            "bicycle",
        ],
        "min_confidence": 0.25,
        "percentile": 1.0,
        "min_depth_m": 0.3,
        "max_depth_m": 80.0,
        "allow_approximate_geometry": False,
        "camera_to_bumper_offset_m": 1.0,
        "calibration": {},
    },
}


class _CliError(Exception):
    pass


def _recursive_merge(
    base: Mapping[str, object],
    override: Mapping[str, object],
    *,
    _path: tuple[str, ...] = (),
) -> dict[str, object]:
    merged = deepcopy(dict(base))
    for key, value in override.items():
        current = merged.get(key)
        path = (*_path, key)
        if (
            isinstance(value, Mapping)
            and not value
            and path == ("vehicle", "calibration")
        ):
            merged[key] = {}
        elif isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = _recursive_merge(
                current,
                value,
                _path=path,
            )
        elif isinstance(value, Mapping) and not value:
            merged[key] = {}
        else:
            merged[key] = deepcopy(value)
    return merged


def _load_config(path: Path) -> dict[str, object]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        raise _CliError("config could not be read") from None

    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        if path.suffix.lower() not in {".yaml", ".yml"}:
            raise _CliError("config is not valid JSON") from None
        try:
            import yaml
        except ImportError:
            raise _CliError(
                "YAML config requires optional PyYAML; use JSON instead"
            ) from None
        try:
            loaded = yaml.safe_load(text)
        except Exception:
            raise _CliError("config is not valid YAML") from None
    if not isinstance(loaded, Mapping):
        raise _CliError("config root must be an object")
    return _recursive_merge(_DEFAULT_CONFIG, loaded)


def _finite_nonnegative_text(value: object) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError):
        raise _CliError(
            "manifest gt_distance_m must be a finite nonnegative number"
        ) from None
    if not math.isfinite(converted) or converted < 0:
        raise _CliError(
            "manifest gt_distance_m must be a finite nonnegative number"
        )
    return converted


def _load_manifest(path: Path) -> list[dict[str, object]]:
    try:
        handle = path.open("r", encoding="utf-8", newline="")
    except (OSError, UnicodeError):
        raise _CliError("manifest could not be read") from None

    with handle:
        try:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames
            if fieldnames is None or not _REQUIRED_MANIFEST_FIELDS.issubset(
                fieldnames
            ):
                raise _CliError(
                    "manifest is missing required fields: "
                    "image_path, scene, gt_distance_m"
                )
            rows: list[dict[str, object]] = []
            for row in reader:
                image_path = row.get("image_path")
                scene = row.get("scene")
                if not isinstance(image_path, str) or not image_path:
                    raise _CliError(
                        "manifest image_path must be nonempty"
                    )
                if scene not in _SCENES:
                    raise _CliError(
                        "manifest scene must be indoor or vehicle"
                    )
                rows.append(
                    {
                        "image_path": image_path,
                        "scene": scene,
                        "gt_distance_m": _finite_nonnegative_text(
                            row.get("gt_distance_m")
                        ),
                    }
                )
        except csv.Error:
            raise _CliError("manifest CSV is invalid") from None
    if not rows:
        raise _CliError("manifest must contain at least one sample")
    return rows


def _metrics_dicts(metrics_by_group):
    return {
        label: asdict(metrics)
        for label, metrics in metrics_by_group.items()
    }


def _build_report(
    rows: list[dict[str, object]],
    *,
    manifest_path: Path,
    estimator: ObstacleDistanceEstimator,
    threshold_m: float,
) -> dict[str, object]:
    ground_truth: list[float] = []
    predictions: list[float] = []
    failures: list[bool] = []
    scenes: list[str] = []
    statuses: list[str] = []
    error_codes: list[str | None] = []
    prediction_rows: list[dict[str, object]] = []

    for row in rows:
        relative_path = str(row["image_path"])
        image_path = Path(relative_path)
        if not image_path.is_absolute():
            image_path = manifest_path.parent / image_path
        try:
            image_bytes = image_path.read_bytes()
        except OSError:
            image_bytes = b""

        result = estimator.estimate(
            image_bytes,
            scene_hint=str(row["scene"]),
            source_name=str(image_path),
        )
        truth = float(row["gt_distance_m"])
        failed = result.fallback is True
        ground_truth.append(truth)
        predictions.append(result.distance_m)
        failures.append(failed)
        scenes.append(str(row["scene"]))
        statuses.append(result.status)
        error_codes.append(result.error_code)
        prediction_rows.append(
            {
                "image_path": relative_path,
                "scene": str(row["scene"]),
                "gt_distance_m": truth,
                **asdict(result),
            }
        )

    overall = evaluate_predictions(
        ground_truth,
        predictions,
        threshold_m,
        failed=failures,
    )
    try:
        scan = scan_thresholds(
            ground_truth,
            predictions,
            failed=failures,
        )
    except ValueError:
        best_threshold = None
    else:
        best_threshold = {
            "threshold_m": scan.threshold_m,
            "metrics": asdict(scan.metrics),
        }
    return {
        "overall": asdict(overall),
        "by_scene": _metrics_dicts(
            evaluate_grouped_predictions(
                ground_truth,
                predictions,
                scenes,
                threshold_m,
                failed=failures,
            )
        ),
        "by_status": _metrics_dicts(
            evaluate_grouped_predictions(
                ground_truth,
                predictions,
                statuses,
                threshold_m,
                failed=failures,
            )
        ),
        "by_error_code": _metrics_dicts(
            evaluate_grouped_predictions(
                ground_truth,
                predictions,
                error_codes,
                threshold_m,
                failed=failures,
            )
        ),
        "best_threshold": best_threshold,
        "predictions": prediction_rows,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate nearest-obstacle distance predictions"
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument(
        "--mode",
        choices=("diagnostic_constant", "model"),
    )
    parser.add_argument("--constant-distance-m", type=float)
    parser.add_argument("--backend-factory")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--threshold-m", type=float)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = (
            _load_config(args.config)
            if args.config is not None
            else deepcopy(_DEFAULT_CONFIG)
        )
        if args.mode is not None:
            config["mode"] = args.mode
        if args.constant_distance_m is not None:
            config["constant_distance_m"] = args.constant_distance_m
        if args.backend_factory is not None:
            config["backend_factory"] = args.backend_factory
        if args.threshold_m is not None:
            config["decision_threshold_m"] = args.threshold_m

        mode = config.get("mode")
        if mode == "diagnostic_constant":
            if "constant_distance_m" not in config:
                raise _CliError(
                    "diagnostic_constant mode requires an explicit "
                    "constant distance"
                )
        elif mode == "model":
            if not isinstance(config.get("backend_factory"), str) or not config[
                "backend_factory"
            ]:
                raise _CliError(
                    "backend factory is required in model mode"
                )
        else:
            raise _CliError(
                "mode must be diagnostic_constant or model"
            )

        rows = _load_manifest(args.manifest)
        try:
            depth_backend, segmentation_backend = (
                backend_loader.create_model_backends(config)
            )
            estimator = ObstacleDistanceEstimator(
                depth_backend,
                segmentation_backend,
                config,
            )
        except (TypeError, ValueError, RuntimeError):
            raise _CliError("backend or estimator initialization failed") from None

        threshold_m = config.get("decision_threshold_m")
        report = _build_report(
            rows,
            manifest_path=args.manifest,
            estimator=estimator,
            threshold_m=threshold_m,
        )
        try:
            rendered = json.dumps(
                report,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
        except (TypeError, ValueError, OverflowError):
            raise _CliError("evaluation report contains invalid values") from None
        if args.output is None:
            sys.stdout.write(rendered + "\n")
        else:
            try:
                args.output.write_text(rendered + "\n", encoding="utf-8")
            except (OSError, UnicodeError):
                raise _CliError("output could not be written") from None
        return 0
    except _CliError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except Exception:
        print("error: obstacle-distance evaluation failed", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
