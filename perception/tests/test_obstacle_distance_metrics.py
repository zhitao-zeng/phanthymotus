import contextlib
import io
import itertools
import json
import math
import random
import tempfile
import unittest
from dataclasses import FrozenInstanceError, fields
from pathlib import Path
from types import MappingProxyType
from unittest import mock

import numpy as np

from perception.plugins.obstacle_distance_core import metrics as metrics_module
from perception.plugins.obstacle_distance_core.contracts import DepthPrediction
from perception.plugins.obstacle_distance_core.metrics import (
    EvaluationMetrics,
    ThresholdScanResult,
    evaluate_grouped_predictions,
    evaluate_predictions,
    scan_thresholds,
)
from perception.tools import evaluate_obstacle_distance


class EvaluationMetricsTest(unittest.TestCase):
    def test_contract_is_frozen_and_fields_are_exact(self):
        self.assertEqual(
            [field.name for field in fields(EvaluationMetrics)],
            [
                "samples",
                "valid_predictions",
                "failures",
                "failure_rate",
                "tp",
                "fp",
                "fn",
                "precision",
                "recall",
                "f1",
                "rmse",
                "positive_rate",
                "roc_auc",
            ],
        )
        result = evaluate_predictions([0.5], [0.5], 1.0)
        with self.assertRaises(FrozenInstanceError):
            result.samples = 2

    def test_public_evaluation_signature_accepts_gt_and_pred_keywords(self):
        result = evaluate_predictions(
            gt=[0.5],
            pred=[0.5],
            threshold_m=1.0,
        )

        self.assertEqual(result.tp, 1)

    def test_confusion_matrix_rmse_and_positive_rate_are_hand_calculated(self):
        result = evaluate_predictions(
            [0.5, 0.8, 1.2, 2.0],
            [0.4, 1.1, 0.9, 2.2],
            threshold_m=1.0,
        )

        self.assertEqual(result.samples, 4)
        self.assertEqual(result.valid_predictions, 4)
        self.assertEqual(result.failures, 0)
        self.assertEqual((result.tp, result.fp, result.fn), (1, 1, 1))
        self.assertAlmostEqual(result.precision, 0.5)
        self.assertAlmostEqual(result.recall, 0.5)
        self.assertAlmostEqual(result.f1, 0.5)
        self.assertAlmostEqual(
            result.rmse,
            math.sqrt((0.1**2 + 0.3**2 + 0.3**2 + 0.2**2) / 4),
        )
        self.assertAlmostEqual(result.positive_rate, 0.5)

    def test_roc_auc_ranks_lower_predictions_as_more_positive(self):
        perfect = evaluate_predictions(
            [0.5, 0.8, 1.2, 2.0],
            [0.4, 0.7, 1.1, 2.2],
            threshold_m=1.0,
        )
        self.assertAlmostEqual(perfect.roc_auc, 1.0)

        reversed_ = evaluate_predictions(
            [0.5, 0.8, 1.2, 2.0],
            [2.2, 1.1, 0.7, 0.4],
            threshold_m=1.0,
        )
        self.assertAlmostEqual(reversed_.roc_auc, 0.0)

        single_class = evaluate_predictions(
            [0.5, 0.8, 1.2, 2.0],
            [0.4, 0.7, 1.1, 2.2],
            threshold_m=0.1,
        )
        self.assertIsNone(single_class.roc_auc)

    def test_roc_auc_ties_count_half(self):
        result = evaluate_predictions(
            [0.5, 1.2, 1.2],
            [1.0, 1.0, 1.0],
            threshold_m=1.0,
        )
        self.assertAlmostEqual(result.roc_auc, 0.5)

    def test_threshold_rule_is_strictly_less_than(self):
        result = evaluate_predictions([1.0, 0.5], [0.5, 1.0], 1.0)

        self.assertEqual((result.tp, result.fp, result.fn), (0, 1, 1))

    def test_zero_division_is_stable(self):
        all_negative = evaluate_predictions([2.0, 3.0], [2.0, 3.0], 1.0)
        self.assertEqual(
            (all_negative.precision, all_negative.recall, all_negative.f1),
            (0.0, 0.0, 0.0),
        )

        no_predicted_positive = evaluate_predictions(
            [0.5, 2.0],
            [2.0, 2.0],
            1.0,
        )
        self.assertEqual(
            (
                no_predicted_positive.precision,
                no_predicted_positive.recall,
                no_predicted_positive.f1,
            ),
            (0.0, 0.0, 0.0),
        )

    def test_invalid_or_explicitly_failed_predictions_are_failures(self):
        result = evaluate_predictions(
            [0.2, 0.4, 0.6, 0.8, 2.0, 3.0],
            [None, math.nan, math.inf, -1.0, 2.5, 3.5],
            1.0,
            failed=[False, False, False, False, True, False],
        )

        self.assertEqual(result.samples, 6)
        self.assertEqual(result.valid_predictions, 1)
        self.assertEqual(result.failures, 5)
        self.assertAlmostEqual(result.failure_rate, 5 / 6)
        self.assertEqual((result.tp, result.fp, result.fn), (0, 0, 0))
        self.assertAlmostEqual(result.rmse, 0.5)
        self.assertAlmostEqual(result.positive_rate, 4 / 6)

    def test_no_valid_predictions_uses_none_rmse(self):
        result = evaluate_predictions([0.5], [None], 1.0)

        self.assertIsNone(result.rmse)
        self.assertEqual(result.failure_rate, 1.0)

    def test_empty_mismatched_and_invalid_inputs_raise_safe_errors(self):
        secret = "private-distance-value-" * 300
        cases = (
            ([], [], 1.0, None),
            ([0.5], [], 1.0, None),
            ([0.5], [0.5], 1.0, []),
            ([True], [0.5], 1.0, None),
            ([-0.1], [0.5], 1.0, None),
            ([math.inf], [0.5], 1.0, None),
            ([secret], [0.5], 1.0, None),
            ([0.5], [0.5], True, None),
            ([0.5], [0.5], 0.0, None),
            ([0.5], [0.5], math.inf, None),
            ([0.5], [0.5], 1.0, [1]),
        )
        for gt, prediction, threshold, failed in cases:
            with self.subTest(gt=gt[:1], threshold=threshold, failed=failed):
                with self.assertRaises(ValueError) as raised:
                    evaluate_predictions(
                        gt,
                        prediction,
                        threshold,
                        failed=failed,
                    )
                self.assertNotIn(secret, str(raised.exception))
                self.assertLess(len(str(raised.exception)), 200)

    def test_rmse_remains_finite_for_huge_finite_values(self):
        result = evaluate_predictions(
            [0.0, 0.0],
            [float.fromhex("0x1.fffffffffffffp+1023")] * 2,
            1.0,
        )

        self.assertTrue(math.isfinite(result.rmse))

    def test_inputs_reject_mappings_and_iterables_without_reliable_length(self):
        secret = "private-iterable-value-" * 100

        class UnreliableLength:
            def __len__(self):
                raise RuntimeError(secret)

            def __iter__(self):
                return iter([0.5])

        invalid_inputs = (
            (value for value in [0.5]),
            itertools.repeat(0.5, 1),
            {0.5: secret},
            UnreliableLength(),
        )
        for invalid in invalid_inputs:
            with self.subTest(input_type=type(invalid).__name__):
                with self.assertRaises(ValueError) as raised:
                    evaluate_predictions(invalid, [0.5], 1.0)
                self.assertNotIn(secret, str(raised.exception))
                self.assertLess(len(str(raised.exception)), 200)

    def test_finite_sized_sequence_inputs_remain_supported(self):
        cases = (
            ([0.5], [0.5]),
            ((0.5,), (0.5,)),
            (range(1), range(1)),
            (np.array([0.5]), np.array([0.5])),
        )
        for ground_truth, predictions in cases:
            with self.subTest(input_type=type(ground_truth).__name__):
                result = evaluate_predictions(
                    ground_truth,
                    predictions,
                    1.0,
                )
                self.assertEqual(result.samples, 1)
                self.assertEqual(result.valid_predictions, 1)


class ThresholdScanTest(unittest.TestCase):
    @staticmethod
    def _brute_scan(
        gt,
        pred,
        failed,
        *,
        ground_truth_threshold_m=1.0,
    ):
        valid_predictions = [
            value
            for value, is_failed in zip(pred, failed, strict=True)
            if not is_failed
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and value >= 0
        ]
        unique = sorted(set(valid_predictions))
        candidates = [
            unique[0]
            if unique[0] > 0
            else math.nextafter(0.0, math.inf)
        ]
        for low, high in zip(unique, unique[1:]):
            midpoint = low + (high - low) / 2.0
            candidates.append(
                midpoint if low < midpoint < high else high
            )
        upper = math.nextafter(unique[-1], math.inf)
        if math.isfinite(upper):
            candidates.append(upper)

        baseline = evaluate_predictions(
            gt,
            pred,
            ground_truth_threshold_m,
            failed=failed,
        )
        results = []
        for threshold in dict.fromkeys(candidates):
            valid_records = [
                (truth, prediction)
                for truth, prediction, is_failed in zip(
                    gt,
                    pred,
                    failed,
                    strict=True,
                )
                if not is_failed
                and isinstance(prediction, (int, float))
                and not isinstance(prediction, bool)
                and math.isfinite(prediction)
                and prediction >= 0
            ]
            tp = sum(
                truth < ground_truth_threshold_m
                and prediction < threshold
                for truth, prediction in valid_records
            )
            fp = sum(
                truth >= ground_truth_threshold_m
                and prediction < threshold
                for truth, prediction in valid_records
            )
            fn = sum(
                truth < ground_truth_threshold_m
                and prediction >= threshold
                for truth, prediction in valid_records
            )
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = (
                2.0 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            )
            results.append(
                ThresholdScanResult(
                    threshold,
                    EvaluationMetrics(
                        samples=baseline.samples,
                        valid_predictions=baseline.valid_predictions,
                        failures=baseline.failures,
                        failure_rate=baseline.failure_rate,
                        tp=tp,
                        fp=fp,
                        fn=fn,
                        precision=precision,
                        recall=recall,
                        f1=f1,
                        rmse=baseline.rmse,
                        positive_rate=sum(
                            truth < ground_truth_threshold_m for truth in gt
                        )
                        / len(gt),
                        roc_auc=baseline.roc_auc,
                    ),
                )
            )
        return max(
            results,
            key=lambda result: (
                result.metrics.f1,
                result.metrics.precision,
                -abs(
                    result.threshold_m - ground_truth_threshold_m
                ),
                -result.threshold_m,
            ),
        )

    def test_contract_is_frozen_and_known_optimum_is_found(self):
        self.assertEqual(
            [field.name for field in fields(ThresholdScanResult)],
            ["threshold_m", "metrics"],
        )
        result = scan_thresholds(
            [0.5, 0.8, 1.2, 2.0],
            [0.4, 0.7, 1.1, 2.2],
        )

        self.assertAlmostEqual(result.threshold_m, 0.9)
        self.assertEqual(result.metrics.f1, 1.0)
        with self.assertRaises(FrozenInstanceError):
            result.threshold_m = 2.0

    def test_scan_keeps_ground_truth_labels_fixed_at_one_meter(self):
        result = scan_thresholds(
            [0.1, 0.5],
            [0.1, 0.1],
        )

        self.assertEqual((result.metrics.tp, result.metrics.fp), (2, 0))
        self.assertEqual(result.metrics.f1, 1.0)
        self.assertEqual(result.metrics.positive_rate, 1.0)

    def test_scan_accepts_an_explicit_fixed_ground_truth_threshold(self):
        result = scan_thresholds(
            [0.1, 0.5],
            [0.1, 0.1],
            ground_truth_threshold_m=0.25,
        )

        self.assertEqual((result.metrics.tp, result.metrics.fp), (1, 1))
        self.assertAlmostEqual(result.metrics.f1, 2 / 3)
        self.assertEqual(result.metrics.positive_rate, 0.5)

        for invalid in (0, -1, math.nan, math.inf, True, "1"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    scan_thresholds(
                        [0.1],
                        [0.1],
                        ground_truth_threshold_m=invalid,
                    )

    def test_ties_prefer_precision_then_distance_to_cutoff_then_smaller(self):
        precision_winner = scan_thresholds(
            [0.1, 2.0, 2.0, 0.1],
            [0.2, 0.4, 0.6, 0.8],
        )
        self.assertAlmostEqual(precision_winner.threshold_m, 0.3)
        self.assertEqual(precision_winner.metrics.precision, 1.0)

        distance_winner = scan_thresholds([3.0], [2.0])
        self.assertEqual(distance_winner.threshold_m, 2.0)

        smaller_winner = scan_thresholds(
            [2.0, 2.0, 2.0],
            [0.0, 1.0, 2.0],
        )
        self.assertEqual(smaller_winner.threshold_m, 0.5)

        custom_anchor_winner = scan_thresholds(
            [2.0, 2.0, 2.0],
            [0.0, 0.25, 0.5],
            ground_truth_threshold_m=0.25,
        )
        self.assertEqual(custom_anchor_winner.threshold_m, 0.125)

    def test_scan_ignores_failures_and_rejects_no_valid_prediction(self):
        result = scan_thresholds(
            [0.5, 2.0],
            [0.25, 100.0],
            failed=[False, True],
        )
        self.assertLess(result.threshold_m, 1.0)

        with self.assertRaises(ValueError):
            scan_thresholds([0.5, 2.0], [None, math.nan])

    def test_candidates_are_finite_for_adjacent_and_huge_values(self):
        maximum = float.fromhex("0x1.fffffffffffffp+1023")
        below = math.nextafter(maximum, 0.0)
        result = scan_thresholds(
            [0.0, maximum],
            [below, maximum],
        )

        self.assertTrue(math.isfinite(result.threshold_m))
        self.assertGreater(result.threshold_m, 0.0)

    def test_scan_matches_brute_force_for_randomized_edge_cases(self):
        maximum = float.fromhex("0x1.fffffffffffffp+1023")
        cases = [
            (
                [0.0, 0.0, 1.0, maximum, maximum],
                [0.0, 0.0, None, math.inf, maximum],
                [False, True, False, False, False],
            )
        ]
        randomizer = random.Random(20260730)
        truth_pool = [0.0, 0.25, 1.0, 2.0, maximum]
        prediction_pool = [
            0.0,
            0.0,
            0.5,
            1.0,
            2.0,
            maximum,
            None,
            math.nan,
        ]
        for _ in range(100):
            size = randomizer.randint(1, 12)
            gt = [randomizer.choice(truth_pool) for _ in range(size)]
            pred = [
                randomizer.choice(prediction_pool) for _ in range(size)
            ]
            failed = [
                randomizer.random() < 0.2 for _ in range(size)
            ]
            if not any(
                not is_failed
                and isinstance(value, (int, float))
                and math.isfinite(value)
                and value >= 0
                for value, is_failed in zip(pred, failed, strict=True)
            ):
                pred[0] = 0.0
                failed[0] = False
            cases.append((gt, pred, failed))

        for gt, pred, failed in cases:
            with self.subTest(size=len(gt)):
                expected = self._brute_scan(gt, pred, failed)
                actual = scan_thresholds(gt, pred, failed=failed)
                self.assertEqual(actual, expected)

    def test_large_scan_does_not_re_evaluate_each_candidate(self):
        size = 2500
        ground_truth = [
            float(index % 17) / 4.0 for index in range(size)
        ]
        predictions = [
            float(index) / size * 4.0 for index in range(size)
        ]

        with mock.patch.object(
            metrics_module,
            "evaluate_predictions",
            side_effect=AssertionError(
                "scan must not re-evaluate every candidate"
            ),
        ) as evaluate:
            result = scan_thresholds(ground_truth, predictions)

        evaluate.assert_not_called()
        self.assertEqual(result.metrics.samples, size)
        self.assertEqual(result.metrics.valid_predictions, size)


class GroupedMetricsTest(unittest.TestCase):
    def test_groups_are_evaluated_independently_and_empty_labels_are_skipped(self):
        result = evaluate_grouped_predictions(
            [0.5, 2.0, 0.25, 3.0],
            [0.4, 2.5, None, 0.5],
            ["indoor", "vehicle", None, ""],
            1.0,
        )

        self.assertEqual(set(result), {"indoor", "vehicle"})
        self.assertEqual(result["indoor"].tp, 1)
        self.assertEqual(result["vehicle"].tp, 0)
        self.assertEqual(result["vehicle"].rmse, 0.5)

    def test_group_length_must_match(self):
        with self.assertRaises(ValueError):
            evaluate_grouped_predictions([0.5], [0.5], [], 1.0)

    def test_skipped_groups_still_validate_boolean_failure_flags(self):
        with self.assertRaises(ValueError):
            evaluate_grouped_predictions(
                [0.5],
                [0.5],
                [None],
                1.0,
                failed=[1],
            )

    def test_empty_groups_still_validate_threshold(self):
        with self.assertRaises(ValueError):
            evaluate_grouped_predictions(
                [0.5],
                [0.5],
                [None],
                math.nan,
            )


class _DepthBackend:
    def __init__(self):
        self.calls = []

    def predict_depth(self, image_bytes, domain, deadline_monotonic):
        self.calls.append((image_bytes, domain, deadline_monotonic))
        return DepthPrediction([[2.0]], 1, 1)


class _SegmentationBackend:
    def predict_instances(self, image_bytes, deadline_monotonic):
        return ()


class EvaluationCliTest(unittest.TestCase):
    def _write_manifest(self, directory, rows, header=None):
        manifest = Path(directory) / "manifest.csv"
        columns = header or ["image_path", "scene", "gt_distance_m"]
        lines = [",".join(columns)]
        lines.extend(",".join(str(row.get(column, "")) for column in columns) for row in rows)
        manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return manifest

    def test_diagnostic_mode_resolves_relative_paths_and_outputs_full_report(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "images").mkdir()
            (root / "images" / "frame.bin").write_bytes(b"rgb")
            manifest = self._write_manifest(
                root,
                [
                    {
                        "image_path": "images/frame.bin",
                        "scene": "indoor",
                        "gt_distance_m": "0.4",
                        "note": "ignored",
                    }
                ],
                header=[
                    "image_path",
                    "scene",
                    "gt_distance_m",
                    "note",
                ],
            )
            output = root / "report.json"

            exit_code = evaluate_obstacle_distance.main(
                [
                    "--manifest",
                    str(manifest),
                    "--mode",
                    "diagnostic_constant",
                    "--constant-distance-m",
                    "0.5",
                    "--output",
                    str(output),
                ]
            )

            self.assertEqual(exit_code, 0)
            raw_report = output.read_text(encoding="utf-8")
            self.assertNotIn("NaN", raw_report)
            self.assertNotIn("Infinity", raw_report)
            report = json.loads(raw_report)
            self.assertEqual(
                set(report),
                {
                    "overall",
                    "by_scene",
                    "by_status",
                    "by_error_code",
                    "best_threshold",
                    "predictions",
                },
            )
            self.assertEqual(report["overall"]["failures"], 0)
            self.assertEqual(report["by_scene"]["indoor"]["samples"], 1)
            self.assertEqual(
                report["by_status"]["diagnostic_constant"]["samples"],
                1,
            )
            self.assertEqual(report["by_error_code"], {})
            prediction = report["predictions"][0]
            self.assertEqual(prediction["image_path"], "images/frame.bin")
            self.assertEqual(prediction["scene"], "indoor")
            self.assertEqual(prediction["gt_distance_m"], 0.4)
            for result_field in (
                "distance_m",
                "near_obstacle",
                "decision_threshold_m",
                "status",
                "error_code",
                "fallback",
                "approximate_geometry",
                "latency_ms",
                "timestamp",
            ):
                self.assertIn(result_field, prediction)

    def test_cli_uses_threshold_m_as_the_fixed_ground_truth_cutoff(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "first.bin").write_bytes(b"rgb")
            (root / "second.bin").write_bytes(b"rgb")
            manifest = self._write_manifest(
                root,
                [
                    {
                        "image_path": "first.bin",
                        "scene": "indoor",
                        "gt_distance_m": "0.1",
                    },
                    {
                        "image_path": "second.bin",
                        "scene": "indoor",
                        "gt_distance_m": "0.5",
                    },
                ],
            )
            output = root / "report.json"

            exit_code = evaluate_obstacle_distance.main(
                [
                    "--manifest",
                    str(manifest),
                    "--mode",
                    "diagnostic_constant",
                    "--constant-distance-m",
                    "0.1",
                    "--threshold-m",
                    "0.25",
                    "--output",
                    str(output),
                ]
            )

            self.assertEqual(exit_code, 0)
            best = json.loads(output.read_text(encoding="utf-8"))[
                "best_threshold"
            ]
            self.assertEqual(
                (best["metrics"]["tp"], best["metrics"]["fp"]),
                (1, 1),
            )
            self.assertEqual(best["metrics"]["positive_rate"], 0.5)
            self.assertEqual(best["ground_truth_threshold_m"], 0.25)

    def test_missing_image_becomes_failed_fallback_and_batch_continues(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "present.bin").write_bytes(b"rgb")
            manifest = self._write_manifest(
                root,
                [
                    {
                        "image_path": "missing.bin",
                        "scene": "indoor",
                        "gt_distance_m": "0.5",
                    },
                    {
                        "image_path": "present.bin",
                        "scene": "vehicle",
                        "gt_distance_m": "2.0",
                    },
                ],
            )
            output = root / "report.json"

            exit_code = evaluate_obstacle_distance.main(
                [
                    "--manifest",
                    str(manifest),
                    "--mode",
                    "diagnostic_constant",
                    "--constant-distance-m",
                    "2.0",
                    "--output",
                    str(output),
                ]
            )

            self.assertEqual(exit_code, 0)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(len(report["predictions"]), 2)
            self.assertEqual(report["overall"]["failures"], 1)
            self.assertEqual(report["overall"]["valid_predictions"], 1)
            self.assertTrue(report["predictions"][0]["fallback"])
            self.assertEqual(
                report["predictions"][0]["error_code"],
                "invalid_image",
            )
            self.assertFalse(report["predictions"][1]["fallback"])
            self.assertEqual(
                report["by_error_code"]["invalid_image"]["samples"],
                1,
            )

    def test_model_mode_uses_patched_safe_backend_factory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "frame.bin").write_bytes(b"rgb")
            manifest = self._write_manifest(
                root,
                [
                    {
                        "image_path": "frame.bin",
                        "scene": "indoor",
                        "gt_distance_m": "2.0",
                    }
                ],
            )
            config = root / "config.json"
            config.write_text(
                json.dumps(
                    {
                        "indoor": {
                            "roi": [0, 1, 0, 1],
                            "min_valid_pixels": 1,
                        }
                    }
                ),
                encoding="utf-8",
            )
            depth_backend = _DepthBackend()
            fake_factory = mock.Mock(
                return_value=(depth_backend, _SegmentationBackend())
            )

            with mock.patch(
                "perception.plugins.obstacle_distance_core.backend_loader."
                "load_backend_factory",
                return_value=fake_factory,
            ) as loader:
                exit_code = evaluate_obstacle_distance.main(
                    [
                        "--manifest",
                        str(manifest),
                        "--mode",
                        "model",
                        "--backend-factory",
                        "tests.fake:create",
                        "--config",
                        str(config),
                        "--output",
                        str(root / "report.json"),
                    ]
                )

            self.assertEqual(exit_code, 0)
            loader.assert_called_once_with("tests.fake:create")
            fake_factory.assert_called_once()
            self.assertEqual(depth_backend.calls[0][0], b"rgb")

    def test_default_model_vehicle_without_calibration_is_failed_fallback(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "frame.bin").write_bytes(b"rgb")
            manifest = self._write_manifest(
                root,
                [
                    {
                        "image_path": "frame.bin",
                        "scene": "vehicle",
                        "gt_distance_m": "2.0",
                    }
                ],
            )
            output = root / "report.json"
            fake_factory = mock.Mock(
                return_value=(_DepthBackend(), _SegmentationBackend())
            )

            with mock.patch(
                "perception.plugins.obstacle_distance_core.backend_loader."
                "load_backend_factory",
                return_value=fake_factory,
            ):
                exit_code = evaluate_obstacle_distance.main(
                    [
                        "--manifest",
                        str(manifest),
                        "--mode",
                        "model",
                        "--backend-factory",
                        "tests.fake:create",
                        "--output",
                        str(output),
                    ]
                )

            self.assertEqual(exit_code, 0)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["overall"]["failures"], 1)
            self.assertTrue(report["predictions"][0]["fallback"])
            self.assertEqual(
                report["predictions"][0]["error_code"],
                "missing_calibration",
            )

    def test_explicit_empty_calibration_clears_existing_mapping(self):
        merged = evaluate_obstacle_distance._recursive_merge(
            {
                "vehicle": {
                    "calibration": {
                        "fx": 600.0,
                        "private_file_value": "must-be-cleared",
                    }
                }
            },
            {"vehicle": {"calibration": {}}},
        )

        self.assertEqual(merged["vehicle"]["calibration"], {})
        self.assertEqual(
            evaluate_obstacle_distance._DEFAULT_CONFIG["vehicle"][
                "calibration"
            ],
            {},
        )

    def test_empty_mapping_clear_is_limited_to_vehicle_calibration(self):
        base = {
            "indoor": {"roi": [1, 2, 3, 4], "percentile": 1.0},
            "vehicle": {
                "min_confidence": 0.25,
                "calibration": {"fx": 600.0},
            },
        }

        merged = evaluate_obstacle_distance._recursive_merge(
            base,
            {
                "indoor": {},
                "vehicle": {},
                "new_section": MappingProxyType({}),
            },
        )

        self.assertEqual(merged["indoor"], base["indoor"])
        self.assertEqual(merged["vehicle"], base["vehicle"])
        self.assertEqual(merged["new_section"], {})

    def test_model_mode_without_factory_returns_nonzero(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = self._write_manifest(
                temp_dir,
                [
                    {
                        "image_path": "missing.bin",
                        "scene": "indoor",
                        "gt_distance_m": "1.0",
                    }
                ],
            )
            stderr = io.StringIO()

            with contextlib.redirect_stderr(stderr):
                exit_code = evaluate_obstacle_distance.main(
                    ["--manifest", str(manifest), "--mode", "model"]
                )

            self.assertNotEqual(exit_code, 0)
            self.assertIn("backend factory", stderr.getvalue().lower())

    def test_bad_manifest_and_config_fail_safely(self):
        secret = "password=private-config-secret"
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            missing_field = self._write_manifest(
                root,
                [{"image_path": "frame.bin", "scene": "indoor"}],
                header=["image_path", "scene"],
            )
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                exit_code = evaluate_obstacle_distance.main(
                    [
                        "--manifest",
                        str(missing_field),
                        "--mode",
                        "diagnostic_constant",
                        "--constant-distance-m",
                        "0.5",
                    ]
                )
            self.assertNotEqual(exit_code, 0)
            self.assertIn("manifest", stderr.getvalue().lower())

            bad_gt = self._write_manifest(
                root,
                [
                    {
                        "image_path": "frame.bin",
                        "scene": "indoor",
                        "gt_distance_m": secret,
                    }
                ],
            )
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                exit_code = evaluate_obstacle_distance.main(
                    [
                        "--manifest",
                        str(bad_gt),
                        "--mode",
                        "diagnostic_constant",
                        "--constant-distance-m",
                        "0.5",
                    ]
                )
            self.assertNotEqual(exit_code, 0)
            self.assertNotIn(secret, stderr.getvalue())

            bad_config = root / "bad.json"
            bad_config.write_text(
                '{"password": "' + secret + '",',
                encoding="utf-8",
            )
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                exit_code = evaluate_obstacle_distance.main(
                    [
                        "--manifest",
                        str(bad_gt),
                        "--mode",
                        "diagnostic_constant",
                        "--constant-distance-m",
                        "0.5",
                        "--config",
                        str(bad_config),
                    ]
                )
            self.assertNotEqual(exit_code, 0)
            self.assertIn("config", stderr.getvalue().lower())
            self.assertNotIn(secret, stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
