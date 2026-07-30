from __future__ import annotations

import math
from collections.abc import Mapping, Sized
from dataclasses import dataclass
from numbers import Real


@dataclass(frozen=True)
class EvaluationMetrics:
    samples: int
    valid_predictions: int
    failures: int
    failure_rate: float
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    rmse: float | None
    positive_rate: float


@dataclass(frozen=True)
class ThresholdScanResult:
    threshold_m: float
    metrics: EvaluationMetrics


def _values(value: object, *, name: str) -> tuple[object, ...]:
    if (
        isinstance(value, (str, bytes, bytearray, memoryview, Mapping))
        or not isinstance(value, Sized)
    ):
        raise ValueError(f"{name} must be a sequence")
    try:
        expected_length = len(value)
    except Exception:
        raise ValueError(f"{name} must be a sequence") from None
    try:
        values = tuple(value)
    except Exception:
        raise ValueError(f"{name} must be a sequence") from None
    if len(values) != expected_length:
        raise ValueError(f"{name} must be a sequence")
    return values


def _finite_real(
    value: object,
    *,
    name: str,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be a finite real number") from None
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be a finite real number")
    if positive and converted <= 0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and converted < 0:
        raise ValueError(f"{name} must be nonnegative")
    return converted


def _validated_inputs(
    ground_truth_m: object,
    predictions_m: object,
    failed: object | None,
) -> tuple[tuple[float, ...], tuple[float | None, ...], tuple[bool, ...]]:
    ground_truth_values = _values(ground_truth_m, name="ground truth")
    prediction_values = _values(predictions_m, name="predictions")
    if not ground_truth_values:
        raise ValueError("evaluation input must not be empty")
    if len(ground_truth_values) != len(prediction_values):
        raise ValueError("ground truth and predictions must have equal length")

    if failed is None:
        failure_flags = (False,) * len(ground_truth_values)
    else:
        failure_flags = _values(failed, name="failed flags")
        if len(failure_flags) != len(ground_truth_values):
            raise ValueError("failed flags must match evaluation input length")
        if any(type(flag) is not bool for flag in failure_flags):
            raise ValueError("failed flags must contain only bool values")

    validated_ground_truth = tuple(
        _finite_real(value, name="ground truth", nonnegative=True)
        for value in ground_truth_values
    )
    validated_predictions: list[float | None] = []
    for value, explicit_failure in zip(
        prediction_values,
        failure_flags,
        strict=True,
    ):
        if explicit_failure:
            validated_predictions.append(None)
            continue
        if isinstance(value, bool) or not isinstance(value, Real):
            validated_predictions.append(None)
            continue
        try:
            converted = float(value)
        except (TypeError, ValueError, OverflowError):
            validated_predictions.append(None)
            continue
        if not math.isfinite(converted) or converted < 0:
            validated_predictions.append(None)
            continue
        validated_predictions.append(converted)

    return (
        validated_ground_truth,
        tuple(validated_predictions),
        tuple(failure_flags),
    )


def _stable_rmse(errors: list[float]) -> float | None:
    if not errors:
        return None
    scale = max(abs(error) for error in errors)
    if scale == 0:
        return 0.0
    normalized_mean = math.fsum(
        (error / scale) ** 2 for error in errors
    ) / len(errors)
    return scale * math.sqrt(normalized_mean)


def evaluate_predictions(
    gt: object,
    pred: object,
    threshold_m: object,
    failed: object | None = None,
) -> EvaluationMetrics:
    threshold = _finite_real(
        threshold_m,
        name="threshold_m",
        positive=True,
    )
    ground_truth, predictions, _ = _validated_inputs(
        gt,
        pred,
        failed,
    )

    tp = fp = fn = 0
    errors: list[float] = []
    for truth, prediction in zip(ground_truth, predictions, strict=True):
        if prediction is None:
            continue
        truth_positive = truth < threshold
        prediction_positive = prediction < threshold
        if prediction_positive and truth_positive:
            tp += 1
        elif prediction_positive:
            fp += 1
        elif truth_positive:
            fn += 1
        errors.append(prediction - truth)

    samples = len(ground_truth)
    valid_predictions = len(errors)
    failures = samples - valid_predictions
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    positive_rate = sum(value < threshold for value in ground_truth) / samples
    return EvaluationMetrics(
        samples=samples,
        valid_predictions=valid_predictions,
        failures=failures,
        failure_rate=failures / samples,
        tp=tp,
        fp=fp,
        fn=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        rmse=_stable_rmse(errors),
        positive_rate=positive_rate,
    )


def _threshold_candidates(predictions: tuple[float, ...]) -> tuple[float, ...]:
    unique = sorted(set(predictions))
    candidates: list[float] = []

    lower = unique[0]
    if lower > 0:
        candidates.append(lower)
    else:
        candidates.append(math.nextafter(0.0, math.inf))

    for low, high in zip(unique, unique[1:]):
        midpoint = low + (high - low) / 2.0
        if not low < midpoint < high:
            midpoint = high
        candidates.append(midpoint)

    upper = math.nextafter(unique[-1], math.inf)
    if math.isfinite(upper):
        candidates.append(upper)

    return tuple(dict.fromkeys(candidates))


def scan_thresholds(
    gt: object,
    pred: object,
    failed: object | None = None,
) -> ThresholdScanResult:
    ground_truth, predictions, _ = _validated_inputs(
        gt,
        pred,
        failed,
    )
    valid_records = [
        (truth, prediction)
        for truth, prediction in zip(
            ground_truth,
            predictions,
            strict=True,
        )
        if prediction is not None
    ]
    valid_predictions = tuple(
        prediction for _, prediction in valid_records
    )
    if not valid_predictions:
        raise ValueError("threshold scan requires a valid prediction")

    candidates = _threshold_candidates(valid_predictions)
    truth_events = sorted(
        (truth, index)
        for index, (truth, _) in enumerate(valid_records)
    )
    prediction_events = sorted(
        (prediction, index)
        for index, (_, prediction) in enumerate(valid_records)
    )
    sorted_ground_truth = sorted(ground_truth)
    truth_positive = [False] * len(valid_records)
    prediction_positive = [False] * len(valid_records)
    truth_event_index = 0
    prediction_event_index = 0
    ground_truth_index = 0
    tp = fp = fn = 0

    errors = [
        prediction - truth for truth, prediction in valid_records
    ]
    samples = len(ground_truth)
    valid_count = len(valid_records)
    failures = samples - valid_count
    rmse = _stable_rmse(errors)

    best: ThresholdScanResult | None = None
    best_key: tuple[float, float, float, float] | None = None
    for threshold in candidates:
        while (
            truth_event_index < valid_count
            and truth_events[truth_event_index][0] < threshold
        ):
            _, index = truth_events[truth_event_index]
            truth_positive[index] = True
            if prediction_positive[index]:
                fp -= 1
                tp += 1
            else:
                fn += 1
            truth_event_index += 1

        while (
            prediction_event_index < valid_count
            and prediction_events[prediction_event_index][0] < threshold
        ):
            _, index = prediction_events[prediction_event_index]
            prediction_positive[index] = True
            if truth_positive[index]:
                fn -= 1
                tp += 1
            else:
                fp += 1
            prediction_event_index += 1

        while (
            ground_truth_index < samples
            and sorted_ground_truth[ground_truth_index] < threshold
        ):
            ground_truth_index += 1

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        metrics = EvaluationMetrics(
            samples=samples,
            valid_predictions=valid_count,
            failures=failures,
            failure_rate=failures / samples,
            tp=tp,
            fp=fp,
            fn=fn,
            precision=precision,
            recall=recall,
            f1=f1,
            rmse=rmse,
            positive_rate=ground_truth_index / samples,
        )
        key = (
            metrics.f1,
            metrics.precision,
            -abs(threshold - 1.0),
            -threshold,
        )
        if best_key is None or key > best_key:
            best_key = key
            best = ThresholdScanResult(threshold, metrics)
    if best is None:
        raise ValueError("threshold scan requires a finite threshold")
    return best


def evaluate_grouped_predictions(
    gt: object,
    pred: object,
    groups: object,
    threshold_m: object,
    failed: object | None = None,
) -> dict[str, EvaluationMetrics]:
    threshold = _finite_real(
        threshold_m,
        name="threshold_m",
        positive=True,
    )
    ground_truth, predictions, failure_flags = _validated_inputs(
        gt,
        pred,
        failed,
    )
    labels = _values(groups, name="groups")
    if len(labels) != len(ground_truth):
        raise ValueError("groups must match evaluation input length")

    indices_by_label: dict[str, list[int]] = {}
    for index, label in enumerate(labels):
        if label is None or label == "":
            continue
        if not isinstance(label, str):
            raise ValueError("group labels must be strings")
        indices_by_label.setdefault(label, []).append(index)

    return {
        label: evaluate_predictions(
            [ground_truth[index] for index in indices],
            [predictions[index] for index in indices],
            threshold,
            failed=[failure_flags[index] for index in indices],
        )
        for label, indices in indices_by_label.items()
    }
