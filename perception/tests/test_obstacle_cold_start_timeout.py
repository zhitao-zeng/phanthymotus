from __future__ import annotations

from plugins.obstacle_distance_core.contracts import DepthPrediction
from plugins.obstacle_distance_core.estimator import ObstacleDistanceEstimator


class _DepthBackend:
    def __init__(self):
        self.deadlines = []

    def predict_depth(self, image_bytes, domain, deadline_monotonic):
        return DepthPrediction([[1.0]], 1, 1)

    def predict_indoor_distance(self, image_bytes, deadline_monotonic):
        self.deadlines.append(deadline_monotonic)
        return 1.0


class _SegmentationBackend:
    def predict_instances(self, image_bytes, deadline_monotonic):
        return ()


def test_cold_start_budget_is_used_once_per_successful_domain():
    depth = _DepthBackend()
    estimator = ObstacleDistanceEstimator(
        depth,
        _SegmentationBackend(),
        {
            "fixed_scene": "indoor",
            "decision_threshold_m": 2.0,
            "fallback_distance_m": 3.0,
            "soft_timeout_s": 2.5,
            "cold_start_timeout_s": 6.0,
        },
        monotonic=lambda: 0.0,
        wall_time=lambda: 0.0,
    )

    first = estimator.estimate(b"jpeg", scene_hint="indoor")
    second = estimator.estimate(b"jpeg", scene_hint="indoor")

    assert depth.deadlines == [6.0, 2.5]
    assert first.cold_start is True
    assert first.timeout_budget_ms == 6000.0
    assert second.cold_start is False
    assert second.timeout_budget_ms == 2500.0
