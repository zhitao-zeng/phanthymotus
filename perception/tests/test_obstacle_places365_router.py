from __future__ import annotations

import numpy as np
import pytest

from plugins.obstacle_distance_core.contracts import SceneDomain
from plugins.obstacle_distance_core.places365_router import Places365SceneRouter


def _router(outdoor_indices: set[int]) -> Places365SceneRouter:
    router = object.__new__(Places365SceneRouter)
    router._top_k = 3
    router._outdoor = np.asarray(
        [1 if index in outdoor_indices else 0 for index in range(365)],
        dtype=np.uint8,
    )
    return router


def test_route_decision_exposes_probabilities_and_margin():
    router = _router({0, 2})
    logits = np.full(365, -20.0, dtype=np.float32)
    logits[0] = 5.0
    logits[1] = 4.0
    logits[2] = 3.0

    decision = router._decision_from_logits(logits)

    assert decision.domain is SceneDomain.VEHICLE
    assert decision.outdoor_vote == pytest.approx(2 / 3)
    assert decision.top1_index == 0
    assert decision.top2_index == 1
    assert decision.top1_probability > decision.top2_probability
    assert decision.top1_top2_margin == pytest.approx(
        decision.top1_probability - decision.top2_probability
    )
    assert decision.confidence == pytest.approx(
        decision.outdoor_probability
    )


def test_route_decision_keeps_top_k_vote_as_selected_domain():
    router = _router({2})
    logits = np.full(365, -20.0, dtype=np.float32)
    logits[0] = 5.0
    logits[1] = 4.0
    logits[2] = 3.0

    decision = router._decision_from_logits(logits)

    assert decision.domain is SceneDomain.INDOOR
    assert decision.outdoor_vote == pytest.approx(1 / 3)
    assert decision.confidence == pytest.approx(
        1.0 - decision.outdoor_probability
    )
