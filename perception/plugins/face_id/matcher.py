"""Closed-set centroid and sub-center identity matching."""

from __future__ import annotations

import numpy as np

from .gallery import IdentityGallery
from .schema import IdentityMatch, l2_normalize


class IdentityMatcher:
    def __init__(
        self,
        gallery: IdentityGallery,
        *,
        centroid_weight: float = 0.6,
        unknown_threshold: float | None = None,
    ):
        if not 0.0 <= centroid_weight <= 1.0:
            raise ValueError("centroid_weight must be between 0 and 1")
        self.gallery = gallery
        self.centroid_weight = float(centroid_weight)
        self.unknown_threshold = (
            None if unknown_threshold is None else float(unknown_threshold)
        )
        self.person_ids = [template.person_id for template in gallery.templates]
        self.centroids = np.vstack(
            [template.centroid for template in gallery.templates]
        ).astype(np.float32)

    def _combined_scores(
        self, embedding: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        query = l2_normalize(embedding)
        centroid_scores = self.centroids @ query
        subcenter_scores = np.asarray(
            [
                float(np.max(template.subcenters @ query))
                for template in self.gallery.templates
            ],
            dtype=np.float32,
        )
        combined = (
            self.centroid_weight * centroid_scores
            + (1.0 - self.centroid_weight) * subcenter_scores
        )
        return combined, centroid_scores, subcenter_scores

    def rank(
        self,
        embedding: np.ndarray,
        *,
        top_k: int | None = None,
    ) -> list[IdentityMatch]:
        combined, centroid_scores, subcenter_scores = self._combined_scores(embedding)
        order = np.argsort(-combined, kind="stable")
        if top_k is not None:
            if top_k < 1:
                raise ValueError("top_k must be at least 1")
            order = order[:top_k]
        return [
            IdentityMatch(
                person_id=self.person_ids[int(index)],
                score=float(combined[index]),
                centroid_score=float(centroid_scores[index]),
                subcenter_score=float(subcenter_scores[index]),
            )
            for index in order
        ]

    def match(self, embedding: np.ndarray) -> IdentityMatch | None:
        ranked = self.rank(embedding, top_k=1)
        best = ranked[0]
        score = best.score
        if self.unknown_threshold is not None and score < self.unknown_threshold:
            return None
        return best

    def rank_with_subcenter_rescue(
        self,
        embedding: np.ndarray,
        *,
        margin_max: float,
        min_margin_gain: float,
        top_k: int | None = None,
    ) -> list[IdentityMatch]:
        """Let exemplar/subcenter evidence overturn only a fragile Top-2 result."""

        if margin_max < 0.0:
            raise ValueError("subcenter rescue margin_max must be non-negative")
        if min_margin_gain < 0.0:
            raise ValueError("subcenter rescue margin gain must be non-negative")
        if top_k is not None and top_k < 1:
            raise ValueError("top_k must be at least 1")
        requested = None if top_k is None else max(2, top_k)
        baseline = self.rank(embedding, top_k=requested)
        if len(baseline) < 2:
            return baseline if top_k is None else baseline[:top_k]
        baseline_margin = baseline[0].score - baseline[1].score
        rescue_margin = baseline[1].subcenter_score - baseline[0].subcenter_score
        if (
            baseline_margin > margin_max
            or rescue_margin < baseline_margin + min_margin_gain
        ):
            return baseline if top_k is None else baseline[:top_k]
        reranked = [
            IdentityMatch(
                person_id=baseline[1].person_id,
                score=baseline[1].subcenter_score,
                centroid_score=baseline[1].centroid_score,
                subcenter_score=baseline[1].subcenter_score,
            ),
            IdentityMatch(
                person_id=baseline[0].person_id,
                score=baseline[0].subcenter_score,
                centroid_score=baseline[0].centroid_score,
                subcenter_score=baseline[0].subcenter_score,
            ),
        ]
        reranked.extend(baseline[2:])
        return reranked if top_k is None else reranked[:top_k]

    @staticmethod
    def confidence(score: float) -> float:
        return float(np.clip((float(score) + 1.0) * 0.5, 0.0, 1.0))
