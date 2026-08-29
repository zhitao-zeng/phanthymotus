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

    def match(self, embedding: np.ndarray) -> IdentityMatch | None:
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
        best = int(np.argmax(combined))
        score = float(combined[best])
        if self.unknown_threshold is not None and score < self.unknown_threshold:
            return None
        return IdentityMatch(
            person_id=self.person_ids[best],
            score=score,
            centroid_score=float(centroid_scores[best]),
            subcenter_score=float(subcenter_scores[best]),
        )

    @staticmethod
    def confidence(score: float) -> float:
        return float(np.clip((float(score) + 1.0) * 0.5, 0.0, 1.0))
