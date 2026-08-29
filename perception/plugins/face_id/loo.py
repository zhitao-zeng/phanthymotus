"""Leakage-aware leave-one-out diagnostics for an identity gallery."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .gallery import (
    IdentityGallery,
    IdentityTemplates,
    spherical_subcenters,
    weighted_centroid,
)
from .matcher import IdentityMatcher


@dataclass(frozen=True)
class LeaveOneOutQuery:
    person_id: str
    exemplar_index: int
    predicted_person_id: str
    true_rank: int
    true_score: float
    top1_score: float
    top2_score: float | None
    margin: float | None
    candidate_count: int = 1
    selected_candidate_index: int = 0


def _weights(template: IdentityTemplates) -> np.ndarray:
    if template.exemplar_weights is None:
        return np.ones(len(template.exemplars), dtype=np.float32)
    weights = np.asarray(template.exemplar_weights, dtype=np.float32).reshape(-1)
    if len(weights) != len(template.exemplars):
        raise ValueError(f"{template.person_id}: exemplar weights have wrong length")
    return weights


def _template_without(
    template: IdentityTemplates,
    index: int,
    *,
    max_subcenters: int,
    subcenter_min_images: int,
) -> IdentityTemplates:
    features = np.delete(np.asarray(template.exemplars, dtype=np.float32), index, axis=0)
    weights = np.delete(_weights(template), index, axis=0)
    if not len(features):
        raise ValueError("leave-one-out requires at least two exemplars per identity")
    return IdentityTemplates(
        person_id=template.person_id,
        centroid=weighted_centroid(features, weights),
        subcenters=spherical_subcenters(
            features,
            weights,
            max_subcenters=max_subcenters,
            enable_from=subcenter_min_images,
        ),
        exemplars=features,
        source_count=template.source_count,
        exemplar_weights=weights,
        query_exemplars=None,
        source_paths=None,
    )


def evaluate_gallery_leave_one_out(
    gallery: IdentityGallery,
    *,
    centroid_weight: float = 0.6,
    query_flip_tta: bool = False,
    candidate_queries: dict[tuple[str, int], np.ndarray] | None = None,
    max_subcenters: int = 3,
    subcenter_min_images: int = 4,
) -> dict:
    """Evaluate every eligible exemplar after excluding it from its own template."""

    details: list[LeaveOneOutQuery] = []
    skipped_singletons = 0
    eligible_identities = 0
    templates = list(gallery.templates)
    for template_index, template in enumerate(templates):
        exemplar_count = len(template.exemplars)
        if exemplar_count < 2:
            skipped_singletons += 1
            continue
        eligible_identities += 1
        if query_flip_tta or template.query_exemplars is None:
            queries = np.asarray(template.exemplars, dtype=np.float32)
        else:
            queries = np.asarray(template.query_exemplars, dtype=np.float32)
            if len(queries) != exemplar_count:
                raise ValueError(
                    f"{template.person_id}: query exemplars have wrong length"
                )
        for exemplar_index, query in enumerate(queries):
            loo_templates = list(templates)
            loo_templates[template_index] = _template_without(
                template,
                exemplar_index,
                max_subcenters=max_subcenters,
                subcenter_min_images=subcenter_min_images,
            )
            matcher = IdentityMatcher(
                IdentityGallery(loo_templates),
                centroid_weight=centroid_weight,
            )
            query_candidates = np.asarray(
                (candidate_queries or {}).get(
                    (template.person_id, exemplar_index), query[None]
                ),
                dtype=np.float32,
            )
            if query_candidates.ndim != 2 or not len(query_candidates):
                raise ValueError("query candidates must have shape (faces, embedding)")
            ranked_candidates = []
            for candidate_index, candidate in enumerate(query_candidates):
                ranked = matcher.rank(candidate)
                margin = ranked[0].score - ranked[1].score if len(ranked) > 1 else 0.0
                ranked_candidates.append(
                    (ranked[0].score, margin, -candidate_index, candidate_index, ranked)
                )
            _score, _margin, _order, selected_index, ranked = max(
                ranked_candidates,
                key=lambda item: item[:3],
            )
            true_rank = next(
                index + 1
                for index, match in enumerate(ranked)
                if match.person_id == template.person_id
            )
            true_score = ranked[true_rank - 1].score
            top2_score = ranked[1].score if len(ranked) > 1 else None
            margin = None if top2_score is None else ranked[0].score - top2_score
            details.append(
                LeaveOneOutQuery(
                    person_id=template.person_id,
                    exemplar_index=exemplar_index,
                    predicted_person_id=ranked[0].person_id,
                    true_rank=true_rank,
                    true_score=true_score,
                    top1_score=ranked[0].score,
                    top2_score=top2_score,
                    margin=margin,
                    candidate_count=len(query_candidates),
                    selected_candidate_index=selected_index,
                )
            )

    query_count = len(details)
    margins = [item.margin for item in details if item.margin is not None]
    return {
        "centroid_weight": float(centroid_weight),
        "query_flip_tta": bool(query_flip_tta),
        "identities": len(templates),
        "eligible_identities": eligible_identities,
        "skipped_singletons": skipped_singletons,
        "queries": query_count,
        "top1_accuracy": (
            float(np.mean([item.true_rank == 1 for item in details]))
            if details
            else None
        ),
        "top5_accuracy": (
            float(np.mean([item.true_rank <= 5 for item in details]))
            if details
            else None
        ),
        "mean_true_rank": (
            float(np.mean([item.true_rank for item in details])) if details else None
        ),
        "margin_mean": float(np.mean(margins)) if margins else None,
        "margin_p10": float(np.percentile(margins, 10)) if margins else None,
        "details": [item.__dict__ for item in details],
    }
