"""Hybrid retrieval using Reciprocal Rank Fusion."""

from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class HybridResult:
    """One result produced by hybrid rank fusion."""

    chunk_id: str
    fused_score: float
    source_ranks: Mapping[str, int]


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[str]],
    *,
    rank_constant: int = 60,
    weights: Mapping[str, float] | None = None,
) -> list[HybridResult]:
    """Combine multiple ranked chunk-ID lists using RRF.

    Args:
        rankings:
            Mapping from retrieval-source name to ordered chunk IDs.
            Rank position 1 must contain that source's best result.
        rank_constant:
            RRF smoothing constant. The conventional default is 60.
        weights:
            Optional importance weight for each retrieval source.

    Returns:
        Results ordered from highest to lowest fused score.
    """

    if rank_constant < 0:
        raise ValueError("rank_constant must be zero or greater")

    source_weights = {
        source_name: 1.0
        for source_name in rankings
    }

    if weights is not None:
        for source_name, weight in weights.items():
            if weight < 0:
                raise ValueError(
                    "retrieval weights cannot be negative"
                )

            if source_name in source_weights:
                source_weights[source_name] = weight

    fused_scores: dict[str, float] = {}
    source_ranks: dict[str, dict[str, int]] = {}
    first_seen_order: dict[str, int] = {}
    next_order = 0

    for source_name, ranked_chunk_ids in rankings.items():
        seen_chunk_ids: set[str] = set()
        unique_rank = 0

        for chunk_id in ranked_chunk_ids:
            if chunk_id in seen_chunk_ids:
                continue

            seen_chunk_ids.add(chunk_id)
            unique_rank += 1

            if chunk_id not in first_seen_order:
                first_seen_order[chunk_id] = next_order
                next_order += 1

            contribution = (
                source_weights[source_name]
                / (rank_constant + unique_rank)
            )

            fused_scores[chunk_id] = (
                fused_scores.get(chunk_id, 0.0)
                + contribution
            )

            source_ranks.setdefault(chunk_id, {})[
                source_name
            ] = unique_rank

    ordered_chunk_ids = sorted(
        fused_scores,
        key=lambda chunk_id: (
            -fused_scores[chunk_id],
            min(source_ranks[chunk_id].values()),
            first_seen_order[chunk_id],
        ),
    )

    return [
        HybridResult(
            chunk_id=chunk_id,
            fused_score=fused_scores[chunk_id],
            source_ranks=source_ranks[chunk_id],
        )
        for chunk_id in ordered_chunk_ids
    ]