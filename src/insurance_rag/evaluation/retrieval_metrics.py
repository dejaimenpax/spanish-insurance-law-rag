"""Retrieval metrics computed at provision level (any chunk of a gold provision counts)."""

from collections.abc import Sequence
from dataclasses import dataclass

from insurance_rag.domain.models import Chunk
from insurance_rag.evaluation.dataset import ProvisionKey


@dataclass(frozen=True)
class RetrievalScores:
    recall_at: dict[int, float]
    """Fraction of gold provisions found in the top k."""
    reciprocal_rank: float
    """1 / rank of the first chunk that belongs to any gold provision (0 if none)."""


def score_retrieval(
    ranked: Sequence[Chunk], gold: Sequence[ProvisionKey], ks: Sequence[int]
) -> RetrievalScores:
    if not gold:
        raise ValueError("Retrieval metrics need at least one gold provision")
    first_rank = next(
        (rank for rank, chunk in enumerate(ranked, start=1) if any(g.matches(chunk) for g in gold)),
        None,
    )
    recall = {}
    for k in ks:
        top = ranked[:k]
        found = sum(1 for g in gold if any(g.matches(c) for c in top))
        recall[k] = found / len(gold)
    return RetrievalScores(recall_at=recall, reciprocal_rank=1 / first_rank if first_rank else 0.0)


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0
