"""Test doubles shared by unit tests."""

from collections.abc import AsyncIterator, Sequence
from dataclasses import replace
from datetime import date
from typing import Any

from insurance_rag.domain.models import Chunk, Jurisdiction, LegalEffect, NormRank, ProvisionKind
from insurance_rag.generation.llm import CitationDelta, Completed, LlmEvent, TextDelta
from insurance_rag.index.qdrant_store import ChunkFilter, ScoredChunk
from insurance_rag.observability.cost import Usage
from insurance_rag.retrieval.service import RetrievalResult


def make_chunk(number: str = "18", **overrides: Any) -> Chunk:
    values: dict[str, Any] = {
        "chunk_id": f"chunk-{number}",
        "norm_id": "BOE-A-1980-22501",
        "norm_short_name": "LCS",
        "block_id": f"a{number}",
        "kind": ProvisionKind.ARTICULO,
        "label": f"Artículo {number}",
        "number": number,
        "heading": None,
        "apartado": None,
        "hierarchy": ("TÍTULO I",),
        "text": f"Texto del artículo {number}.\nSegundo párrafo.",
        "embed_text": "",
        "url": f"https://www.boe.es/buscar/act.php?id=BOE-A-1980-22501#a{number}",
        "version_in_force_since": date(2020, 1, 1),
        "block_updated_at": date(2020, 1, 1),
        "consolidated_as_of": date(2025, 7, 25),
        "jurisdiction": Jurisdiction.ES,
        "legal_effect": LegalEffect.NATIONAL,
        "rank": NormRank.LEY,
        "topics": ("contrato",),
    }
    values.update(overrides)
    return Chunk(**values)


class FakeRetriever:
    def __init__(self, hits: Sequence[ScoredChunk]) -> None:
        self.hits = list(hits)
        self.calls: list[tuple[str, ChunkFilter | None]] = []

    def retrieve(
        self, query: str, *, k: int = 8, chunk_filter: ChunkFilter | None = None
    ) -> RetrievalResult:
        self.calls.append((query, chunk_filter))
        return RetrievalResult(query, self.hits[:k], {"search": 1.0})


class FakeLlm:
    model = "claude-opus-5-5"

    def __init__(self, events: Sequence[LlmEvent]) -> None:
        self.events = list(events)
        self.calls: list[list[dict[str, Any]]] = []

    async def stream(
        self, *, system: str, content: list[dict[str, Any]]
    ) -> AsyncIterator[LlmEvent]:
        self.calls.append(content)
        for event in self.events:
            yield event


def answer_events(text: str = "El asegurador debe pagar en cuarenta días.") -> list[LlmEvent]:
    return [
        TextDelta(text),
        CitationDelta(0, "Texto del artículo 18."),
        Completed("claude-opus-5-5", "end_turn", Usage(input_tokens=1000, output_tokens=100)),
    ]


def scored(
    chunk: Chunk, score: float = 0.9, sources: tuple[str, ...] = ("hybrid", "rerank")
) -> ScoredChunk:
    return ScoredChunk(chunk, score, sources)


__all__ = ["FakeLlm", "FakeRetriever", "answer_events", "make_chunk", "replace", "scored"]
