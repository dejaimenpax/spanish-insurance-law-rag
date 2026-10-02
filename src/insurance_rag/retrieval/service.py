"""Retrieval: literal reference lookup + hybrid (dense + BM25) search with optional reranking."""

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

import structlog

from insurance_rag.domain.models import Chunk
from insurance_rag.embeddings.base import Embedder
from insurance_rag.embeddings.sparse import Bm25Encoder
from insurance_rag.index.qdrant_store import ChunkFilter, QdrantStore, ScoredChunk, SearchMode
from insurance_rag.retrieval.references import ReferenceParser

log = structlog.get_logger(__name__)

# Score given to exact reference matches so they rank above any fused score.
REFERENCE_SCORE = 1.0


class Reranker(Protocol):
    def rerank(self, query: str, chunks: Sequence[Chunk]) -> list[float]: ...


@dataclass
class RetrievalResult:
    query: str
    hits: list[ScoredChunk]
    timings_ms: dict[str, float] = field(default_factory=dict)
    referenced_norms: list[str] = field(default_factory=list)


class Retriever:
    def __init__(
        self,
        *,
        store: QdrantStore,
        embedder: Embedder,
        bm25: Bm25Encoder,
        references: ReferenceParser,
        reranker: Reranker | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.bm25 = bm25
        self.references = references
        self.reranker = reranker

    def retrieve(
        self,
        query: str,
        *,
        k: int = 8,
        mode: SearchMode = "hybrid",
        chunk_filter: ChunkFilter | None = None,
        use_references: bool = True,
        candidates: int = 50,
    ) -> RetrievalResult:
        chunk_filter = chunk_filter or ChunkFilter()
        timings: dict[str, float] = {}

        start = time.perf_counter()
        exact = self._reference_hits(query, chunk_filter) if use_references else []
        timings["references"] = _ms(start)

        start = time.perf_counter()
        dense = self.embedder.embed_query(query) if mode in ("dense", "hybrid") else None
        timings["embed"] = _ms(start)
        sparse = self.bm25.encode_query(query) if mode in ("sparse", "hybrid") else None

        start = time.perf_counter()
        pool = candidates if self.reranker else k
        found = self.store.search(
            dense=dense,
            sparse=sparse,
            mode=mode,
            limit=pool,
            chunk_filter=chunk_filter,
            candidates=candidates,
        )
        timings["search"] = _ms(start)

        if self.reranker and found:
            start = time.perf_counter()
            scores = self.reranker.rerank(query, [h.chunk for h in found])
            found = sorted(
                (
                    ScoredChunk(h.chunk, s, (*h.sources, "rerank"))
                    for h, s in zip(found, scores, strict=True)
                ),
                key=lambda h: h.score,
                reverse=True,
            )
            timings["rerank"] = _ms(start)

        hits = _merge(exact, found, k)
        log.debug("retrieval.done", query=query, hits=len(hits), exact=len(exact), **timings)
        return RetrievalResult(
            query, hits, timings, referenced_norms=self.references.mentioned_norms(query)
        )

    def _reference_hits(self, query: str, chunk_filter: ChunkFilter) -> list[ScoredChunk]:
        hits: list[ScoredChunk] = []
        for ref in self.references.parse(query):
            norm_ids = [ref.norm_id] if ref.norm_id else list(chunk_filter.norm_ids)
            if not norm_ids:
                # "artículo 10" with no norm named or selected is too ambiguous to look up.
                continue
            chunks = self.store.find_provision(
                kind=ref.kind, number=ref.number, norm_ids=norm_ids, apartado=ref.apartado, limit=3
            )
            hits.extend(ScoredChunk(c, REFERENCE_SCORE, ("reference",)) for c in chunks)
        return hits


def _merge(
    first: Sequence[ScoredChunk], second: Sequence[ScoredChunk], k: int
) -> list[ScoredChunk]:
    seen: set[str] = set()
    merged: list[ScoredChunk] = []
    for hit in [*first, *second]:
        if hit.chunk.chunk_id in seen:
            continue
        seen.add(hit.chunk.chunk_id)
        merged.append(hit)
    return merged[:k]


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)
