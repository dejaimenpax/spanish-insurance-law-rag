"""Qdrant collection holding chunks with a dense (semantic) and a sparse (BM25) vector each."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import structlog
from qdrant_client import QdrantClient
from qdrant_client import models as qm

from insurance_rag.domain.models import Chunk, ProvisionKind
from insurance_rag.embeddings.sparse import SparseVector

log = structlog.get_logger(__name__)

DENSE = "dense"
SPARSE = "bm25"
_KEYWORD_FIELDS = ("norm_id", "block_id", "number", "kind", "jurisdiction", "topics", "apartado")

SearchMode = Literal["dense", "sparse", "hybrid"]


@dataclass(frozen=True)
class ChunkFilter:
    norm_ids: tuple[str, ...] = ()
    topics: tuple[str, ...] = ()
    jurisdictions: tuple[str, ...] = ()
    include_repealed: bool = False

    def to_qdrant(self, *extra: qm.Condition) -> qm.Filter | None:
        must: list[qm.Condition] = list(extra)
        if self.norm_ids:
            must.append(
                qm.FieldCondition(key="norm_id", match=qm.MatchAny(any=list(self.norm_ids)))
            )
        if self.topics:
            must.append(qm.FieldCondition(key="topics", match=qm.MatchAny(any=list(self.topics))))
        if self.jurisdictions:
            must.append(
                qm.FieldCondition(
                    key="jurisdiction", match=qm.MatchAny(any=list(self.jurisdictions))
                )
            )
        must_not: list[qm.Condition] = []
        if not self.include_repealed:
            must_not.append(qm.FieldCondition(key="repealed", match=qm.MatchValue(value=True)))
        if not must and not must_not:
            return None
        return qm.Filter(must=must or None, must_not=must_not or None)


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float
    sources: tuple[str, ...] = field(default=())


def chunk_payload(chunk: Chunk) -> dict[str, Any]:
    return chunk.model_dump(mode="json", exclude={"embed_text"})


def chunk_from_payload(payload: dict[str, Any]) -> Chunk:
    return Chunk.model_validate({**payload, "embed_text": ""})


class QdrantStore:
    def __init__(self, url: str, collection: str = "chunks", *, timeout: int = 30) -> None:
        self.collection = collection
        self.client = QdrantClient(url=url, timeout=timeout)

    def ready(self) -> bool:
        try:
            return bool(self.client.collection_exists(self.collection))
        except Exception:  # any connection error means "not ready"
            return False

    def ensure_collection(self, dimension: int, *, recreate: bool = False) -> None:
        exists = self.client.collection_exists(self.collection)
        if exists and recreate:
            self.client.delete_collection(self.collection)
            exists = False
        if exists:
            info = self.client.get_collection(self.collection)
            vectors = info.config.params.vectors
            current = vectors.get(DENSE) if isinstance(vectors, dict) else None
            if current is None or current.size != dimension:
                raise ValueError(
                    f"Collection {self.collection!r} has a different dense vector size; "
                    "re-run ingestion with --recreate"
                )
            return
        log.info("qdrant.create_collection", collection=self.collection, dimension=dimension)
        self.client.create_collection(
            self.collection,
            vectors_config={DENSE: qm.VectorParams(size=dimension, distance=qm.Distance.COSINE)},
            sparse_vectors_config={SPARSE: qm.SparseVectorParams(modifier=qm.Modifier.IDF)},
        )
        for name in _KEYWORD_FIELDS:
            self.client.create_payload_index(
                self.collection, name, field_schema=qm.PayloadSchemaType.KEYWORD
            )
        self.client.create_payload_index(
            self.collection, "repealed", field_schema=qm.PayloadSchemaType.BOOL
        )

    def upsert(
        self,
        chunks: Sequence[Chunk],
        dense: Sequence[Sequence[float]],
        sparse: Sequence[SparseVector],
        *,
        batch_size: int = 128,
    ) -> None:
        points = [
            qm.PointStruct(
                id=chunk.chunk_id,
                vector={
                    DENSE: list(d),
                    SPARSE: qm.SparseVector(indices=s.indices, values=s.values),
                },
                payload=chunk_payload(chunk),
            )
            for chunk, d, s in zip(chunks, dense, sparse, strict=True)
        ]
        for start in range(0, len(points), batch_size):
            self.client.upsert(self.collection, points[start : start + batch_size], wait=True)

    def delete_blocks(self, norm_id: str, block_ids: Iterable[str]) -> None:
        ids = list(block_ids)
        if not ids:
            return
        self._delete(
            qm.Filter(
                must=[
                    qm.FieldCondition(key="norm_id", match=qm.MatchValue(value=norm_id)),
                    qm.FieldCondition(key="block_id", match=qm.MatchAny(any=ids)),
                ]
            )
        )

    def delete_norm(self, norm_id: str) -> None:
        self._delete(
            qm.Filter(must=[qm.FieldCondition(key="norm_id", match=qm.MatchValue(value=norm_id))])
        )

    def set_norm_payload(self, norm_id: str, payload: dict[str, Any]) -> None:
        self.client.set_payload(
            self.collection,
            payload=payload,
            points=qm.Filter(
                must=[qm.FieldCondition(key="norm_id", match=qm.MatchValue(value=norm_id))]
            ),
            wait=True,
        )

    def count(self, norm_id: str | None = None) -> int:
        flt = (
            qm.Filter(must=[qm.FieldCondition(key="norm_id", match=qm.MatchValue(value=norm_id))])
            if norm_id
            else None
        )
        return self.client.count(self.collection, count_filter=flt, exact=True).count

    def search(
        self,
        *,
        dense: Sequence[float] | None,
        sparse: SparseVector | None,
        mode: SearchMode,
        limit: int,
        chunk_filter: ChunkFilter,
        candidates: int = 50,
    ) -> list[ScoredChunk]:
        flt = chunk_filter.to_qdrant()
        if mode == "dense":
            if dense is None:
                raise ValueError("dense mode needs a dense query vector")
            response = self.client.query_points(
                self.collection, query=list(dense), using=DENSE, query_filter=flt, limit=limit
            )
        elif mode == "sparse":
            if sparse is None:
                raise ValueError("sparse mode needs a sparse query vector")
            response = self.client.query_points(
                self.collection,
                query=qm.SparseVector(indices=sparse.indices, values=sparse.values),
                using=SPARSE,
                query_filter=flt,
                limit=limit,
            )
        else:
            if dense is None or sparse is None:
                raise ValueError("hybrid mode needs dense and sparse query vectors")
            response = self.client.query_points(
                self.collection,
                prefetch=[
                    qm.Prefetch(query=list(dense), using=DENSE, filter=flt, limit=candidates),
                    qm.Prefetch(
                        query=qm.SparseVector(indices=sparse.indices, values=sparse.values),
                        using=SPARSE,
                        filter=flt,
                        limit=candidates,
                    ),
                ],
                query=qm.FusionQuery(fusion=qm.Fusion.RRF),
                limit=limit,
            )
        return [
            ScoredChunk(chunk_from_payload(p.payload or {}), float(p.score), (mode,))
            for p in response.points
        ]

    def find_provision(
        self,
        *,
        kind: ProvisionKind,
        number: str,
        norm_ids: Sequence[str],
        apartado: str | None = None,
        limit: int = 10,
    ) -> list[Chunk]:
        """Exact lookup of a provision by its coordinates (e.g. LCS, article 10)."""
        conditions: list[qm.Condition] = [
            qm.FieldCondition(key="kind", match=qm.MatchValue(value=kind.value)),
            qm.FieldCondition(key="number", match=qm.MatchValue(value=number)),
        ]
        if norm_ids:
            conditions.append(
                qm.FieldCondition(key="norm_id", match=qm.MatchAny(any=list(norm_ids)))
            )
        points, _ = self.client.scroll(
            self.collection, scroll_filter=qm.Filter(must=conditions), limit=200
        )
        chunks = [chunk_from_payload(p.payload or {}) for p in points]
        chunks.sort(key=lambda c: (c.norm_id, c.seq))
        if apartado is not None:
            matching = [c for c in chunks if c.apartado == apartado]
            # Short articles are a single chunk without apartado; keep them as the answer.
            chunks = matching or [c for c in chunks if c.apartado is None] or chunks
        return chunks[:limit]

    def find_by_block(self, norm_id: str, block_id: str) -> list[Chunk]:
        points, _ = self.client.scroll(
            self.collection,
            scroll_filter=qm.Filter(
                must=[
                    qm.FieldCondition(key="norm_id", match=qm.MatchValue(value=norm_id)),
                    qm.FieldCondition(key="block_id", match=qm.MatchValue(value=block_id)),
                ]
            ),
            limit=200,
        )
        return sorted((chunk_from_payload(p.payload or {}) for p in points), key=lambda c: c.seq)

    def _delete(self, flt: qm.Filter) -> None:
        self.client.delete(
            self.collection, points_selector=qm.FilterSelector(filter=flt), wait=True
        )
