"""Keep the Qdrant collection in sync with the BOE, re-indexing only what changed."""

import hashlib
import json
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date

import structlog

from insurance_rag.domain.models import Chunk, NormSpec
from insurance_rag.embeddings.base import Embedder
from insurance_rag.embeddings.sparse import Bm25Encoder
from insurance_rag.index.qdrant_store import QdrantStore
from insurance_rag.ingestion.boe.client import BoeClient, IndexEntry
from insurance_rag.ingestion.boe.parser import parse_blocks
from insurance_rag.ingestion.pipeline import RawStore, fetch_snapshot, process_norm
from insurance_rag.ingestion.state import IngestionState, NormState, text_hash
from insurance_rag.ingestion.structure import next_scheduled_change

log = structlog.get_logger(__name__)

# Bump when parsing or chunking changes in a way that alters chunks: forces a full rebuild.
PIPELINE_VERSION = "boe-chunks-v2"


@dataclass
class SyncReport:
    norm_id: str
    short_name: str
    status: str
    """"skipped" (nothing to do), "updated" or "created"."""
    reason: str = ""
    blocks_changed: int = 0
    blocks_removed: int = 0
    chunks_indexed: int = 0
    embeddings_computed: int = 0
    consolidated_as_of: date | None = None
    notes: list[str] = field(default_factory=list)


def index_fingerprint(index: Sequence[IndexEntry]) -> str:
    payload = [(e.block_id, e.updated_at.isoformat()) for e in index]
    return hashlib.sha256(json.dumps(payload).encode()).hexdigest()


def block_content_hash(chunks: Sequence[Chunk]) -> str:
    """Hash of everything indexed for a block except norm-wide fields that change together."""
    payload = [
        c.model_dump(mode="json", exclude={"consolidated_as_of", "chunk_id"}) for c in chunks
    ]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class Indexer:
    def __init__(
        self,
        *,
        client: BoeClient,
        raw_store: RawStore,
        state: IngestionState,
        store: QdrantStore,
        embedder: Embedder,
        bm25: Bm25Encoder,
        max_chars: int,
    ) -> None:
        self.client = client
        self.raw_store = raw_store
        self.state = state
        self.store = store
        self.embedder = embedder
        self.bm25 = bm25
        self.max_chars = max_chars

    def prepare(self, *, recreate: bool = False) -> None:
        """Create the collection; rebuild everything if the pipeline or model changed."""
        signature = f"{PIPELINE_VERSION}|{self.embedder.model_name}|{self.max_chars}"
        if self.state.get_meta("signature") != signature:
            if self.state.get_meta("signature") is not None:
                log.info("index.signature_changed", new=signature)
            recreate = True
        if not self.store.ready():
            recreate = True
        self.store.ensure_collection(self.embedder.dimension, recreate=recreate)
        if recreate:
            self.state.reset_index_state()
            self.state.set_meta("signature", signature)
        avg = self.state.get_meta("bm25_avg_doc_len")
        if avg is not None:
            self.bm25.avg_doc_len = float(avg)

    def sync(
        self,
        specs: Sequence[NormSpec],
        *,
        as_of: date,
        force: bool = False,
        catalog_ids: Sequence[str] | None = None,
    ) -> list[SyncReport]:
        """Sync ``specs``. When ``catalog_ids`` is given, norms not in it are removed."""
        if catalog_ids is not None:
            for norm_id in self.state.forget_norms_except(catalog_ids):
                log.info("index.norm_removed", norm=norm_id)
                self.store.delete_norm(norm_id)
        return [self._sync_norm(spec, as_of=as_of, force=force) for spec in specs]

    def _sync_norm(self, spec: NormSpec, *, as_of: date, force: bool) -> SyncReport:
        previous = self.state.get_norm(spec.id)
        index = self.client.fetch_index(spec.id)
        fingerprint = index_fingerprint(index)

        reason = self._due_reason(previous, fingerprint, as_of=as_of, force=force)
        if reason is None and previous is not None:
            return SyncReport(
                spec.id,
                spec.short_name,
                "skipped",
                "index unchanged",
                consolidated_as_of=previous.consolidated_as_of,
            )

        snapshot = fetch_snapshot(
            self.client,
            self.raw_store,
            spec.id,
            refresh=previous is None or previous.index_fingerprint != fingerprint,
        )
        result = process_norm(spec, snapshot, as_of=as_of, max_chars=self.max_chars)
        self._fit_bm25_if_new([c.embed_text for c in result.chunks])

        by_block: dict[str, list[Chunk]] = defaultdict(list)
        for chunk in result.chunks:
            by_block[chunk.block_id].append(chunk)
        new_hashes = {block_id: block_content_hash(cs) for block_id, cs in by_block.items()}
        old_hashes = self.state.block_hashes(spec.id)
        changed = [b for b, h in new_hashes.items() if old_hashes.get(b) != h]
        removed = [b for b in old_hashes if b not in new_hashes]

        self.store.delete_blocks(spec.id, [*changed, *removed])
        to_index = [c for b in changed for c in by_block[b]]
        computed = self._index_chunks(to_index)
        if previous is not None and previous.consolidated_as_of != result.norm.consolidated_as_of:
            self.store.set_norm_payload(
                spec.id, {"consolidated_as_of": result.norm.consolidated_as_of.isoformat()}
            )

        next_change = next_scheduled_change(spec, parse_blocks(snapshot.text_xml), as_of)
        self.state.save_norm(
            spec.id,
            NormState(fingerprint, result.norm.consolidated_as_of, next_change),
            new_hashes,
        )
        return SyncReport(
            spec.id,
            spec.short_name,
            "created" if previous is None else "updated",
            reason or "",
            blocks_changed=len(changed),
            blocks_removed=len(removed),
            chunks_indexed=len(to_index),
            embeddings_computed=computed,
            consolidated_as_of=result.norm.consolidated_as_of,
        )

    @staticmethod
    def _due_reason(
        previous: NormState | None, fingerprint: str, *, as_of: date, force: bool
    ) -> str | None:
        """Why the norm must be re-processed, or None if it is up to date."""
        if previous is None:
            return "not indexed yet"
        if force:
            return "forced"
        if previous.index_fingerprint != fingerprint:
            return "BOE index changed"
        if previous.next_change is not None and as_of >= previous.next_change:
            return f"scheduled change in force since {previous.next_change}"
        return None

    def _fit_bm25_if_new(self, texts: list[str]) -> None:
        if self.state.get_meta("bm25_avg_doc_len") is None:
            avg = self.bm25.fit_avg_doc_len(texts)
            self.state.set_meta("bm25_avg_doc_len", f"{avg:.3f}")

    def _index_chunks(self, chunks: Sequence[Chunk]) -> int:
        if not chunks:
            return 0
        model = self.embedder.model_name
        hashes = [text_hash(c.embed_text) for c in chunks]
        cached = self.state.cached_embeddings(model, list(set(hashes)))
        missing = {h: c.embed_text for h, c in zip(hashes, chunks, strict=True) if h not in cached}
        if missing:
            keys = list(missing)
            vectors = self.embedder.embed_documents([missing[k] for k in keys])
            fresh = dict(zip(keys, vectors, strict=True))
            self.state.cache_embeddings(model, fresh)
            cached.update(fresh)
        dense = [cached[h] for h in hashes]
        sparse = [self.bm25.encode_document(c.embed_text) for c in chunks]
        self.store.upsert(chunks, dense, sparse)
        return len(missing)
