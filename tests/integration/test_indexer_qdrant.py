"""Incremental indexing against a real Qdrant. Run with ``pytest -m integration``.

Needs Qdrant at ``QDRANT_URL`` (default http://localhost:6333), e.g. ``docker compose up qdrant``.
"""

import hashlib
import os
import uuid
from collections.abc import Iterator
from datetime import date
from pathlib import Path

import pytest

from insurance_rag.domain.models import (
    Jurisdiction,
    LegalEffect,
    NormRank,
    NormSpec,
    ProvisionKind,
)
from insurance_rag.embeddings.sparse import Bm25Encoder
from insurance_rag.index.qdrant_store import ChunkFilter, QdrantStore
from insurance_rag.ingestion.boe.client import IndexEntry, NormMetadata
from insurance_rag.ingestion.indexer import Indexer
from insurance_rag.ingestion.pipeline import RawStore
from insurance_rag.ingestion.state import IngestionState

pytestmark = pytest.mark.integration

FIXTURE = Path(__file__).parents[1] / "fixtures" / "boe_synthetic.xml"
SPEC = NormSpec(
    id="TEST-1",
    short_name="RT",
    title="Reglamento de prueba",
    rank=NormRank.REAL_DECRETO,
    jurisdiction=Jurisdiction.ES,
    legal_effect=LegalEffect.NATIONAL,
    topics=("prueba",),
)


class FakeEmbedder:
    model_name = "fake-hash-embedder"
    dimension = 8

    def __init__(self) -> None:
        self.calls = 0

    def _vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode()).digest()
        return [b / 255 for b in digest[: self.dimension]]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls += len(texts)
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


class FakeBoeClient:
    def __init__(self, xml: bytes) -> None:
        self.xml = xml
        self.updates = {"aprimero": date(2023, 1, 5)}
        self.text_downloads = 0

    def fetch_index(self, norm_id: str) -> list[IndexEntry]:
        ids = ["aprimero", "asegundo", "atercero", "daprimera", "an"]
        return [
            IndexEntry(block_id=b, title="", updated_at=self.updates.get(b, date(2020, 1, 2)))
            for b in ids
        ]

    def fetch_metadata(self, norm_id: str) -> NormMetadata:
        return NormMetadata(
            norm_id=norm_id,
            title="Reglamento de prueba",
            rank="Real Decreto",
            entry_into_force=date(2020, 1, 2),
            eli_url=None,
            consolidated_html_url="https://example.invalid",
            repealed=False,
        )

    def fetch_text_xml(self, norm_id: str) -> bytes:
        self.text_downloads += 1
        return self.xml


@pytest.fixture
def store() -> Iterator[QdrantStore]:
    store = QdrantStore(
        os.environ.get("QDRANT_URL", "http://localhost:6333"), f"test-{uuid.uuid4().hex[:8]}"
    )
    try:
        store.client.get_collections()
    except Exception:  # pragma: no cover - environment without Qdrant
        pytest.skip("Qdrant is not reachable")
    yield store
    store.client.delete_collection(store.collection)


def _indexer(
    tmp_path: Path, store: QdrantStore, client: FakeBoeClient, embedder: FakeEmbedder
) -> tuple[Indexer, IngestionState]:
    state = IngestionState(tmp_path / "state.db")
    indexer = Indexer(
        client=client,  # type: ignore[arg-type]
        raw_store=RawStore(tmp_path / "raw"),
        state=state,
        store=store,
        embedder=embedder,
        bm25=Bm25Encoder(),
        max_chars=3200,
    )
    indexer.prepare()
    return indexer, state


def test_sync_is_incremental(tmp_path: Path, store: QdrantStore) -> None:
    client = FakeBoeClient(FIXTURE.read_bytes())
    embedder = FakeEmbedder()
    indexer, state = _indexer(tmp_path, store, client, embedder)
    today = date(2026, 1, 1)

    [first] = indexer.sync([SPEC], as_of=today)
    total = store.count("TEST-1")
    assert first.status == "created"
    assert first.chunks_indexed == total > 0

    [second] = indexer.sync([SPEC], as_of=today)
    assert second.status == "skipped"
    assert client.text_downloads == 1

    # The BOE amends article 1: only that block is re-processed and re-embedded.
    client.xml = client.xml.replace(
        b"Texto modificado del art\xc3\xadculo primero.",
        b"Texto reformado del art\xc3\xadculo primero.",
    )
    client.updates["aprimero"] = date(2025, 6, 1)
    calls_before = embedder.calls
    [third] = indexer.sync([SPEC], as_of=today)

    assert third.status == "updated"
    assert third.blocks_changed == 1
    assert embedder.calls - calls_before == 1
    assert store.count("TEST-1") == total
    [article] = store.find_provision(kind=ProvisionKind.ARTICULO, number="1", norm_ids=["TEST-1"])
    assert "reformado" in article.text
    assert article.consolidated_as_of == date(2025, 6, 1)
    state.close()


def test_scheduled_change_triggers_reprocessing(tmp_path: Path, store: QdrantStore) -> None:
    client = FakeBoeClient(FIXTURE.read_bytes())
    indexer, state = _indexer(tmp_path, store, client, FakeEmbedder())

    indexer.sync([SPEC], as_of=date(2026, 1, 1))
    # The fixture has a wording of article 1 that comes into force on 2030-01-02.
    [before] = indexer.sync([SPEC], as_of=date(2030, 1, 1))
    [after] = indexer.sync([SPEC], as_of=date(2030, 1, 2))

    assert before.status == "skipped"
    assert after.status == "updated"
    assert after.blocks_changed == 1
    hits = store.search(
        dense=None,
        sparse=Bm25Encoder().encode_query("redacción futura"),
        mode="sparse",
        limit=1,
        chunk_filter=ChunkFilter(),
    )
    assert "Redacción futura" in hits[0].chunk.text
    state.close()
