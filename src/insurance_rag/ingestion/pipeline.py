"""Ingestion pipeline: download (with a raw cache), parse, structure and chunk each norm."""

import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import structlog

from insurance_rag.domain.models import Chunk, Norm, NormSpec, Provision
from insurance_rag.ingestion.boe.client import BoeClient, IndexEntry, NormMetadata
from insurance_rag.ingestion.boe.parser import parse_blocks
from insurance_rag.ingestion.chunking import DEFAULT_MAX_CHARS, chunk_provision
from insurance_rag.ingestion.structure import build_provisions

log = structlog.get_logger(__name__)

BOE_REUSE_TERMS = (
    "Fuente: Agencia Estatal Boletín Oficial del Estado (https://www.boe.es). Texto consolidado "
    "sin valor jurídico oficial, reutilizado conforme al aviso legal del BOE."
)


class RawStore:
    """Raw API responses on disk, so parsing can be re-run without hitting the API."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def _dir(self, norm_id: str) -> Path:
        path = self.root / norm_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def text_path(self, norm_id: str) -> Path:
        return self._dir(norm_id) / "texto.xml"

    def index_path(self, norm_id: str) -> Path:
        return self._dir(norm_id) / "indice.json"

    def metadata_path(self, norm_id: str) -> Path:
        return self._dir(norm_id) / "metadatos.json"


@dataclass(frozen=True)
class NormSnapshot:
    """Everything fetched from the BOE for one norm."""

    metadata: NormMetadata
    index: tuple[IndexEntry, ...]
    text_xml: bytes


def fetch_snapshot(
    client: BoeClient, store: RawStore, norm_id: str, *, refresh: bool = False
) -> NormSnapshot:
    paths = (store.text_path(norm_id), store.index_path(norm_id), store.metadata_path(norm_id))
    if refresh or not all(p.exists() for p in paths):
        log.info("ingest.download", norm=norm_id)
        metadata = client.fetch_metadata(norm_id)
        index = client.fetch_index(norm_id)
        text_xml = client.fetch_text_xml(norm_id)
        store.metadata_path(norm_id).write_text(metadata.model_dump_json(indent=2), "utf-8")
        store.index_path(norm_id).write_text(
            json.dumps([e.model_dump(mode="json") for e in index], ensure_ascii=False, indent=1),
            "utf-8",
        )
        store.text_path(norm_id).write_bytes(text_xml)
    return NormSnapshot(
        metadata=NormMetadata.model_validate_json(store.metadata_path(norm_id).read_text("utf-8")),
        index=tuple(
            IndexEntry.model_validate(e)
            for e in json.loads(store.index_path(norm_id).read_text("utf-8"))
        ),
        text_xml=store.text_path(norm_id).read_bytes(),
    )


@dataclass
class NormResult:
    norm: Norm
    provisions: list[Provision]
    chunks: list[Chunk]
    stats: Counter[str] = field(default_factory=Counter)


def process_norm(
    spec: NormSpec,
    snapshot: NormSnapshot,
    *,
    as_of: date,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> NormResult:
    blocks = parse_blocks(snapshot.text_xml)
    updates = {e.block_id: e.updated_at for e in snapshot.index}
    provisions = build_provisions(spec, blocks, block_updates=updates, as_of=as_of)
    if not provisions:
        raise ValueError(f"{spec.id}: no provisions in scope")
    norm = Norm(
        spec=spec,
        entry_into_force=snapshot.metadata.entry_into_force,
        consolidated_as_of=max(p.block_updated_at for p in provisions),
        source_url=snapshot.metadata.consolidated_html_url,
        eli_url=snapshot.metadata.eli_url,
        reuse_terms=BOE_REUSE_TERMS,
    )
    chunks = [c for p in provisions for c in chunk_provision(norm, p, max_chars=max_chars)]
    stats: Counter[str] = Counter(p.kind.value for p in provisions)
    stats["repealed"] = sum(p.repealed for p in provisions)
    stats["chunks"] = len(chunks)
    stats["max_chunk_chars"] = max(len(c.text) for c in chunks)
    return NormResult(norm, provisions, chunks, stats)


def write_chunks(result: NormResult, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{result.norm.spec.id}.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for chunk in result.chunks:
            fh.write(chunk.model_dump_json() + "\n")
    (out_dir / f"{result.norm.spec.id}.norm.json").write_text(
        result.norm.model_dump_json(indent=2), "utf-8"
    )
    return path
