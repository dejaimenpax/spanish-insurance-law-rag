"""Local SQLite state that makes re-ingestion incremental.

It records, per norm, a fingerprint of the BOE index (block ids and update dates) and the next
date on which an already published change comes into force; per block, a hash of the chunks it
produced; and a cache of dense embeddings keyed by the embedded text.
"""

import hashlib
import sqlite3
from array import array
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import Self

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS norms (
    norm_id TEXT PRIMARY KEY,
    index_fingerprint TEXT NOT NULL,
    consolidated_as_of TEXT NOT NULL,
    next_change TEXT,
    synced_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS blocks (
    norm_id TEXT NOT NULL,
    block_id TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    PRIMARY KEY (norm_id, block_id)
);
CREATE TABLE IF NOT EXISTS embeddings (
    text_hash TEXT NOT NULL,
    model TEXT NOT NULL,
    vector BLOB NOT NULL,
    PRIMARY KEY (text_hash, model)
);
"""


@dataclass(frozen=True)
class NormState:
    index_fingerprint: str
    consolidated_as_of: date
    next_change: date | None


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class IngestionState:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path)
        self._db.executescript(_SCHEMA)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self._db.close()

    # --- meta ---------------------------------------------------------------------------------
    def get_meta(self, key: str) -> str | None:
        row = self._db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return None if row is None else str(row[0])

    def set_meta(self, key: str, value: str) -> None:
        with self._db:
            self._db.execute(
                "INSERT INTO meta (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )

    def reset_index_state(self) -> None:
        """Forget every synced norm and block (the embedding cache is kept)."""
        with self._db:
            self._db.execute("DELETE FROM norms")
            self._db.execute("DELETE FROM blocks")

    # --- norms and blocks ---------------------------------------------------------------------
    def get_norm(self, norm_id: str) -> NormState | None:
        row = self._db.execute(
            "SELECT index_fingerprint, consolidated_as_of, next_change FROM norms "
            "WHERE norm_id = ?",
            (norm_id,),
        ).fetchone()
        if row is None:
            return None
        return NormState(
            index_fingerprint=row[0],
            consolidated_as_of=date.fromisoformat(row[1]),
            next_change=date.fromisoformat(row[2]) if row[2] else None,
        )

    def block_hashes(self, norm_id: str) -> dict[str, str]:
        rows = self._db.execute(
            "SELECT block_id, content_hash FROM blocks WHERE norm_id = ?", (norm_id,)
        ).fetchall()
        return {str(block_id): str(content_hash) for block_id, content_hash in rows}

    def save_norm(self, norm_id: str, state: NormState, block_hashes: Mapping[str, str]) -> None:
        with self._db:
            self._db.execute(
                "INSERT INTO norms (norm_id, index_fingerprint, consolidated_as_of, next_change, "
                "synced_at) VALUES (?, ?, ?, ?, ?) ON CONFLICT(norm_id) DO UPDATE SET "
                "index_fingerprint = excluded.index_fingerprint, "
                "consolidated_as_of = excluded.consolidated_as_of, "
                "next_change = excluded.next_change, synced_at = excluded.synced_at",
                (
                    norm_id,
                    state.index_fingerprint,
                    state.consolidated_as_of.isoformat(),
                    state.next_change.isoformat() if state.next_change else None,
                    datetime.now(timezone.utc).isoformat(),  # noqa: UP017 - explicit for readers
                ),
            )
            self._db.execute("DELETE FROM blocks WHERE norm_id = ?", (norm_id,))
            self._db.executemany(
                "INSERT INTO blocks (norm_id, block_id, content_hash) VALUES (?, ?, ?)",
                [(norm_id, block_id, h) for block_id, h in block_hashes.items()],
            )

    def forget_norms_except(self, norm_ids: Iterable[str]) -> list[str]:
        keep = set(norm_ids)
        stored = [str(r[0]) for r in self._db.execute("SELECT norm_id FROM norms").fetchall()]
        removed = [n for n in stored if n not in keep]
        with self._db:
            for norm_id in removed:
                self._db.execute("DELETE FROM norms WHERE norm_id = ?", (norm_id,))
                self._db.execute("DELETE FROM blocks WHERE norm_id = ?", (norm_id,))
        return removed

    # --- embedding cache ----------------------------------------------------------------------
    def cached_embeddings(self, model: str, hashes: list[str]) -> dict[str, list[float]]:
        found: dict[str, list[float]] = {}
        for start in range(0, len(hashes), 500):
            batch = hashes[start : start + 500]
            placeholders = ",".join("?" * len(batch))
            rows = self._db.execute(
                f"SELECT text_hash, vector FROM embeddings WHERE model = ? "  # noqa: S608
                f"AND text_hash IN ({placeholders})",
                (model, *batch),
            ).fetchall()
            for h, blob in rows:
                found[str(h)] = array("f", blob).tolist()
        return found

    def cache_embeddings(self, model: str, vectors: Mapping[str, list[float]]) -> None:
        with self._db:
            self._db.executemany(
                "INSERT OR REPLACE INTO embeddings (text_hash, model, vector) VALUES (?, ?, ?)",
                [(h, model, array("f", v).tobytes()) for h, v in vectors.items()],
            )
