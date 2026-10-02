from datetime import date
from pathlib import Path

from insurance_rag.ingestion.state import IngestionState, NormState


def test_round_trips_norm_and_block_state(tmp_path: Path) -> None:
    with IngestionState(tmp_path / "state.db") as state:
        state.save_norm("N1", NormState("fp1", date(2024, 1, 1), date(2027, 1, 30)), {"a1": "h1"})
        state.save_norm("N1", NormState("fp2", date(2025, 1, 1), None), {"a1": "h2", "a2": "h3"})

        assert state.get_norm("N1") == NormState("fp2", date(2025, 1, 1), None)
        assert state.block_hashes("N1") == {"a1": "h2", "a2": "h3"}
        assert state.get_norm("missing") is None


def test_forgets_norms_removed_from_the_catalog(tmp_path: Path) -> None:
    with IngestionState(tmp_path / "state.db") as state:
        state.save_norm("N1", NormState("fp", date(2024, 1, 1), None), {"a": "h"})
        state.save_norm("N2", NormState("fp", date(2024, 1, 1), None), {"a": "h"})

        assert state.forget_norms_except(["N1"]) == ["N2"]
        assert state.get_norm("N2") is None
        assert state.block_hashes("N2") == {}


def test_caches_embeddings_by_model(tmp_path: Path) -> None:
    with IngestionState(tmp_path / "state.db") as state:
        state.cache_embeddings("m1", {"h1": [0.5, -0.25]})

        assert state.cached_embeddings("m1", ["h1", "h2"]) == {"h1": [0.5, -0.25]}
        assert state.cached_embeddings("m2", ["h1"]) == {}
