from datetime import date
from pathlib import Path

import pytest

from insurance_rag.corpus.catalog import load_catalog
from insurance_rag.domain.models import (
    Chunk,
    Jurisdiction,
    LegalEffect,
    NormRank,
    ProvisionKind,
)
from insurance_rag.evaluation.dataset import ProvisionKey, load_dataset, parse_ref
from insurance_rag.evaluation.retrieval_metrics import score_retrieval

LCS = "BOE-A-1980-22501"


def _chunk(number: str, kind: ProvisionKind = ProvisionKind.ARTICULO) -> Chunk:
    return Chunk(
        chunk_id=f"c-{number}",
        norm_id=LCS,
        norm_short_name="LCS",
        block_id=f"a{number}",
        kind=kind,
        label=f"Artículo {number}",
        number=number,
        heading=None,
        apartado=None,
        hierarchy=(),
        text="",
        embed_text="",
        url="",
        version_in_force_since=date(2020, 1, 1),
        block_updated_at=date(2020, 1, 1),
        consolidated_as_of=date(2020, 1, 1),
        jurisdiction=Jurisdiction.ES,
        legal_effect=LegalEffect.NATIONAL,
        rank=NormRank.LEY,
        topics=(),
    )


def test_parses_gold_references() -> None:
    catalog = load_catalog()

    assert parse_ref("LCS art. 76 a", catalog) == ProvisionKey(
        norm_id=LCS, kind=ProvisionKind.ARTICULO, number="76 a"
    )
    assert parse_ref("RDL 3/2020 DT 4", catalog).kind is ProvisionKind.DISPOSICION_TRANSITORIA
    assert parse_ref("RDL 3/2020 anexo ax-3", catalog).block_id == "ax-3"
    with pytest.raises(ValueError, match="Unknown norm"):
        parse_ref("XYZ art. 1", catalog)


def test_recall_and_reciprocal_rank() -> None:
    gold = [
        ProvisionKey(norm_id=LCS, kind=ProvisionKind.ARTICULO, number="20"),
        ProvisionKey(norm_id=LCS, kind=ProvisionKind.ARTICULO, number="9"),
    ]
    ranked = [_chunk("1"), _chunk("20"), _chunk("3"), _chunk("9")]

    scores = score_retrieval(ranked, gold, ks=(1, 3, 5))

    assert scores.reciprocal_rank == 0.5
    assert scores.recall_at == {1: 0.0, 3: 0.5, 5: 1.0}


def test_repository_dataset_is_well_formed() -> None:
    dataset = load_dataset(Path(__file__).parents[2] / "eval" / "questions.yaml", load_catalog())

    assert len(dataset.questions) >= 150
    covered = {g.norm_id for q in dataset.questions for g in q.gold}
    assert covered == {n.id for n in load_catalog().norms}
