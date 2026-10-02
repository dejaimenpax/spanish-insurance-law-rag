from datetime import date

import pytest

from insurance_rag.domain.models import (
    Jurisdiction,
    LegalEffect,
    NormRank,
    NormSpec,
    ProvisionKind,
    Scope,
)
from insurance_rag.ingestion.boe.parser import parse_blocks
from insurance_rag.ingestion.structure import build_provisions, select_in_scope

AS_OF = date(2026, 1, 1)


def _spec(scope: Scope | None = None) -> NormSpec:
    return NormSpec(
        id="TEST-1",
        short_name="RT",
        title="Reglamento de prueba",
        rank=NormRank.REAL_DECRETO,
        jurisdiction=Jurisdiction.ES,
        legal_effect=LegalEffect.NATIONAL,
        topics=("prueba",),
        scope=scope,
    )


def test_skips_the_approving_instrument(synthetic_xml: bytes) -> None:
    provisions = build_provisions(
        _spec(), parse_blocks(synthetic_xml), block_updates={}, as_of=AS_OF
    )

    labels = [p.label for p in provisions]
    assert "Artículo único" not in labels
    assert labels[0] == "Artículo primero"


def test_builds_hierarchy_numbers_and_rubrics(synthetic_xml: bytes) -> None:
    provisions = {
        p.block_id: p
        for p in build_provisions(
            _spec(),
            parse_blocks(synthetic_xml),
            block_updates={"aprimero": date(2023, 1, 5)},
            as_of=AS_OF,
        )
    }

    first = provisions["aprimero"]
    assert first.number == "1"
    assert first.heading == "Objeto"
    assert first.hierarchy == ("TÍTULO I. Disposiciones generales", "CAPÍTULO I. Objeto")
    assert first.block_updated_at == date(2023, 1, 5)
    assert first.version_in_force_since == date(2023, 1, 2)

    # A section heading typed as "precepto" in the source still nests correctly.
    second = provisions["asegundo"]
    assert second.number == "2 bis"
    assert second.hierarchy[-1] == "Sección primera. Definiciones"

    additional = provisions["daprimera"]
    assert additional.kind is ProvisionKind.DISPOSICION_ADICIONAL
    assert additional.number == "13"
    assert additional.hierarchy == ()

    annex = provisions["an"]
    assert annex.kind is ProvisionKind.ANEXO
    assert annex.heading == "Tablas de prueba"


def test_marks_repealed_and_drops_expired_blocks(synthetic_xml: bytes) -> None:
    provisions = {
        p.block_id: p
        for p in build_provisions(
            _spec(), parse_blocks(synthetic_xml), block_updates={}, as_of=AS_OF
        )
    }

    assert provisions["atercero"].repealed
    assert not provisions["aprimero"].repealed
    assert "acuarto" not in provisions


def test_scope_keeps_the_range_and_extra_blocks(synthetic_xml: bytes) -> None:
    scope = Scope(from_block="ci", until_block="atercero", extra_blocks=("an",))

    selected = [b.block_id for b in select_in_scope(_spec(scope), parse_blocks(synthetic_xml))]

    assert selected == ["ci", "aprimero", "sprimera", "asegundo", "an"]


def test_scope_with_unknown_block_fails_loudly(synthetic_xml: bytes) -> None:
    scope = Scope(from_block="ci", until_block="nope")

    with pytest.raises(ValueError, match="nope"):
        select_in_scope(_spec(scope), parse_blocks(synthetic_xml))
