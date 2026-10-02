from datetime import date

from insurance_rag.domain.models import (
    Jurisdiction,
    LegalEffect,
    Norm,
    NormRank,
    NormSpec,
    Paragraph,
    ParagraphKind,
    Provision,
    ProvisionKind,
)
from insurance_rag.ingestion.chunking import chunk_id, chunk_provision

SPEC = NormSpec(
    id="TEST-1",
    short_name="LT",
    title="Ley de prueba",
    rank=NormRank.LEY,
    jurisdiction=Jurisdiction.ES,
    legal_effect=LegalEffect.NATIONAL,
    topics=("prueba",),
)
NORM = Norm(
    spec=SPEC,
    entry_into_force=date(2020, 1, 2),
    consolidated_as_of=date(2024, 5, 1),
    source_url="https://example.invalid",
    reuse_terms="test",
)


def _text(text: str) -> Paragraph:
    return Paragraph(kind=ParagraphKind.TEXT, text=text)


def _provision(*paragraphs: Paragraph) -> Provision:
    return Provision(
        norm_id="TEST-1",
        block_id="a20",
        kind=ProvisionKind.ARTICULO,
        label="Artículo 20",
        number="20",
        heading="Mora del asegurador",
        hierarchy=("TÍTULO I. Disposiciones generales",),
        paragraphs=paragraphs,
        version_in_force_since=date(2020, 1, 2),
        block_updated_at=date(2023, 1, 1),
    )


def test_short_article_is_a_single_chunk_with_context() -> None:
    chunks = chunk_provision(NORM, _provision(_text("1. Uno."), _text("2. Dos.")))

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.apartado is None
    assert chunk.text == "1. Uno.\n2. Dos."
    assert chunk.url == "https://www.boe.es/buscar/act.php?id=TEST-1#a20"
    assert chunk.embed_text.splitlines()[:3] == [
        "LT — Ley de prueba",
        "TÍTULO I. Disposiciones generales",
        "Artículo 20. Mora del asegurador",
    ]
    assert chunk.consolidated_as_of == date(2024, 5, 1)
    assert chunk.citation_label == "LT, art. 20"


def test_long_article_is_split_by_apartado_ignoring_inner_lists() -> None:
    filler = "x" * 120
    provision = _provision(
        _text(f"Introducción {filler}"),
        _text(f"1. Primero {filler}"),
        _text(f"2. Segundo {filler}"),
        _text(f"1.º Regla interna {filler}"),
        _text(f"3. Tercero {filler}"),
    )

    chunks = chunk_provision(NORM, provision, max_chars=400)

    assert [c.apartado for c in chunks] == ["1", "2", "3"]
    assert chunks[0].text.startswith("Introducción")
    assert "Regla interna" in chunks[1].text
    assert chunks[1].citation_label == "LT, art. 20.2"
    assert "(apartado 2)" in chunks[1].embed_text


def test_oversized_apartado_is_windowed_and_ids_are_stable() -> None:
    paragraphs = [_text(f"1. Inicio {'y' * 100}")] + [_text("z" * 150) for _ in range(6)]

    first = chunk_provision(NORM, _provision(*paragraphs), max_chars=400)
    again = chunk_provision(NORM, _provision(*paragraphs), max_chars=400)

    assert len(first) > 1
    assert all(len(c.text) <= 400 for c in first)
    assert {c.apartado for c in first} == {"1"}
    assert [c.chunk_id for c in first] == [c.chunk_id for c in again]
    assert first[0].chunk_id == chunk_id("TEST-1", "a20", 0)


def test_long_tables_are_split_by_rows_repeating_the_header() -> None:
    rows = "\n".join(f"| fila {i} | {'v' * 40} |" for i in range(30))
    table = Paragraph(kind=ParagraphKind.TABLE, text=f"| Col | Valor |\n| --- | --- |\n{rows}")

    chunks = chunk_provision(NORM, _provision(table), max_chars=500)

    assert len(chunks) > 1
    assert all(c.text.startswith("| Col | Valor |\n| --- | --- |") for c in chunks)


def test_notes_are_labelled() -> None:
    note = Paragraph(kind=ParagraphKind.NOTE, text="Importes actualizados.")

    chunk = chunk_provision(NORM, _provision(_text("Texto."), note))[0]

    assert chunk.text.endswith("[Nota de la edición consolidada: Importes actualizados.]")
