from datetime import date

from insurance_rag.domain.models import ParagraphKind
from insurance_rag.ingestion.boe.parser import parse_blocks, parse_version


def _block(xml: bytes, block_id: str):  # type: ignore[no-untyped-def]
    return next(b for b in parse_blocks(xml) if b.block_id == block_id)


def test_selects_latest_version_in_force(synthetic_xml: bytes) -> None:
    block = _block(synthetic_xml, "aprimero")

    assert block.version_as_of(date(2021, 1, 1)).in_force_since == date(2020, 1, 2)
    assert block.version_as_of(date(2026, 1, 1)).in_force_since == date(2023, 1, 2)
    assert block.version_as_of(date(2019, 1, 1)) is None


def test_future_versions_are_ignored(synthetic_xml: bytes) -> None:
    version = _block(synthetic_xml, "aprimero").version_as_of(date(2026, 1, 1))
    assert version is not None

    parsed = parse_version(version.element)

    assert [p.text for p in parsed.paragraphs if p.kind is ParagraphKind.TEXT] == [
        "Texto modificado del artículo primero."
    ]


def test_expired_blocks_have_no_version_in_force(synthetic_xml: bytes) -> None:
    block = _block(synthetic_xml, "acuarto")

    assert block.version_as_of(date(2021, 6, 1)) is not None
    assert block.version_as_of(date(2022, 1, 1)) is None


def test_editorial_notes_and_amendment_notes_are_separated(synthetic_xml: bytes) -> None:
    version = _block(synthetic_xml, "aprimero").version_as_of(date(2026, 1, 1))
    assert version is not None

    parsed = parse_version(version.element)

    assert parsed.label_line == "Artículo primero. Objeto."
    notes = [p.text for p in parsed.paragraphs if p.kind is ParagraphKind.NOTE]
    assert notes == ["Téngase en cuenta la actualización de importes."]
    assert parsed.amendment_notes == ("Se modifica por la Ley de prueba 2/2023.",)


def test_text_tables_become_markdown_and_numeric_tables_become_stubs(
    synthetic_xml: bytes,
) -> None:
    version = _block(synthetic_xml, "an").version_as_of(date(2026, 1, 1))
    assert version is not None

    tables = [p for p in parse_version(version.element).paragraphs if p.kind is ParagraphKind.TABLE]

    assert len(tables) == 3
    markdown, stub_c1, stub_c2 = (t.text for t in tables)
    assert markdown.startswith("Tabla 1.A Importes\n| Concepto | Importe |")
    assert "| Gastos de traslado | 500 € |" in markdown
    # The continuation is merged into the 1.C.1 stub; 1.C.2 starts a new one.
    assert "TABLA 1.C.1 Coeficientes" in stub_c1
    assert "9 filas" in stub_c1
    assert "TABLA 1.C.2 Otros coeficientes" in stub_c2
    assert "8 filas" in stub_c2
    assert "1,1" not in stub_c1
