import pytest

from insurance_rag.ingestion.spanish_numbers import (
    parse_cardinal,
    parse_ordinal,
    parse_provision_number,
)


@pytest.mark.parametrize(
    ("words", "expected"),
    [
        ("dieciocho", 18),
        ("dieciséis", 16),
        ("veintidós", 22),
        ("treinta y uno", 31),
        ("ciento", 100),
        ("ciento veintiuno", 121),
        ("cien", 100),
        ("setenta y seis", 76),
        ("perro", None),
    ],
)
def test_parse_cardinal(words: str, expected: int | None) -> None:
    assert parse_cardinal(words) == expected


@pytest.mark.parametrize(
    ("words", "expected"),
    [
        ("primera", 1),
        ("tercero", 3),
        ("séptima", 7),
        ("décima", 10),
        ("undécima", 11),
        ("duodécimo", 12),
        ("decimotercera", 13),
        ("decimoctava", 18),
        ("vigésima", 20),
        ("vigésima primera", 21),
        ("vigesimoprimera", 21),
        ("única", None),
    ],
)
def test_parse_ordinal(words: str, expected: int | None) -> None:
    assert parse_ordinal(words) == expected


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("Artículo dieciocho", "18"),
        ("Artículo ciento veintiuno", "121"),
        ("Artículo sexto bis", "6 bis"),
        ("Artículo primero", "1"),
        ("Artículo 10", "10"),
        ("Artículo 10 bis", "10 bis"),
        ("Artículo setenta y seis. a)", "76 a"),
        ("Artículo treinta y tres.a)", "33 a"),
        ("Artículo\xa0127", "127"),
        ("Artículo 3.º", "3"),
        ("Artículo único", "único"),
        ("Disposición adicional decimotercera", "13"),
        ("Disposición transitoria cuarta", "4"),
        ("Disposición final vigésima primera", "21"),
        ("Disposición derogatoria única", "única"),
        ("Disposición adicional", None),
    ],
)
def test_parse_provision_number(title: str, expected: str | None) -> None:
    assert parse_provision_number(title) == expected
