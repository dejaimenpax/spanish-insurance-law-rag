import pytest

from insurance_rag.corpus.catalog import load_catalog
from insurance_rag.domain.models import ProvisionKind
from insurance_rag.retrieval.references import ProvisionRef, ReferenceParser

LCS = "BOE-A-1980-22501"
LOSSEAR = "BOE-A-2015-7897"
ROSSEAR = "BOE-A-2015-13057"
RDL_3_2020 = "BOE-A-2020-1651"


@pytest.fixture(scope="module")
def parser() -> ReferenceParser:
    return ReferenceParser(load_catalog().norms)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "¿Qué dice el artículo 10 de la Ley 50/1980?",
            ProvisionRef(ProvisionKind.ARTICULO, "10", None, LCS),
        ),
        (
            "art. 20.3 de la LCS",
            ProvisionRef(ProvisionKind.ARTICULO, "20", "3", LCS),
        ),
        (
            "lo dispuesto en el artículo 76 bis de la Ley 20/2015",
            ProvisionRef(ProvisionKind.ARTICULO, "76 bis", None, LOSSEAR),
        ),
        (
            "el artículo 3 del Real Decreto 1060/2015",
            ProvisionRef(ProvisionKind.ARTICULO, "3", None, ROSSEAR),
        ),
        (
            "la disposición transitoria cuarta del RDL 3/2020",
            ProvisionRef(ProvisionKind.DISPOSICION_TRANSITORIA, "4", None, RDL_3_2020),
        ),
        (
            "según el artículo 12 de esta Ley",
            ProvisionRef(ProvisionKind.ARTICULO, "12", None, None),
        ),
    ],
)
def test_parses_single_references(
    parser: ReferenceParser, text: str, expected: ProvisionRef
) -> None:
    assert parser.parse(text) == [expected]


def test_each_mention_gets_its_own_norm(parser: ReferenceParser) -> None:
    refs = parser.parse("el artículo 3 de la LCS y el artículo 15 de la Ley 20/2015")

    assert [(r.number, r.norm_id) for r in refs] == [("3", LCS), ("15", LOSSEAR)]


def test_does_not_confuse_norm_numbers_with_articles(parser: ReferenceParser) -> None:
    assert parser.parse("¿Qué regula la Ley 50/1980?") == []


def test_lists_mentioned_norms(parser: ReferenceParser) -> None:
    assert parser.mentioned_norms("diferencias entre la LOSSEAR y el ROSSEAR") == [
        LOSSEAR,
        ROSSEAR,
    ]
