"""Checks against the live BOE API. Run with ``pytest -m network``."""

from datetime import date

import pytest

from insurance_rag.corpus.catalog import load_catalog
from insurance_rag.ingestion.boe.client import BoeClient
from insurance_rag.ingestion.boe.parser import parse_blocks
from insurance_rag.ingestion.structure import build_provisions

pytestmark = pytest.mark.network


@pytest.fixture(scope="module")
def client() -> BoeClient:
    return BoeClient()


def test_full_text_and_index_list_the_same_blocks(client: BoeClient) -> None:
    norm_id = "BOE-A-1980-22501"

    index_ids = [e.block_id for e in client.fetch_index(norm_id)]
    text_ids = [b.block_id for b in parse_blocks(client.fetch_text_xml(norm_id))]

    assert index_ids == text_ids


def test_distribution_scope_resolves_against_the_live_text(client: BoeClient) -> None:
    spec = load_catalog().get("BOE-A-2020-1651")
    blocks = parse_blocks(client.fetch_text_xml(spec.id))

    provisions = build_provisions(spec, blocks, block_updates={}, as_of=date.today())

    numbers = {p.number for p in provisions if p.kind.value == "articulo"}
    assert {"127", "211"} <= numbers
    assert "212" not in numbers
    assert "1" not in numbers
