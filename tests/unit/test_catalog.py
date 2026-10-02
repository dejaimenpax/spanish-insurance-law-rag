from insurance_rag.corpus.catalog import load_catalog


def test_catalog_lists_the_ten_phase_one_norms() -> None:
    catalog = load_catalog()

    ids = [n.id for n in catalog.norms]
    assert len(ids) == 10
    assert len(set(ids)) == len(ids)
    assert len({n.short_name for n in catalog.norms}) == len(ids)


def test_partial_norms_document_their_scope() -> None:
    distribution = load_catalog().get("BOE-A-2020-1651")

    assert distribution.scope is not None
    assert distribution.scope.from_block == "ls"
    assert distribution.scope.until_block == "ti-6"
    assert "dt-4" in distribution.scope.extra_blocks
    assert distribution.scope_rationale
