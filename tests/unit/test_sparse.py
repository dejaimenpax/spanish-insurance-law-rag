from insurance_rag.embeddings.sparse import Bm25Encoder, token_id


def test_tokenizer_stems_drops_stopwords_and_keeps_references() -> None:
    tokens = Bm25Encoder().tokenize("El artículo 10 de la Ley 20/2015 regula las indemnizaciones")

    assert "10" in tokens
    assert "20/2015" in tokens
    assert "de" not in tokens
    assert "la" not in tokens
    assert "indemniz" in tokens


def test_document_weights_saturate_with_term_frequency() -> None:
    encoder = Bm25Encoder(avg_doc_len=10)
    once = encoder.encode_document("prima")
    many = encoder.encode_document("prima prima prima prima prima")

    weight_once = dict(zip(once.indices, once.values, strict=True))[token_id("prim")]
    weight_many = dict(zip(many.indices, many.values, strict=True))[token_id("prim")]
    assert weight_once < weight_many < encoder.k1 + 1


def test_query_vector_has_unit_weights_and_sorted_indices() -> None:
    vector = Bm25Encoder().encode_query("prima del seguro prima")

    assert vector.values == [1.0] * len(vector.values)
    assert vector.indices == sorted(vector.indices)
    assert len(vector.indices) == 2
