"""Build application components from settings."""

from insurance_rag.config import Settings
from insurance_rag.corpus.catalog import load_catalog
from insurance_rag.embeddings.sentence_transformer import SentenceTransformerEmbedder
from insurance_rag.embeddings.sparse import Bm25Encoder
from insurance_rag.index.qdrant_store import QdrantStore
from insurance_rag.ingestion.state import IngestionState
from insurance_rag.retrieval.references import ReferenceParser
from insurance_rag.retrieval.service import Reranker, Retriever


def build_store(settings: Settings) -> QdrantStore:
    return QdrantStore(settings.qdrant_url, settings.qdrant_collection)


def build_embedder(settings: Settings) -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder(
        settings.embedding_model,
        device=settings.embedding_device,
        query_prompt=settings.embedding_query_prompt,
    )


def build_bm25(settings: Settings) -> Bm25Encoder:
    bm25 = Bm25Encoder()
    if settings.state_path.exists():
        with IngestionState(settings.state_path) as state:
            avg = state.get_meta("bm25_avg_doc_len")
        if avg is not None:
            bm25.avg_doc_len = float(avg)
    return bm25


def build_reranker(settings: Settings) -> Reranker | None:
    if not settings.reranker_model:
        return None
    from insurance_rag.retrieval.rerank import CrossEncoderReranker

    return CrossEncoderReranker(settings.reranker_model, device=settings.embedding_device)


def build_retriever(settings: Settings) -> Retriever:
    return Retriever(
        store=build_store(settings),
        embedder=build_embedder(settings),
        bm25=build_bm25(settings),
        references=ReferenceParser(load_catalog().norms),
        reranker=build_reranker(settings),
    )
