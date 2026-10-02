"""Cross-encoder reranking (e.g. BAAI/bge-reranker-v2-m3)."""

from collections.abc import Sequence

from insurance_rag.domain.models import Chunk
from insurance_rag.embeddings.sentence_transformer import best_device


class CrossEncoderReranker:
    def __init__(self, model_name: str, *, device: str | None = None, batch_size: int = 16) -> None:
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name, device=device or best_device(), max_length=1024)
        self._batch_size = batch_size

    def rerank(self, query: str, chunks: Sequence[Chunk]) -> list[float]:
        pairs = [(query, _passage(c)) for c in chunks]
        scores = self._model.predict(pairs, batch_size=self._batch_size, convert_to_numpy=True)
        return [float(s) for s in scores]


def _passage(chunk: Chunk) -> str:
    title = f"{chunk.norm_short_name}, {chunk.label}" + (
        f". {chunk.heading}" if chunk.heading else ""
    )
    return f"{title}\n{chunk.text}"
