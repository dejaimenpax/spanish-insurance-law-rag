"""Local dense embeddings with sentence-transformers (default model: BAAI/bge-m3)."""

import structlog

log = structlog.get_logger(__name__)


def best_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class SentenceTransformerEmbedder:
    """Wraps a sentence-transformers model; vectors are L2-normalised for cosine search.

    ``query_prompt`` is prepended to queries for models trained with instructions (e.g. the
    Qwen3 embedding family); bge-m3 needs none.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        *,
        device: str | None = None,
        batch_size: int = 16,
        max_seq_length: int = 1024,
        query_prompt: str = "",
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self._model_name = model_name
        self._batch_size = batch_size
        self._query_prompt = query_prompt
        resolved_device = device or best_device()
        log.info("embedder.load", model=model_name, device=resolved_device)
        self._model: SentenceTransformer = SentenceTransformer(model_name, device=resolved_device)
        self._model.max_seq_length = max_seq_length
        dimension = self._model.get_embedding_dimension()
        if dimension is None:
            raise ValueError(f"{model_name} does not report an embedding dimension")
        self._dimension = int(dimension)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 64,
            convert_to_numpy=True,
        )
        return [list(map(float, v)) for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        vector = self._model.encode(
            self._query_prompt + text, normalize_embeddings=True, convert_to_numpy=True
        )
        return list(map(float, vector))
