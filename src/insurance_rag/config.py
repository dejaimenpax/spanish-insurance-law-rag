"""Application settings, loaded from environment variables and an optional .env file."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    anthropic_api_key: SecretStr | None = None

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_json: bool = True

    data_dir: Path = Path("data")

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "chunks"

    embedding_model: str = "BAAI/bge-m3"
    embedding_device: str | None = None
    """cpu, mps or cuda; autodetected when unset."""
    embedding_query_prompt: str = ""
    chunk_max_chars: int = 3200

    reranker_model: str | None = "BAAI/bge-reranker-v2-m3"
    """Cross-encoder that reranks the fused candidates; set to an empty value to disable."""
    reranker_candidates: int = 10
    reranker_max_length: int = 512
    retrieval_candidates: int = 50
    """Candidates fetched from each of the dense and sparse searches before fusion."""

    llm_model: str = "claude-opus-5-5"
    llm_effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    llm_max_tokens: int = 16000
    answer_top_k: int = 8
    """Retrieved chunks passed to the model."""
    abstain_below: float | None = None
    """Best reranker score under which the assistant abstains without calling the model."""

    query_log: bool = True
    """Append one JSON line per answered question to data/queries.jsonl."""

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def state_path(self) -> Path:
        return self.data_dir / "state.db"


@lru_cache
def get_settings() -> Settings:
    return Settings()
