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

    reranker_model: str | None = None
    """Cross-encoder used to rerank candidates, e.g. BAAI/bge-reranker-v2-m3; off when unset."""

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def state_path(self) -> Path:
        return self.data_dir / "state.db"


@lru_cache
def get_settings() -> Settings:
    return Settings()
