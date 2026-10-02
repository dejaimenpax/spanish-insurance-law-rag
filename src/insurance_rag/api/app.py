"""FastAPI application factory."""

from fastapi import FastAPI
from pydantic import BaseModel

from insurance_rag import __version__
from insurance_rag.config import get_settings
from insurance_rag.observability.logging import configure_logging


class HealthResponse(BaseModel):
    status: str
    version: str


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level, json=settings.log_json)

    app = FastAPI(title="Spanish Insurance Law RAG", version=__version__)

    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    def health() -> HealthResponse:
        """Liveness probe: the process is up and serving requests."""
        return HealthResponse(status="ok", version=__version__)

    return app
