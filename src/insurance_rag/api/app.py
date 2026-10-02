"""FastAPI application: question answering over the indexed norms, with SSE streaming."""

import json
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Annotated

import structlog
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from insurance_rag import __version__
from insurance_rag.api.schemas import (
    AskRequest,
    AskResponse,
    HealthResponse,
    NormInfo,
    ReadyResponse,
)
from insurance_rag.config import Settings, get_settings
from insurance_rag.corpus.catalog import load_catalog
from insurance_rag.generation.answer import AnswerEvent, AnswerService, DoneEvent
from insurance_rag.index.qdrant_store import ChunkFilter, QdrantStore
from insurance_rag.ingestion.state import IngestionState
from insurance_rag.observability.logging import configure_logging

log = structlog.get_logger(__name__)
STATIC_DIR = Path(__file__).parent / "static"


@dataclass
class AppState:
    settings: Settings
    store: QdrantStore
    answers: AnswerService | None


def _services(request: Request) -> AppState:
    services: AppState = request.app.state.services
    return services


Services = Annotated[AppState, Depends(_services)]


def create_app(*, state: AppState | None = None) -> FastAPI:
    """Build the app. Tests pass a prepared ``state``; otherwise components load at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if state is not None:
            app.state.services = state
        else:
            from insurance_rag.wiring import build_answer_service, build_retriever, build_store

            settings = get_settings()
            configure_logging(settings.log_level, json=settings.log_json)
            answers = None
            if settings.anthropic_api_key is not None:
                answers = build_answer_service(settings, build_retriever(settings))
            else:
                log.warning("api.no_api_key", detail="question answering is disabled")
            app.state.services = AppState(settings, build_store(settings), answers)
        yield

    app = FastAPI(
        title="Spanish Insurance Law RAG",
        version=__version__,
        description="Questions and answers over Spanish insurance legislation, with citations. "
        "Not legal advice.",
        lifespan=lifespan,
    )
    app.middleware("http")(_request_context)

    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    def health() -> HealthResponse:
        """Liveness probe: the process is up and serving requests."""
        return HealthResponse(status="ok", version=__version__)

    @app.get("/ready", response_model=ReadyResponse, tags=["ops"])
    def ready(response: Response, services: Services) -> ReadyResponse:
        """Readiness probe: the index is reachable and populated and answering is configured."""
        index_ready = services.store.ready()
        checks = {
            "index": index_ready,
            "index_populated": index_ready and services.store.count() > 0,
            "answering": services.answers is not None,
        }
        is_ready = all(checks.values())
        if not is_ready:
            response.status_code = 503
        return ReadyResponse(ready=is_ready, checks=checks)

    @app.get("/norms", response_model=list[NormInfo], tags=["corpus"])
    def norms(services: Services) -> list[NormInfo]:
        """Indexed norms with the consolidation date of the text used."""
        settings = services.settings
        consolidated: dict[str, date | None] = {}
        if settings.state_path.exists():
            with IngestionState(settings.state_path) as ingestion:
                for spec in load_catalog().norms:
                    norm_state = ingestion.get_norm(spec.id)
                    consolidated[spec.id] = norm_state.consolidated_as_of if norm_state else None
        index_ready = services.store.ready()
        return [
            NormInfo(
                id=spec.id,
                short_name=spec.short_name,
                title=spec.title,
                jurisdiction=spec.jurisdiction.value,
                legal_effect=spec.legal_effect.value,
                topics=list(spec.topics),
                source_url=f"https://www.boe.es/buscar/act.php?id={spec.id}",
                consolidated_as_of=consolidated.get(spec.id),
                application_date=spec.application_date,
                indexed_chunks=services.store.count(spec.id) if index_ready else 0,
            )
            for spec in load_catalog().norms
        ]

    @app.post("/ask", response_model=AskResponse, tags=["qa"])
    async def ask(request: AskRequest, services: Services) -> AskResponse:
        """Answer a question and return the full result at once."""
        answers = _answers(services)
        sources, warnings, done = await answers.answer(
            request.question, chunk_filter=_filter(request)
        )
        _log_query(services.settings, request, done)
        return AskResponse(
            answer=done.answer,
            abstained=done.abstained,
            sources=sources,
            citations=done.citations,
            cited_sources=done.cited_sources,
            warnings=warnings,
            model=done.model,
            usage=done.usage,
            cost_usd=done.cost_usd,
            timings_ms=done.timings_ms,
        )

    @app.post("/ask/stream", tags=["qa"])
    async def ask_stream(request: AskRequest, services: Services) -> StreamingResponse:
        """Answer a question as Server-Sent Events: sources, delta, citation, warning, done."""
        answers = _answers(services)

        async def events() -> AsyncIterator[str]:
            try:
                async for event in answers.stream(request.question, chunk_filter=_filter(request)):
                    if isinstance(event, DoneEvent):
                        _log_query(services.settings, request, event)
                    yield _sse(event)
            except Exception:
                log.exception("ask_stream.failed")
                yield 'event: error\ndata: {"message": "Error al generar la respuesta."}\n\n'

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

        @app.get("/", include_in_schema=False)
        def index() -> FileResponse:
            return FileResponse(STATIC_DIR / "index.html")

    return app


def _answers(services: AppState) -> AnswerService:
    if services.answers is None:
        raise HTTPException(503, "Question answering is not configured: set ANTHROPIC_API_KEY.")
    return services.answers


def _filter(request: AskRequest) -> ChunkFilter:
    return ChunkFilter(norm_ids=tuple(request.norm_ids), topics=tuple(request.topics))


def _sse(event: AnswerEvent) -> str:
    return f"event: {event.type}\ndata: {event.model_dump_json()}\n\n"


def _log_query(settings: Settings, request: AskRequest, done: DoneEvent) -> None:
    if not settings.query_log:
        return
    record = {
        "ts": datetime.now(UTC).isoformat(),
        "request_id": structlog.contextvars.get_contextvars().get("request_id"),
        "question": request.question,
        "norm_ids": request.norm_ids,
        "abstained": done.abstained,
        "cited_sources": done.cited_sources,
        "model": done.model,
        "usage": done.usage,
        "cost_usd": done.cost_usd,
        "timings_ms": done.timings_ms,
    }
    try:
        path = settings.data_dir / "queries.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        log.warning("query_log.write_failed")


async def _request_context(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:16]
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(request_id=request_id)
    started = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    log.info(
        "http.request",
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        duration_ms=round((time.perf_counter() - started) * 1000, 1),
    )
    return response
