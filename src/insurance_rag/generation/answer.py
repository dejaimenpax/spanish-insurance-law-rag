"""Answer questions: retrieve, decide whether to abstain, generate with citations, stream events."""

import time
from collections.abc import AsyncIterator, Sequence
from datetime import date
from typing import Literal

import structlog
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from insurance_rag.domain.models import Chunk
from insurance_rag.generation.llm import CitationDelta, Completed, LlmClient, TextDelta
from insurance_rag.generation.prompts import ABSTENTION, SYSTEM_PROMPT, user_content
from insurance_rag.index.qdrant_store import ChunkFilter, ScoredChunk
from insurance_rag.observability.cost import Usage, cost_usd
from insurance_rag.retrieval.service import SupportsRetrieve

log = structlog.get_logger(__name__)


class Source(BaseModel):
    """A retrieved provision passed to the model, as shown to the user."""

    index: int
    chunk_id: str
    norm_id: str
    norm_short_name: str
    citation: str
    heading: str | None
    url: str
    text: str
    consolidated_as_of: date
    version_in_force_since: date
    application_date: date | None
    not_yet_applicable: bool
    repealed: bool
    score: float
    retrieved_by: list[str]


class Citation(BaseModel):
    source_index: int
    cited_text: str


class SourcesEvent(BaseModel):
    type: Literal["sources"] = "sources"
    sources: list[Source]


class DeltaEvent(BaseModel):
    type: Literal["delta"] = "delta"
    text: str


class CitationEvent(BaseModel):
    type: Literal["citation"] = "citation"
    citation: Citation


class WarningEvent(BaseModel):
    type: Literal["warning"] = "warning"
    code: str
    message: str


class DoneEvent(BaseModel):
    type: Literal["done"] = "done"
    answer: str
    citations: list[Citation]
    cited_sources: list[int]
    abstained: bool
    model: str | None
    stop_reason: str | None
    usage: dict[str, int] = Field(default_factory=dict)
    cost_usd: float | None = None
    timings_ms: dict[str, float] = Field(default_factory=dict)


AnswerEvent = SourcesEvent | DeltaEvent | CitationEvent | WarningEvent | DoneEvent


class AnswerService:
    def __init__(
        self,
        *,
        retriever: SupportsRetrieve,
        llm: LlmClient,
        top_k: int = 8,
        abstain_below: float | None = None,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k
        self.abstain_below = abstain_below
        """Minimum reranker score of the best hit; below it the LLM is not called."""

    async def stream(
        self, question: str, *, chunk_filter: ChunkFilter | None = None, today: date | None = None
    ) -> AsyncIterator[AnswerEvent]:
        today = today or date.today()
        started = time.perf_counter()
        result = await run_in_threadpool(
            self.retriever.retrieve, question, k=self.top_k, chunk_filter=chunk_filter
        )
        timings = {f"retrieval_{k}": v for k, v in result.timings_ms.items()}
        timings["retrieval_total"] = _ms(started)

        sources = [_source(i, hit, today) for i, hit in enumerate(result.hits)]
        yield SourcesEvent(sources=sources)
        for source in sources:
            if source.not_yet_applicable and source.application_date:
                yield WarningEvent(
                    code="not_yet_applicable",
                    message=f"{source.citation} no se aplica hasta el "
                    f"{source.application_date.isoformat()}.",
                )

        if self._should_abstain(result.hits):
            yield DeltaEvent(text=ABSTENTION)
            timings["total"] = _ms(started)
            log.info("answer.abstained_before_llm", question=question, hits=len(result.hits))
            yield DoneEvent(
                answer=ABSTENTION,
                citations=[],
                cited_sources=[],
                abstained=True,
                model=None,
                stop_reason=None,
                timings_ms=timings,
            )
            return

        parts: list[str] = []
        citations: list[Citation] = []
        completed: Completed | None = None
        llm_started = time.perf_counter()
        content = user_content(question, [h.chunk for h in result.hits], today=today)
        async for event in self.llm.stream(system=SYSTEM_PROMPT, content=content):
            if isinstance(event, TextDelta):
                if not parts:
                    timings["time_to_first_token"] = _ms(started)
                parts.append(event.text)
                yield DeltaEvent(text=event.text)
            elif isinstance(event, CitationDelta):
                if 0 <= event.search_result_index < len(sources):
                    citation = Citation(
                        source_index=event.search_result_index, cited_text=event.cited_text
                    )
                    citations.append(citation)
                    yield CitationEvent(citation=citation)
                else:  # pragma: no cover - the API only cites blocks it was given
                    log.warning("answer.citation_out_of_range", index=event.search_result_index)
            elif isinstance(event, Completed):
                completed = event
        timings["generation"] = _ms(llm_started)
        timings["total"] = _ms(started)

        answer = "".join(parts).strip()
        if completed and completed.stop_reason == "refusal":
            yield WarningEvent(code="refusal", message="El modelo ha declinado responder.")
        if completed and completed.stop_reason == "max_tokens":
            yield WarningEvent(code="truncated", message="La respuesta se ha truncado.")

        usage = completed.usage if completed else Usage()
        model = completed.model if completed else self.llm.model
        done = DoneEvent(
            answer=answer,
            citations=citations,
            cited_sources=sorted({c.source_index for c in citations}),
            abstained=answer.startswith(ABSTENTION.rstrip(".")),
            model=model,
            stop_reason=completed.stop_reason if completed else None,
            usage=usage.__dict__,
            cost_usd=cost_usd(model, usage),
            timings_ms=timings,
        )
        log.info(
            "answer.completed",
            question=question,
            abstained=done.abstained,
            citations=len(citations),
            sources=len(sources),
            cost_usd=done.cost_usd,
            **usage.__dict__,
            **timings,
        )
        yield done

    async def answer(
        self, question: str, *, chunk_filter: ChunkFilter | None = None, today: date | None = None
    ) -> tuple[list[Source], list[str], DoneEvent]:
        """Non-streaming convenience: sources, warnings and the final event."""
        sources: list[Source] = []
        warnings: list[str] = []
        async for event in self.stream(question, chunk_filter=chunk_filter, today=today):
            if isinstance(event, SourcesEvent):
                sources = event.sources
            elif isinstance(event, WarningEvent):
                warnings.append(event.message)
            elif isinstance(event, DoneEvent):
                return sources, warnings, event
        raise RuntimeError("answer stream ended without a done event")

    def _should_abstain(self, hits: Sequence[ScoredChunk]) -> bool:
        if not hits:
            return True
        if self.abstain_below is None:
            return False
        reranked = [h.score for h in hits if "rerank" in h.sources]
        return bool(reranked) and max(reranked) < self.abstain_below


def _source(index: int, hit: ScoredChunk, today: date) -> Source:
    chunk: Chunk = hit.chunk
    return Source(
        index=index,
        chunk_id=chunk.chunk_id,
        norm_id=chunk.norm_id,
        norm_short_name=chunk.norm_short_name,
        citation=chunk.citation_label,
        heading=chunk.heading,
        url=chunk.url,
        text=chunk.text,
        consolidated_as_of=chunk.consolidated_as_of,
        version_in_force_since=chunk.version_in_force_since,
        application_date=chunk.application_date,
        not_yet_applicable=bool(chunk.application_date and chunk.application_date > today),
        repealed=chunk.repealed,
        score=round(hit.score, 4),
        retrieved_by=list(hit.sources),
    )


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)
