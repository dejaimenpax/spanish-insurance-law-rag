"""Request and response models of the HTTP API."""

from datetime import date

from pydantic import BaseModel, Field

from insurance_rag.generation.answer import Citation, Source


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    norm_ids: list[str] = Field(
        default_factory=list, description="Restrict retrieval to these norms."
    )
    topics: list[str] = Field(
        default_factory=list, description="Restrict retrieval to these topics."
    )


class AskResponse(BaseModel):
    answer: str
    abstained: bool
    sources: list[Source]
    citations: list[Citation]
    cited_sources: list[int]
    warnings: list[str]
    model: str | None
    usage: dict[str, int]
    cost_usd: float | None
    timings_ms: dict[str, float]


class NormInfo(BaseModel):
    id: str
    short_name: str
    title: str
    jurisdiction: str
    legal_effect: str
    topics: list[str]
    source_url: str
    consolidated_as_of: date | None
    application_date: date | None
    indexed_chunks: int


class HealthResponse(BaseModel):
    status: str
    version: str


class ReadyResponse(BaseModel):
    ready: bool
    checks: dict[str, bool]
