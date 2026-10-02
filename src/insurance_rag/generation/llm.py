"""Streaming access to Claude with native search-result citations."""

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import structlog
from anthropic.types.beta import BetaOutputConfigParam

from insurance_rag.observability.cost import Usage

log = structlog.get_logger(__name__)

FALLBACK_BETA = "server-side-fallback-2026-07-01"
Effort = Literal["low", "medium", "high", "xhigh", "max"]


@dataclass(frozen=True)
class TextDelta:
    text: str


@dataclass(frozen=True)
class CitationDelta:
    search_result_index: int
    cited_text: str


@dataclass(frozen=True)
class Completed:
    model: str
    stop_reason: str | None
    usage: Usage


LlmEvent = TextDelta | CitationDelta | Completed


class LlmClient(Protocol):
    @property
    def model(self) -> str: ...

    def stream(self, *, system: str, content: list[dict[str, Any]]) -> AsyncIterator[LlmEvent]: ...


class ClaudeClient:
    """Claude via the Anthropic SDK: adaptive thinking, configurable effort, refusal fallback."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        effort: Effort,
        max_tokens: int,
        timeout: float = 120.0,
    ) -> None:
        from anthropic import AsyncAnthropic

        self._client = AsyncAnthropic(api_key=api_key, timeout=timeout, max_retries=2)
        self._model = model
        self._effort = effort
        self._max_tokens = max_tokens

    @property
    def model(self) -> str:
        return self._model

    async def stream(
        self, *, system: str, content: list[dict[str, Any]]
    ) -> AsyncIterator[LlmEvent]:
        output_config: BetaOutputConfigParam = {"effort": self._effort}
        async with self._client.beta.messages.stream(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system,
            messages=[{"role": "user", "content": content}],  # type: ignore[typeddict-item]
            output_config=output_config,
            betas=[FALLBACK_BETA],
            fallbacks="default",
        ) as stream:
            async for event in stream:
                if event.type == "text":
                    yield TextDelta(event.text)
                elif event.type == "content_block_delta" and event.delta.type == "citations_delta":
                    citation = event.delta.citation
                    if citation.type == "search_result_location":
                        yield CitationDelta(citation.search_result_index, citation.cited_text)
            final = await stream.get_final_message()

        usage = final.usage
        yield Completed(
            model=final.model,
            stop_reason=final.stop_reason,
            usage=Usage(
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cache_read_input_tokens=usage.cache_read_input_tokens or 0,
                cache_creation_input_tokens=usage.cache_creation_input_tokens or 0,
            ),
        )
