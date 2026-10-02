from datetime import date

from insurance_rag.generation.answer import (
    AnswerService,
    CitationEvent,
    DeltaEvent,
    DoneEvent,
    SourcesEvent,
    WarningEvent,
)
from insurance_rag.generation.prompts import ABSTENTION
from tests.fakes import FakeLlm, FakeRetriever, answer_events, make_chunk, scored

TODAY = date(2026, 10, 2)


async def _collect(service: AnswerService, question: str = "¿Plazo de pago?") -> list[object]:
    return [e async for e in service.stream(question, today=TODAY)]


async def test_streams_sources_text_citations_and_done() -> None:
    llm = FakeLlm(answer_events())
    service = AnswerService(retriever=FakeRetriever([scored(make_chunk("18"))]), llm=llm)

    events = await _collect(service)

    assert [type(e) for e in events] == [SourcesEvent, DeltaEvent, CitationEvent, DoneEvent]
    done = events[-1]
    assert isinstance(done, DoneEvent)
    assert done.cited_sources == [0]
    assert not done.abstained
    assert done.cost_usd == round((1000 * 4 + 100 * 20) / 1_000_000, 6)
    assert "time_to_first_token" in done.timings_ms


async def test_passes_chunks_as_citable_search_results() -> None:
    llm = FakeLlm(answer_events())
    chunk = make_chunk("20", apartado="4", heading="Mora")
    service = AnswerService(retriever=FakeRetriever([scored(chunk)]), llm=llm)

    await _collect(service)

    [content] = llm.calls
    result, question = content
    assert result["type"] == "search_result"
    assert result["citations"] == {"enabled": True}
    assert result["source"] == chunk.url
    assert result["title"].startswith("LCS, art. 20.4 — Mora (texto consolidado a 2025-07-25")
    assert [b["text"] for b in result["content"][1:]] == [
        "Texto del artículo 20.",
        "Segundo párrafo.",
    ]
    assert question["text"].endswith("Pregunta: ¿Plazo de pago?")


async def test_abstains_without_calling_the_model_when_retrieval_is_weak() -> None:
    llm = FakeLlm(answer_events())
    service = AnswerService(
        retriever=FakeRetriever([scored(make_chunk(), score=0.01)]), llm=llm, abstain_below=0.1
    )

    events = await _collect(service)

    done = events[-1]
    assert isinstance(done, DoneEvent)
    assert done.abstained
    assert done.answer == ABSTENTION
    assert llm.calls == []


async def test_detects_abstention_written_by_the_model() -> None:
    llm = FakeLlm(answer_events(ABSTENTION + " Se han consultado la LCS y la LOSSEAR."))
    service = AnswerService(retriever=FakeRetriever([scored(make_chunk())]), llm=llm)

    done = (await _collect(service))[-1]

    assert isinstance(done, DoneEvent)
    assert done.abstained


async def test_warns_about_provisions_that_do_not_apply_yet() -> None:
    future = make_chunk("1", application_date=date(2027, 1, 30), norm_short_name="Directiva 2025/2")
    service = AnswerService(retriever=FakeRetriever([scored(future)]), llm=FakeLlm(answer_events()))

    events = await _collect(service)

    warnings = [e for e in events if isinstance(e, WarningEvent)]
    assert [w.code for w in warnings] == ["not_yet_applicable"]
    assert "2027-01-30" in warnings[0].message
    sources = events[0]
    assert isinstance(sources, SourcesEvent)
    assert sources.sources[0].not_yet_applicable
