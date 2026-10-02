import json
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from insurance_rag.api.app import AppState, create_app
from insurance_rag.config import Settings
from insurance_rag.generation.answer import AnswerService
from tests.fakes import FakeLlm, FakeRetriever, answer_events, make_chunk, scored


@pytest.fixture
def store() -> MagicMock:
    store = MagicMock()
    store.ready.return_value = True
    store.count.return_value = 42
    return store


def _client(tmp_path: Path, store: MagicMock, *, with_answers: bool = True) -> TestClient:
    retriever = FakeRetriever([scored(make_chunk("18"))])
    answers = (
        AnswerService(retriever=retriever, llm=FakeLlm(answer_events())) if with_answers else None
    )
    settings = Settings(data_dir=tmp_path, anthropic_api_key=None)
    return TestClient(create_app(state=AppState(settings, store, answers)))


@pytest.fixture
def client(tmp_path: Path, store: MagicMock) -> Iterator[TestClient]:
    with _client(tmp_path, store) as c:
        yield c


def test_ready_reports_each_check(tmp_path: Path, store: MagicMock) -> None:
    with _client(tmp_path, store, with_answers=False) as c:
        response = c.get("/ready")

    assert response.status_code == 503
    assert response.json()["checks"] == {"index": True, "index_populated": True, "answering": False}


def test_ask_returns_answer_sources_and_citations(client: TestClient, tmp_path: Path) -> None:
    response = client.post(
        "/ask", json={"question": "¿Plazo de pago?", "norm_ids": ["BOE-A-1980-22501"]}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"].startswith("El asegurador")
    assert body["sources"][0]["citation"] == "LCS, art. 18"
    assert body["sources"][0]["url"].endswith("#a18")
    assert body["cited_sources"] == [0]
    assert response.headers["X-Request-ID"]
    logged = [json.loads(line) for line in (tmp_path / "queries.jsonl").read_text().splitlines()]
    assert logged[0]["question"] == "¿Plazo de pago?"
    assert logged[0]["cost_usd"] > 0


def test_ask_stream_emits_server_sent_events_in_order(client: TestClient) -> None:
    with client.stream("POST", "/ask/stream", json={"question": "¿Plazo de pago?"}) as response:
        body = "".join(response.iter_text())

    assert response.headers["content-type"].startswith("text/event-stream")
    events = [
        line.removeprefix("event: ") for line in body.splitlines() if line.startswith("event: ")
    ]
    assert events == ["sources", "delta", "citation", "done"]


def test_ask_without_api_key_is_unavailable(tmp_path: Path, store: MagicMock) -> None:
    with _client(tmp_path, store, with_answers=False) as c:
        response = c.post("/ask", json={"question": "¿Plazo de pago?"})

    assert response.status_code == 503


def test_rejects_empty_questions(client: TestClient) -> None:
    assert client.post("/ask", json={"question": ""}).status_code == 422


def test_lists_norms_with_index_counts(client: TestClient) -> None:
    norms = client.get("/norms").json()

    assert len(norms) == 10
    assert norms[0]["indexed_chunks"] == 42
