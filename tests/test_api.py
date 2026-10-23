"""HTTP API via FastAPI's TestClient."""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    with TestClient(app) as c:
        yield c


def post(client: TestClient, message: str, role: str = "student", session_id: str = "api-test") -> dict:
    response = client.post("/api/chat", json={"session_id": session_id, "role": role, "message": message})
    assert response.status_code == 200, response.text
    return response.json()


def test_health(client: TestClient) -> None:
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["intents"] == 25


def test_chat_response_shape(client: TestClient) -> None:
    body = post(client, "What are the library timings?", session_id="shape")
    assert set(body) >= {"reply", "intent", "confidence", "suggestions"}
    assert body["intent"] == "library"
    assert 0.4 <= body["confidence"] <= 1


def test_two_turn_conversation(client: TestClient) -> None:
    first = post(client, "What's the exam schedule for CSE?", session_id="two-turn")
    second = post(client, "and for ECE?", session_id="two-turn")
    assert first["intent"] == second["intent"] == "exam_schedule"
    assert second["resolved_by"] == "follow_up"
    assert second["slots"]["department"] == "ECE"


def test_role_restriction_over_http(client: TestClient) -> None:
    body = post(client, "show admission statistics", role="student", session_id="restricted")
    assert body["resolved_by"] == "restricted"


@pytest.mark.parametrize(
    "payload",
    [
        {"session_id": "x", "role": "dean", "message": "hi"},
        {"session_id": "x", "role": "student", "message": ""},
        {"session_id": "", "role": "student", "message": "hi"},
        {"role": "student", "message": "hi"},
    ],
)
def test_invalid_requests_are_rejected(client: TestClient, payload: dict) -> None:
    assert client.post("/api/chat", json=payload).status_code == 422


def test_cors_allows_dev_frontend(client: TestClient) -> None:
    response = client.options(
        "/api/chat",
        headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"},
    )
    assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"
