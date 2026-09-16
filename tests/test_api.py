from fastapi.testclient import TestClient

from aftersales_rag import agent
from aftersales_rag.api import app


def test_ask_endpoint(assistant, monkeypatch):
    monkeypatch.setattr(agent, "get_assistant", lambda: assistant)
    monkeypatch.setattr("aftersales_rag.api.get_assistant", lambda: assistant)
    client = TestClient(app)
    assert client.get("/health").json() == {"status": "ok"}
    body = client.post("/ask", json={"question": "Tern EV charge port light flashing red"}).json()
    assert body["route"] == "manuals"
    assert body["citations"]


def test_rejects_empty_question():
    client = TestClient(app)
    assert client.post("/ask", json={"question": ""}).status_code == 422
