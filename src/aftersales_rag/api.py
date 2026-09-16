from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .agent import get_assistant

app = FastAPI(title="Norvik After-Sales Assistant", version="0.1.0")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    model: str | None = None
    year: int | None = None


class AskResponse(BaseModel):
    answer: str
    route: str
    model: str | None
    citations: list[str]
    abstained: bool


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    result = get_assistant().ask(req.question, req.model, req.year)
    return AskResponse(
        answer=result["answer"],
        route=result["route"],
        model=result.get("model"),
        citations=result.get("citations", []),
        abstained=result.get("abstained", False),
    )
