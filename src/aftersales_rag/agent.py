"""LangGraph workflow: route the question, then search manuals, check recalls, or ask for the model."""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from .config import Settings, get_settings
from .embeddings import get_embedder
from .generation import ClaudeGenerator, ExtractiveGenerator, answer_question
from .ingest import load_corpus
from .recalls import RecallDB
from .retrieval import HybridRetriever, detect_model

Route = Literal["manuals", "recalls", "clarify"]

GENERAL_WORDS = ("warranty", "covered", "claim", "goodwill", "book a service", "booking")
RECALL_WORDS = ("recall", "recalls", "campaign", "safety notice")
# Topics whose answer differs by model, so we must know which car before answering.
MODEL_SPECIFIC = ("tyre", "tire", "pressure", "oil", "coolant", "charging", "charge", "battery", "filter", "error code")


class AgentState(TypedDict, total=False):
    question: str
    model: str | None
    year: int | None
    route: Route
    answer: str
    citations: list[str]
    abstained: bool
    recalls: list[dict]


def detect_year(text: str) -> int | None:
    match = re.search(r"\b(20[12]\d)\b", text)
    return int(match.group(1)) if match else None


def classify(question: str, model: str | None) -> Route:
    q = question.lower()
    if any(w in q for w in RECALL_WORDS):
        return "recalls" if model else "clarify"
    if model is None and any(w in q for w in MODEL_SPECIFIC) and not any(w in q for w in GENERAL_WORDS):
        return "clarify"
    return "manuals"


class Assistant:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        s = self.settings
        chunks = load_corpus(s.docs_dir, s.chunk_words, s.chunk_overlap)
        self.retriever = HybridRetriever(chunks, get_embedder(s.embedder, s.embedding_model), s.reranker_model)
        self.recall_db = RecallDB(s.recalls_path)
        self.generator = (
            ClaudeGenerator(s.llm_model, s.anthropic_api_key) if s.anthropic_api_key else ExtractiveGenerator()
        )
        self.graph = self._build_graph()

    # --- nodes -----------------------------------------------------------
    def _route(self, state: AgentState) -> AgentState:
        q = state["question"]
        model = state.get("model") or detect_model(q)
        return {"model": model, "year": state.get("year") or detect_year(q), "route": classify(q, model)}

    def _manuals(self, state: AgentState) -> AgentState:
        s = self.settings
        hits = self.retriever.search(state["question"], k=s.top_k, mode=s.retrieval_mode, model=state.get("model"))
        ans = answer_question(state["question"], hits, self.generator, s.min_relevance)
        return {"answer": ans.text, "citations": ans.citations, "abstained": ans.abstained}

    def _recalls(self, state: AgentState) -> AgentState:
        model, year = state["model"], state.get("year")
        found = self.recall_db.lookup(model, year)
        label = f"{model} {year}" if year else model
        if not found:
            text = f"No open recalls found for the {label}."
        else:
            lines = [f"Open recalls for the {label}:"]
            for r in found:
                lines.append(f"- {r['campaign_id']} ({r['component']}): {r['summary']} Remedy: {r['remedy']}")
            text = "\n".join(lines)
        return {"answer": text, "recalls": found, "citations": [r["campaign_id"] for r in found], "abstained": False}

    def _clarify(self, state: AgentState) -> AgentState:
        models = " or ".join(self.recall_db_models())
        return {
            "answer": f"Which model is this about: {models}? The answer depends on the car.",
            "citations": [],
            "abstained": False,
        }

    def recall_db_models(self) -> list[str]:
        return sorted({r["model"] for r in self.recall_db.records})

    def _build_graph(self):
        g = StateGraph(AgentState)
        g.add_node("route", self._route)
        g.add_node("manuals", self._manuals)
        g.add_node("recalls", self._recalls)
        g.add_node("clarify", self._clarify)
        g.add_edge(START, "route")
        g.add_conditional_edges("route", lambda s: s["route"], ["manuals", "recalls", "clarify"])
        for node in ("manuals", "recalls", "clarify"):
            g.add_edge(node, END)
        return g.compile()

    # --- public ----------------------------------------------------------
    def ask(self, question: str, model: str | None = None, year: int | None = None) -> AgentState:
        return self.graph.invoke({"question": question, "model": model, "year": year})


@lru_cache(maxsize=1)
def get_assistant() -> Assistant:
    return Assistant()
