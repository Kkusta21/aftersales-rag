"""Answer generation with citations. Uses Claude when a key is set, otherwise an extractive fallback."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .retrieval import Hit, content_terms, tokenize

NOT_FOUND = (
    "I couldn't find that in the Norvik service documents. "
    "Please contact your dealer or Norvik Roadside Assistance."
)

SYSTEM_PROMPT = """You are an after-sales assistant for Norvik Motors dealers and customers.
Answer ONLY from the numbered sources. Cite every factual sentence with its source number, like [1].
If the sources do not contain the answer, reply exactly: NOT_FOUND
For safety warnings (red lights, high voltage, brakes), put the safety instruction first.
Keep answers short and practical."""


@dataclass
class Answer:
    text: str
    citations: list[str] = field(default_factory=list)
    abstained: bool = False


def format_sources(hits: list[Hit]) -> str:
    return "\n\n".join(f"[{n}] ({h.chunk.citation()})\n{h.chunk.text}" for n, h in enumerate(hits, 1))


def _cited(text: str, hits: list[Hit]) -> list[str]:
    nums = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text)})
    return [f"[{n}] {hits[n - 1].chunk.citation()}" for n in nums if 0 < n <= len(hits)]


class ExtractiveGenerator:
    """Picks the sentences that best overlap the question. No LLM, fully deterministic."""

    def generate(self, question: str, hits: list[Hit], max_sentences: int = 3) -> Answer:
        terms = content_terms(question)
        scored = []
        for n, hit in enumerate(hits, 1):
            body = hit.chunk.text.split(". ", 1)[-1]
            for pos, sent in enumerate(re.split(r"(?<=[.!?])\s+", body)):
                overlap = len(terms & set(tokenize(sent)))
                if overlap:
                    # prefer higher-ranked sources, then earlier sentences
                    scored.append((overlap - 0.1 * (n - 1) - 0.01 * pos, n, sent.strip()))
        if not scored:
            return Answer(NOT_FOUND, abstained=True)
        best = sorted(scored, key=lambda s: -s[0])[:max_sentences]
        text = " ".join(f"{sent} [{n}]" for _, n, sent in best)
        return Answer(text, _cited(text, hits))


class ClaudeGenerator:
    def __init__(self, model: str, api_key: str):
        import anthropic

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def generate(self, question: str, hits: list[Hit]) -> Answer:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=500,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": f"Sources:\n{format_sources(hits)}\n\nQuestion: {question}"}],
        )
        text = "".join(b.text for b in response.content if b.type == "text").strip()
        if "NOT_FOUND" in text:
            return Answer(NOT_FOUND, abstained=True)
        return Answer(text, _cited(text, hits))


def answer_question(question: str, hits: list[Hit], generator, min_relevance: float) -> Answer:
    if not hits or hits[0].relevance < min_relevance:
        return Answer(NOT_FOUND, abstained=True)
    return generator.generate(question, hits)
