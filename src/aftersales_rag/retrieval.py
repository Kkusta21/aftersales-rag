"""Hybrid retrieval: BM25 + dense vectors, fused with reciprocal rank fusion, optional reranking."""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from .embeddings import Embedder
from .ingest import Chunk

STOPWORDS = set(
    """a an and are as at be by can do does for from how i if in is it my of on or should the
    this to what when where which why with you your me we our there their than then after
    about into much many long often car vehicle norvik""".split()
)

MODEL_ALIASES = {
    "Aster Hybrid": ("aster",),
    "Tern EV": ("tern",),
}

MODEL_TOKENS = {"aster", "hybrid", "tern", "ev"}

RRF_K = 60


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


def content_terms(text: str) -> set[str]:
    """Query terms minus model names, which match almost every chunk of that model's manual."""
    return set(tokenize(text)) - MODEL_TOKENS


def detect_model(text: str) -> str | None:
    lowered = text.lower()
    for model, aliases in MODEL_ALIASES.items():
        if any(re.search(rf"\b{a}\b", lowered) for a in aliases):
            return model
    return None


@dataclass
class Hit:
    chunk: Chunk
    score: float  # fused ranking score (only meaningful for ordering)
    relevance: float  # 0..1, used to decide whether to answer at all
    bm25: float = 0.0
    cosine: float = 0.0


class HybridRetriever:
    def __init__(self, chunks: list[Chunk], embedder: Embedder, reranker_model: str | None = None):
        self.chunks = chunks
        self.embedder = embedder
        self._tokens = [tokenize(c.text) for c in chunks]
        self.bm25 = BM25Okapi(self._tokens)
        self.vectors = embedder.embed([c.text for c in chunks])
        self.reranker = None
        if reranker_model:
            from sentence_transformers import CrossEncoder  # optional dependency

            self.reranker = CrossEncoder(reranker_model)

    def _candidates(self, model: str | None) -> np.ndarray:
        if model is None:
            return np.arange(len(self.chunks))
        return np.array([i for i, c in enumerate(self.chunks) if not c.models or model in c.models])

    @staticmethod
    def _ranks(scores: np.ndarray, idx: np.ndarray) -> dict[int, int]:
        order = idx[np.argsort(-scores[idx], kind="stable")]
        return {int(i): r for r, i in enumerate(order)}

    def _coverage(self, query_terms: list[str], i: int) -> float:
        terms = set(query_terms) - MODEL_TOKENS
        if not terms:
            return 0.0
        chunk_terms = set(self._tokens[i])
        return sum(t in chunk_terms for t in terms) / len(terms)

    def search(self, query: str, k: int = 4, mode: str = "hybrid", model: str | None = None) -> list[Hit]:
        if mode not in {"bm25", "dense", "hybrid"}:
            raise ValueError(f"Unknown retrieval mode: {mode}")
        model = model or detect_model(query)
        idx = self._candidates(model)
        terms = tokenize(query)
        bm25_scores = np.asarray(self.bm25.get_scores(terms), dtype=np.float32)
        cosines = self.vectors @ self.embedder.embed([query])[0]

        fused: dict[int, float] = {}
        if mode in {"bm25", "hybrid"}:
            for i, r in self._ranks(bm25_scores, idx).items():
                fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + r)
        if mode in {"dense", "hybrid"}:
            for i, r in self._ranks(cosines, idx).items():
                fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + r)

        pool = sorted(fused, key=fused.get, reverse=True)[: max(k * 3, 10)]
        if self.reranker is not None:
            scores = self.reranker.predict([(query, self.chunks[i].text) for i in pool])
            pool = [i for _, i in sorted(zip(scores, pool, strict=True), key=lambda p: -p[0])]

        hits = []
        for i in pool[:k]:
            relevance = 0.5 * self._coverage(terms, i) + 0.5 * max(0.0, float(cosines[i]))
            hits.append(Hit(self.chunks[i], fused[i], relevance, float(bm25_scores[i]), float(cosines[i])))
        return hits
