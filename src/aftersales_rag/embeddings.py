"""Embedding backends. The hashing embedder needs no downloads, so tests and CI run offline."""
from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class HashingEmbedder:
    """Character n-gram feature hashing. A lexical-ish baseline, not a semantic model.

    It exists so the pipeline runs anywhere. Swap in SentenceTransformerEmbedder for real use.
    """

    def __init__(self, dim: int = 4096, ngram_range: tuple[int, int] = (3, 5)):
        self.dim = dim
        self.ngram_range = ngram_range

    def _features(self, text: str) -> list[str]:
        text = " " + re.sub(r"\s+", " ", text.lower()) + " "
        feats = re.findall(r"[a-z0-9]+", text)
        lo, hi = self.ngram_range
        for n in range(lo, hi + 1):
            feats.extend(text[i : i + n] for i in range(len(text) - n + 1))
        return feats

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feat in self._features(text):
                h = int(hashlib.md5(feat.encode()).hexdigest(), 16)
                sign = 1.0 if (h >> 1) & 1 else -1.0
                out[row, h % self.dim] += sign
            out[row] = np.sign(out[row]) * np.log1p(np.abs(out[row]))
        return _normalize(out)


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer  # optional dependency

        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vectors, dtype=np.float32)


def get_embedder(name: str, model_name: str | None = None) -> Embedder:
    if name == "hashing":
        return HashingEmbedder()
    if name == "sentence-transformers":
        return SentenceTransformerEmbedder(model_name or "sentence-transformers/all-MiniLM-L6-v2")
    raise ValueError(f"Unknown embedder: {name}")
