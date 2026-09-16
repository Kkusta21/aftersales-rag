from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    docs_dir: Path = field(default_factory=lambda: Path(os.getenv("DOCS_DIR", ROOT / "data" / "docs")))
    recalls_path: Path = field(
        default_factory=lambda: Path(os.getenv("RECALLS_PATH", ROOT / "data" / "recalls" / "recalls.json"))
    )
    # "hashing" works offline with no downloads. Use "sentence-transformers" for real semantic search.
    embedder: str = os.getenv("EMBEDDER", "hashing")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    reranker_model: str | None = os.getenv("RERANKER_MODEL") or None
    retrieval_mode: str = os.getenv("RETRIEVAL_MODE", "hybrid")
    top_k: int = int(os.getenv("TOP_K", "4"))
    chunk_words: int = int(os.getenv("CHUNK_WORDS", "120"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "30"))
    # Below this relevance the assistant says it does not know instead of guessing.
    min_relevance: float = float(os.getenv("MIN_RELEVANCE", "0.25"))
    llm_model: str = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5")
    anthropic_api_key: str | None = os.getenv("ANTHROPIC_API_KEY") or None


def get_settings() -> Settings:
    return Settings()
