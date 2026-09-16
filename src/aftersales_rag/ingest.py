"""Load markdown documents with YAML front matter and split them into section-aware chunks."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

FRONT_MATTER = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)


@dataclass
class Chunk:
    id: str
    doc_id: str
    title: str
    section: str
    doc_type: str
    text: str
    models: list[str] = field(default_factory=list)

    def citation(self) -> str:
        return f"{self.title} > {self.section}"


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _split_sections(body: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    current, lines = "Introduction", []
    for line in body.splitlines():
        if line.startswith("# "):
            if any(ln.strip() for ln in lines):
                sections.append((current, "\n".join(lines).strip()))
            current, lines = line[2:].strip(), []
        else:
            lines.append(line)
    if any(ln.strip() for ln in lines):
        sections.append((current, "\n".join(lines).strip()))
    return sections


def _windows(text: str, size: int, overlap: int) -> list[str]:
    words = text.split()
    if len(words) <= size:
        return [text]
    step = max(1, size - overlap)
    out = []
    for start in range(0, len(words), step):
        out.append(" ".join(words[start : start + size]))
        if start + size >= len(words):
            break
    return out


def load_document(path: Path, chunk_words: int = 120, overlap: int = 30) -> list[Chunk]:
    raw = path.read_text(encoding="utf-8")
    meta: dict = {}
    match = FRONT_MATTER.match(raw)
    if match:
        meta = yaml.safe_load(match.group(1)) or {}
        raw = raw[match.end() :]
    doc_id = path.stem
    title = meta.get("title", doc_id)
    chunks = []
    for section, text in _split_sections(raw):
        for i, window in enumerate(_windows(text, chunk_words, overlap)):
            chunks.append(
                Chunk(
                    id=f"{doc_id}::{slugify(section)}::{i}",
                    doc_id=doc_id,
                    title=title,
                    section=section,
                    doc_type=meta.get("doc_type", "document"),
                    # Prefix the heading so both BM25 and embeddings see it.
                    text=f"{section}. {window}",
                    models=list(meta.get("models", [])),
                )
            )
    return chunks


def load_corpus(docs_dir: Path, chunk_words: int = 120, overlap: int = 30) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(docs_dir.glob("*.md")):
        chunks.extend(load_document(path, chunk_words, overlap))
    if not chunks:
        raise FileNotFoundError(f"No markdown documents found in {docs_dir}")
    return chunks
