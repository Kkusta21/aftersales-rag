"""Structured recall lookup. Recalls are exact records, so they are filtered, not embedded."""
from __future__ import annotations

import json
from pathlib import Path


class RecallDB:
    def __init__(self, path: Path):
        self.records: list[dict] = json.loads(Path(path).read_text(encoding="utf-8"))

    def lookup(self, model: str | None = None, year: int | None = None, include_closed: bool = False) -> list[dict]:
        results = []
        for rec in self.records:
            if model and rec["model"].lower() != model.lower():
                continue
            if year and year not in rec["years"]:
                continue
            if not include_closed and rec["status"] != "open":
                continue
            results.append(rec)
        return sorted(results, key=lambda r: r["date"], reverse=True)
