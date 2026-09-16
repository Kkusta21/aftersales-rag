"""Offline evaluation: retrieval quality per mode, abstention accuracy, and routing accuracy.

Usage: python eval/run_eval.py [--write]
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))

from aftersales_rag.agent import Assistant, classify  # noqa: E402
from aftersales_rag.config import get_settings  # noqa: E402
from aftersales_rag.retrieval import detect_model  # noqa: E402


def load(name: str) -> list[dict]:
    return [json.loads(row) for row in (HERE / name).read_text().splitlines() if row.strip()]


def section_key(chunk_id: str) -> str:
    return chunk_id.rsplit("::", 1)[0]


def retrieval_metrics(assistant: Assistant, mode: str, k: int = 3) -> dict:
    rows = load("retrieval_set.jsonl")
    hit1 = hitk = mrr = 0.0
    misses = []
    for row in rows:
        hits = assistant.retriever.search(row["question"], k=10, mode=mode)
        keys = [section_key(h.chunk.id) for h in hits]
        rank = keys.index(row["expected"]) + 1 if row["expected"] in keys else None
        hit1 += rank == 1
        hitk += bool(rank and rank <= k)
        mrr += 1 / rank if rank else 0
        if rank != 1:
            misses.append((row["question"], rank))
    n = len(rows)
    return {"mode": mode, "n": n, "hit@1": hit1 / n, f"hit@{k}": hitk / n, "mrr": mrr / n, "misses": misses}


def abstention_metrics(assistant: Assistant) -> dict:
    rows = load("abstention_set.jsonl")
    s = assistant.settings
    correct, errors = 0, []
    for row in rows:
        hits = assistant.retriever.search(row["question"], k=s.top_k, mode=s.retrieval_mode)
        answered = bool(hits) and hits[0].relevance >= s.min_relevance
        if answered == row["answerable"]:
            correct += 1
        else:
            errors.append((row["question"], round(hits[0].relevance, 3)))
    return {"n": len(rows), "accuracy": correct / len(rows), "errors": errors}


def routing_metrics() -> dict:
    rows = load("routing_set.jsonl")
    errors = [
        (r["question"], got)
        for r in rows
        if (got := classify(r["question"], detect_model(r["question"]))) != r["route"]
    ]
    return {"n": len(rows), "accuracy": 1 - len(errors) / len(rows), "errors": errors}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="write eval/results.md")
    args = parser.parse_args()

    settings = replace(get_settings(), anthropic_api_key=None)  # retrieval eval needs no LLM
    assistant = Assistant(settings)

    ret = [retrieval_metrics(assistant, m) for m in ("bm25", "dense", "hybrid")]
    abst = abstention_metrics(assistant)
    route = routing_metrics()

    lines = [
        f"Embedder: `{settings.embedder}`, reranker: `{settings.reranker_model or 'none'}`",
        "",
        "| Retrieval mode | Questions | Hit@1 | Hit@3 | MRR |",
        "|---|---|---|---|---|",
    ]
    for r in ret:
        lines.append(f"| {r['mode']} | {r['n']} | {r['hit@1']:.2f} | {r['hit@3']:.2f} | {r['mrr']:.2f} |")
    lines += [
        "",
        f"Abstention accuracy (answer vs. say \"not found\"): **{abst['accuracy']:.2f}** on {abst['n']} questions",
        f"Routing accuracy (manuals / recalls / clarify): **{route['accuracy']:.2f}** on {route['n']} questions",
    ]
    report = "\n".join(lines)
    print(report)
    for r in ret:
        if r["misses"]:
            print(f"\n{r['mode']} not ranked first:", *r["misses"], sep="\n  ")
    if abst["errors"]:
        print("\nabstention errors:", *abst["errors"], sep="\n  ")
    if route["errors"]:
        print("\nrouting errors:", *route["errors"], sep="\n  ")
    if args.write:
        (HERE / "results.md").write_text(report + "\n")


if __name__ == "__main__":
    main()
