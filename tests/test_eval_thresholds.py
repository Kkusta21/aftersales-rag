"""Fails CI if retrieval or routing quality regresses."""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("run_eval", Path(__file__).parents[1] / "eval" / "run_eval.py")
run_eval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_eval)


def test_hybrid_retrieval_quality(assistant):
    m = run_eval.retrieval_metrics(assistant, "hybrid")
    assert m["hit@3"] >= 0.9
    assert m["mrr"] >= 0.85


def test_abstention_quality(assistant):
    assert run_eval.abstention_metrics(assistant)["accuracy"] >= 0.85


def test_routing_quality():
    assert run_eval.routing_metrics()["accuracy"] == 1.0
