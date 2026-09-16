from aftersales_rag.generation import NOT_FOUND


def test_manual_answer_has_citation(assistant):
    r = assistant.ask("Which engine oil grade does the Aster Hybrid take?")
    assert r["route"] == "manuals"
    assert "0W-20" in r["answer"]
    assert r["citations"]


def test_recalls_filtered_by_year(assistant):
    r = assistant.ask("Any recalls on my 2023 Tern EV?")
    assert r["route"] == "recalls"
    ids = [x["campaign_id"] for x in r["recalls"]]
    assert ids == ["NVK-24-011"]


def test_closed_recalls_hidden(assistant):
    ids = [x["campaign_id"] for x in assistant.recall_db.lookup("Aster Hybrid", 2022)]
    assert "NVK-23-007" not in ids


def test_asks_for_model_when_needed(assistant):
    r = assistant.ask("What tyre pressure should I use?")
    assert r["route"] == "clarify"


def test_abstains_out_of_scope(assistant):
    r = assistant.ask("Recipe for paella")
    assert r["abstained"] and r["answer"] == NOT_FOUND
