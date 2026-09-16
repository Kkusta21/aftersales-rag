import pytest

from aftersales_rag.retrieval import detect_model


def test_detect_model():
    assert detect_model("my Tern keeps beeping") == "Tern EV"
    assert detect_model("Aster oil change") == "Aster Hybrid"
    assert detect_model("asteroid") is None


@pytest.mark.parametrize("mode", ["bm25", "dense", "hybrid"])
def test_finds_charging_fault(assistant, mode):
    hits = assistant.retriever.search("charge port light flashing red", k=3, mode=mode, model="Tern EV")
    assert "Charging fault" in hits[0].chunk.section


def test_model_filter_excludes_other_manual(assistant):
    hits = assistant.retriever.search("tyre pressure", k=5, model="Tern EV")
    assert all("Aster" not in h.chunk.title for h in hits)


def test_unknown_mode(assistant):
    with pytest.raises(ValueError):
        assistant.retriever.search("oil", mode="magic")
