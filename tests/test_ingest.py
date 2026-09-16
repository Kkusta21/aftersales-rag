from aftersales_rag.config import get_settings
from aftersales_rag.ingest import _windows, load_corpus


def test_corpus_has_metadata():
    chunks = load_corpus(get_settings().docs_dir)
    assert chunks
    assert all(c.title and c.section and c.doc_type for c in chunks)
    assert len({c.id for c in chunks}) == len(chunks)


def test_windows_overlap():
    text = " ".join(str(i) for i in range(250))
    wins = _windows(text, size=100, overlap=20)
    assert len(wins) == 3
    assert wins[1].split()[0] == "80"
    assert wins[-1].split()[-1] == "249"
