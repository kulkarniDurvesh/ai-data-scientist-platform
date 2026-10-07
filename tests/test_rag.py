"""
Phase 7: retrieval-augmented answers over documents.

A synthetic online-shop document set (returns, warehouse, shipping) with
known answers; local LSA vectors so no model is needed; the language
model is a scripted stand-in.
"""

import time

import pytest
from fastapi.testclient import TestClient

from core.llm import ScriptedProvider, reset_provider, use_provider
from core.rag import NOT_COVERED, OllamaEmbedder, build_index, chunk_markdown, extractive_answer, model_answer
from service.knowledge import KnowledgeBase

RETURNS = """# Returns policy

*Example shop documentation.*

## Time limit

Items can be returned within 30 days of delivery. Sale items can be returned within 14 days.

## Refunds

Refunds are paid to the original payment method within 5 working days after the item arrives at the warehouse.

## Exceptions

Opened software, gift cards and personalised items cannot be returned.
"""

WAREHOUSE = """# WH-07: Warehouse storage

## Storage zones

| Zone | Use | Temperature |
|---|---|---|
| A | Fast-moving parcels | room temperature |
| C | Chilled food | 2 to 8 degrees |

## Stock counts

A full stock count is done every quarter; damaged items are written off weekly.
"""

SHIPPING = """Shipping options

Standard delivery takes 3 to 5 working days and is free above 50 euros.

Express delivery arrives the next working day if ordered before 2 pm.
"""


@pytest.fixture(autouse=True)
def local_vectors(monkeypatch):
    monkeypatch.setenv("AIDS_EMBED_PROVIDER", "lsa")
    use_provider(None)
    yield
    use_provider(None)


@pytest.fixture(scope="module", autouse=True)
def reset_after():
    yield
    reset_provider()


@pytest.fixture
def folder(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "returns_policy.md").write_text(RETURNS, encoding="utf-8")
    (docs / "warehouse.md").write_text(WAREHOUSE, encoding="utf-8")
    (docs / "shipping.txt").write_text(SHIPPING, encoding="utf-8")
    return docs


@pytest.fixture
def kb(folder, tmp_path):
    return KnowledgeBase(paths=[folder], cache=tmp_path / "cache" / "index.joblib", uploads=folder / "uploads")


def test_chunks_keep_heading_paths_and_whole_tables():
    chunks = chunk_markdown(WAREHOUSE, "shop/warehouse.md")
    zones = next(c for c in chunks if "Storage zones" in c.headings)
    assert zones.section == "WH-07: Warehouse storage > Storage zones"
    assert "| C | Chilled food | 2 to 8 degrees |" in zones.text and "| A |" in zones.text
    assert chunk_markdown(WAREHOUSE, "shop/warehouse.md")[0].id == chunks[0].id


def test_hybrid_search_finds_the_answering_passage(folder):
    index = build_index([folder])
    assert index.embedder_name == "lsa" and len(index.files) == 3

    top = index.search("how long do refunds take", k=3)[0]
    assert "Refunds" in top.chunk.section and top.keyword_rank == 1

    assert "Storage zones" in index.search("which zone is chilled", k=3)[0].chunk.section
    assert index.search("WH-07 stock count", k=3)[0].chunk.title.startswith("WH-07")
    assert index.search("express delivery", k=1)[0].chunk.source.endswith("shipping.txt")


def test_extractive_answer_cites_and_refuses(folder):
    index = build_index([folder])
    question = "within how many days can items be returned"
    answer = extractive_answer(question, index.search(question, 6), index.embedder_name, index.bm25.idf)
    assert answer.covered and "30 days" in answer.text and "[1]" in answer.text
    assert answer.sources[0].hit.chunk.section.endswith("Time limit")

    off_topic = "what is the capital of France"
    refused = extractive_answer(off_topic, index.search(off_topic, 6), index.embedder_name, index.bm25.idf)
    assert not refused.covered and refused.text == NOT_COVERED


def test_model_answers_are_checked_against_their_sources(folder):
    index = build_index([folder])
    question = "within how many days can items be returned"
    hits = index.search(question, 6)

    def ask(reply):
        return model_answer(question, hits, ScriptedProvider([reply]), index.embedder_name, index.bm25.idf)

    good = ask("Items can be returned within 30 days of delivery [1]. Sale items have 14 days [1].")
    assert good.method == "language model" and good.grounded == 1.0 and [s.number for s in good.sources] == [1]

    invented = ask("Items can be returned within 45 days of delivery [1].")
    assert invented.method == "extractive" and "45" in invented.problems[0] and invented.note

    uncited = ask("Items can be returned within 30 days of delivery.")
    assert uncited.method == "extractive" and "No citation" in uncited.problems[0]

    wrong_source = ask("Items can be returned within 30 days of delivery [9].")
    assert wrong_source.method == "extractive" and "not retrieved" in wrong_source.problems[0]

    declined = ask(NOT_COVERED)
    assert not declined.covered


def test_saved_index_is_reused_and_rebuilt_when_files_change(kb, folder):
    first = kb.index()
    assert KnowledgeBase(paths=kb.paths, cache=kb.cache).index().built_at == first.built_at

    time.sleep(0.05)
    (folder / "shipping.txt").write_text(SHIPPING + "\nCollection points keep parcels for 7 days.\n", encoding="utf-8")
    refreshed = kb.refresh()
    assert refreshed.fingerprint != first.fingerprint
    assert "7 days" in kb.ask("how long do collection points keep parcels").text


def test_added_documents_are_validated_and_searchable(kb):
    with pytest.raises(ValueError, match="Only .md and .txt"):
        kb.add_document("notes.pdf", b"%PDF")
    name = kb.add_document("gift wrap.md", b"# Gift wrapping\n\nGift wrapping costs 3 euros per item.\n")
    assert name == "gift_wrap.md"
    assert "3 euros" in kb.ask("how much does gift wrapping cost").text


def test_background_answer_job(kb):
    job_id = kb.start_ask("how long do refunds take", use_model=False)
    while kb.job(job_id)["status"] == "running":
        time.sleep(0.05)
    job = kb.job(job_id)
    assert job["status"] == "done" and "5 working days" in job["result"].text


def test_unreachable_embedding_model_falls_back_to_lsa(folder):
    index = build_index([folder], embedder=OllamaEmbedder(host="http://127.0.0.1:9"))
    assert index.embedder_name == "lsa" and "LSA" in index.notes[0]


def test_document_api(kb, monkeypatch):
    import api.main as api

    monkeypatch.setattr(api, "knowledge", kb)
    client = TestClient(api.app)

    assert client.get("/documents").json()["files"] == 3
    hits = client.post("/documents/search", json={"query": "chilled zone", "k": 2}).json()
    assert len(hits) == 2 and "Storage zones" in hits[0]["section"]

    answer = client.post("/documents/ask", json={"question": "how long do refunds take"}).json()
    assert answer["covered"] and answer["sources"][0]["number"] == 1 and "5 working days" in answer["text"]

    added = client.post("/documents", files={"file": ("faq.md", b"# FAQ\n\nGift cards never expire.\n", "text/markdown")}).json()
    assert added["files"] == 4
    assert client.post("/documents", files={"file": ("x.exe", b"MZ", "application/octet-stream")}).status_code == 400


def test_retrieval_evaluation_on_the_knowledge_folder():
    from evals.runner import report, run_documents

    results = run_documents()
    plain = [r for r in results if not r.needs_model]
    assert len(plain) >= 15
    assert all(r.passed for r in plain), report(results, "rules only")
