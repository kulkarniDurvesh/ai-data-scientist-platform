"""
The knowledge base shared by the dashboard and the API (Phase 7).

One index over the knowledge folder and the project documentation,
loaded from its saved copy or rebuilt when files or the embedder change.
Uploaded documents go to knowledge/uploads/.
"""

from __future__ import annotations

import os
import re
import threading
import uuid
from pathlib import Path

from core.llm import get_provider
from core.rag import DocumentAnswer, KnowledgeIndex, extractive_answer, load_or_build, model_answer

PROJECT = Path(__file__).resolve().parent.parent
# Top-level docs only: docs/examples holds generated outputs, not documentation.
DEFAULT_PATHS = [PROJECT / "knowledge", *sorted((PROJECT / "docs").glob("*.md")), PROJECT / "README.md"]
UPLOADS = PROJECT / "knowledge" / "uploads"
CACHE = PROJECT / ".rag_index" / "index.joblib"
MAX_UPLOAD_BYTES = 2_000_000


class KnowledgeBase:
    def __init__(self, paths: list[Path] | None = None, cache: Path | None = CACHE, uploads: Path | None = UPLOADS):
        env = os.environ.get("AIDS_KNOWLEDGE_PATHS")
        self.paths = paths or ([Path(p) for p in env.split(os.pathsep)] if env else DEFAULT_PATHS)
        self.cache = cache
        self.uploads = uploads
        self._index: KnowledgeIndex | None = None
        self._lock = threading.Lock()
        self._jobs: dict[str, dict] = {}

    def index(self, rebuild: bool = False) -> KnowledgeIndex:
        with self._lock:
            if self._index is None or rebuild:
                self._index = load_or_build(self.paths, self.cache, rebuild)
            return self._index

    def refresh(self) -> KnowledgeIndex:
        """Reload if files changed since the index was built (cheap check)."""

        from core.rag import load_corpus

        with self._lock:
            if self._index is not None and load_corpus(self.paths).fingerprint != self._index.fingerprint:
                self._index = None
        return self.index()

    def status(self) -> dict:
        return self.index().status()

    def search(self, query: str, k: int = 5, mode: str = "hybrid"):
        return self.index().search(query, k, mode)

    def ask(self, question: str, use_model: bool = False) -> DocumentAnswer:
        index = self.index()
        hits = index.search(question, k=6)
        provider = get_provider() if use_model else None
        if provider is None:
            answer = extractive_answer(question, hits, index.embedder_name, index.bm25.idf)
            if use_model:
                answer.note = "No language model is switched on; showing quoted passages."
            return answer
        return model_answer(question, hits, provider, index.embedder_name, index.bm25.idf)

    def examples(self, limit: int = 4) -> list[str]:
        """Example questions: examples.txt in a knowledge folder, else built from section headings."""

        for path in self.paths:
            file = Path(path) / "examples.txt"
            if file.is_file():
                lines = [l.strip() for l in file.read_text(encoding="utf-8").splitlines() if l.strip()]
                if lines:
                    return lines[:limit]
        found, titles = [], set()
        for chunk in self.index().chunks:
            if chunk.headings and chunk.title not in titles and len(chunk.headings) > 1:
                titles.add(chunk.title)
                found.append(f"what does {chunk.title} say about {chunk.headings[-1].lower()}")
            if len(found) == limit:
                break
        return found

    def start_ask(self, question: str, use_model: bool) -> str:
        """Answer in the background (a language model can take a while)."""

        job_id = uuid.uuid4().hex[:8]
        job = {"id": job_id, "status": "running", "messages": ["Searching the documents..."], "result": None, "error": None}
        self._jobs[job_id] = job

        def run() -> None:
            try:
                if use_model:
                    job["messages"].append("Writing the answer with the language model...")
                job["result"] = self.ask(question, use_model)
                job["status"] = "done"
            except Exception as error:  # noqa: BLE001 - shown to the user
                job["error"] = str(error)
                job["status"] = "error"

        threading.Thread(target=run, daemon=True).start()
        return job_id

    def job(self, job_id: str | None) -> dict | None:
        return self._jobs.get(job_id) if job_id else None

    def add_document(self, name: str, content: bytes) -> str:
        if self.uploads is None:
            raise ValueError("Uploads are switched off.")
        suffix = Path(name).suffix.lower()
        if suffix not in (".md", ".txt"):
            raise ValueError("Only .md and .txt documents can be added.")
        if len(content) > MAX_UPLOAD_BYTES:
            raise ValueError("The document is larger than 2 MB.")
        try:
            content.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("The document must be UTF-8 text.") from error
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(name).stem)[:80] or "document"
        self.uploads.mkdir(parents=True, exist_ok=True)
        target = self.uploads / f"{safe}{suffix}"
        target.write_bytes(content)
        self.index(rebuild=True)
        return target.name


knowledge = KnowledgeBase()
