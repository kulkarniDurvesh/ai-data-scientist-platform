"""
The knowledge index: chunks + BM25 + vectors, saved to disk.

The index remembers a fingerprint of the files (paths, sizes, times) and
the embedder it was built with, and is rebuilt when either changes. A
few hundred chunks fit comfortably in a NumPy matrix; Azure AI Search
takes this role in the cloud (Phase 9).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np

from core.llm import LLMError

from .documents import Chunk, load_corpus
from .embed import LsaEmbedder, get_embedder
from .search import BM25, Hit, fuse

VERSION = 1


@dataclass
class KnowledgeIndex:
    chunks: list[Chunk]
    files: list[str]
    fingerprint: str
    embedder: object
    vectors: np.ndarray | None
    bm25: BM25
    built_at: float = field(default_factory=time.time)
    seconds: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def embedder_name(self) -> str:
        return getattr(self.embedder, "name", "none")

    def search(self, query: str, k: int = 5, mode: str = "hybrid") -> list[Hit]:
        if not self.chunks or not query.strip():
            return []
        keyword = self.bm25.scores(query)
        similarity = None
        if self.vectors is not None and mode in ("hybrid", "vector"):
            try:
                similarity = self.vectors @ self.embedder.query(query)
            except LLMError:
                similarity = None
        return fuse(self.chunks, keyword, similarity, k, mode)

    def status(self) -> dict:
        return {
            "files": len(self.files),
            "chunks": len(self.chunks),
            "embedder": self.embedder_name,
            "built_at": time.strftime("%Y-%m-%d %H:%M", time.localtime(self.built_at)),
            "seconds": round(self.seconds, 1),
            "notes": list(self.notes),
            "sources": list(self.files),
        }


def build_index(paths: list[Path], embedder=None) -> KnowledgeIndex:
    start = time.time()
    corpus = load_corpus(paths)
    texts = [c.searchable() for c in corpus.chunks]
    embedder = embedder or get_embedder()
    notes: list[str] = []
    vectors = None
    if texts:
        try:
            embedder.fit(texts)
            vectors = embedder.documents(texts)
        except LLMError as error:
            notes.append(f"{error} Using local LSA vectors instead.")
            embedder = LsaEmbedder()
            embedder.fit(texts)
            vectors = embedder.documents(texts)
    return KnowledgeIndex(corpus.chunks, corpus.files, corpus.fingerprint, embedder, vectors,
                          BM25(texts), seconds=time.time() - start, notes=notes)


def load_or_build(paths: list[Path], cache: Path | None, rebuild: bool = False) -> KnowledgeIndex:
    """The saved index if files and embedder are unchanged, else a fresh one (saved)."""

    fingerprint = load_corpus(paths).fingerprint
    wanted = get_embedder().name
    if cache is not None and cache.exists() and not rebuild:
        try:
            saved = joblib.load(cache)
            index = saved["index"]
            if saved.get("version") == VERSION and index.fingerprint == fingerprint and index.embedder_name == wanted:
                return index
        except Exception:  # noqa: BLE001 - a broken cache is simply rebuilt
            pass
    index = build_index(paths)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"version": VERSION, "index": index}, cache)
    return index
