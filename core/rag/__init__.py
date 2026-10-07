"""Retrieval-augmented answers over documents, with citations (Phase 7)."""

from .answer import NOT_COVERED, DocumentAnswer, Source, check_answer, extractive_answer, model_answer
from .documents import Chunk, chunk_markdown, load_corpus, words
from .embed import LsaEmbedder, OllamaEmbedder, get_embedder
from .index import KnowledgeIndex, build_index, load_or_build
from .search import BM25, Hit

__all__ = [
    "BM25",
    "NOT_COVERED",
    "Chunk",
    "DocumentAnswer",
    "Hit",
    "KnowledgeIndex",
    "LsaEmbedder",
    "OllamaEmbedder",
    "Source",
    "build_index",
    "check_answer",
    "chunk_markdown",
    "extractive_answer",
    "get_embedder",
    "load_corpus",
    "load_or_build",
    "model_answer",
    "words",
]
