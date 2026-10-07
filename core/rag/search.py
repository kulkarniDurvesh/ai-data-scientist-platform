"""
Retrieval: BM25 keyword search, vector search, and their fusion.

    keyword  BM25 over content words of heading path + text
    vector   cosine similarity of embeddings (Ollama model or local LSA)
    hybrid   reciprocal rank fusion of both (default): exact terms such as
             "SOP-FF-01" or "class A" and paraphrases both find their passage
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

import numpy as np

from .documents import Chunk, words

RRF_K = 60
K1, B = 1.5, 0.75


class BM25:
    def __init__(self, texts: list[str]):
        self.docs = [Counter(words(t)) for t in texts]
        self.lengths = np.array([sum(d.values()) for d in self.docs], dtype=float)
        self.average = float(self.lengths.mean()) if len(self.docs) else 0.0
        frequency = Counter(term for doc in self.docs for term in doc)
        n = len(self.docs)
        self.idf = {term: math.log(1 + (n - df + 0.5) / (df + 0.5)) for term, df in frequency.items()}

    def scores(self, query: str) -> np.ndarray:
        terms = words(query)
        result = np.zeros(len(self.docs))
        for index, doc in enumerate(self.docs):
            norm = K1 * (1 - B + B * self.lengths[index] / (self.average or 1))
            for term in terms:
                tf = doc.get(term, 0)
                if tf:
                    result[index] += self.idf.get(term, 0) * tf * (K1 + 1) / (tf + norm)
        return result

    def matched_terms(self, query: str, index: int) -> list[str]:
        return [t for t in dict.fromkeys(words(query)) if t in self.docs[index]]


@dataclass
class Hit:
    chunk: Chunk
    score: float                 # fused score (higher is better)
    keyword_rank: int | None     # 1-based, None when not matched
    vector_rank: int | None
    keyword_score: float
    similarity: float

    def to_dict(self) -> dict:
        return {
            **self.chunk.to_dict(),
            "score": round(self.score, 5),
            "keyword_rank": self.keyword_rank,
            "vector_rank": self.vector_rank,
            "similarity": round(self.similarity, 4),
        }


def rank(values: np.ndarray, positive_only: bool = True) -> dict[int, int]:
    order = np.argsort(-values, kind="stable")
    ranked = [i for i in order if not positive_only or values[i] > 0]
    return {int(i): r + 1 for r, i in enumerate(ranked)}


def fuse(chunks: list[Chunk], keyword: np.ndarray, similarity: np.ndarray | None, k: int, mode: str) -> list[Hit]:
    keyword_ranks = rank(keyword)
    vector_ranks = rank(similarity, positive_only=False) if similarity is not None else {}

    scores = np.zeros(len(chunks))
    if mode in ("hybrid", "keyword"):
        for i, r in keyword_ranks.items():
            scores[i] += 1 / (RRF_K + r)
    if mode in ("hybrid", "vector") and similarity is not None:
        for i, r in vector_ranks.items():
            scores[i] += 1 / (RRF_K + r)

    order = [i for i in np.argsort(-scores, kind="stable") if scores[i] > 0][:k]
    return [
        Hit(chunks[i], float(scores[i]), keyword_ranks.get(int(i)), vector_ranks.get(int(i)),
            float(keyword[i]), float(similarity[i]) if similarity is not None else 0.0)
        for i in order
    ]
