"""
Embeddings: an Ollama embedding model, or local LSA vectors as a fallback.

    AIDS_EMBED_PROVIDER   auto (default) | ollama | lsa
    AIDS_EMBED_MODEL      Ollama embedding model (default nomic-embed-text)

Embedding models are small and fast on a CPU (unlike chat models), so
"auto" uses the Ollama model when it is installed and falls back to LSA
(TF-IDF + truncated SVD, fitted on the documents) otherwise. LSA needs no
download but only knows the words in the documents.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from core.llm.provider import DEFAULT_OLLAMA_HOST, LLMError

DEFAULT_EMBED_MODEL = "nomic-embed-text"
BATCH = 32
LSA_DIMENSIONS = 128


def _normalise(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms == 0, 1, norms)


class OllamaEmbedder:
    # nomic-embed-text was trained with task prefixes.
    PREFIXES = {"nomic-embed-text": ("search_document: ", "search_query: ")}

    def __init__(self, model: str | None = None, host: str | None = None):
        self.model = model or os.environ.get("AIDS_EMBED_MODEL") or DEFAULT_EMBED_MODEL
        self.host = (host or os.environ.get("AIDS_OLLAMA_HOST") or DEFAULT_OLLAMA_HOST).rstrip("/")
        self.name = f"ollama:{self.model}"
        self.document_prefix, self.query_prefix = self.PREFIXES.get(self.model.split(":")[0], ("", ""))

    def fit(self, texts: list[str]) -> None:
        return None

    def _embed(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for start in range(0, len(texts), BATCH):
            payload = {"model": self.model, "input": texts[start: start + BATCH]}
            request = urllib.request.Request(f"{self.host}/api/embed", data=json.dumps(payload).encode(),
                                             headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(request, timeout=300) as response:
                    vectors.extend(json.loads(response.read())["embeddings"])
            except (urllib.error.URLError, TimeoutError, OSError, KeyError, ValueError) as error:
                raise LLMError(f"Embedding with '{self.model}' failed: {error}") from error
        return _normalise(np.asarray(vectors, dtype=np.float32))

    def documents(self, texts: list[str]) -> np.ndarray:
        return self._embed([self.document_prefix + t for t in texts])

    def query(self, text: str) -> np.ndarray:
        return self._embed([self.query_prefix + text])[0]

    def available(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.host}/api/tags", timeout=1.5) as response:
                names = [m.get("name", "") for m in json.loads(response.read()).get("models", [])]
        except (urllib.error.URLError, TimeoutError, OSError, ValueError):
            return False
        wanted = self.model if ":" in self.model else f"{self.model}:latest"
        return wanted in names or self.model in names


class LsaEmbedder:
    """TF-IDF + truncated SVD fitted on the documents: no model download."""

    name = "lsa"

    def __init__(self):
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1, stop_words="english")
        self.svd: TruncatedSVD | None = None

    def fit(self, texts: list[str]) -> None:
        matrix = self.vectorizer.fit_transform(texts)
        dimensions = max(1, min(LSA_DIMENSIONS, matrix.shape[0] - 1, matrix.shape[1] - 1))
        self.svd = TruncatedSVD(n_components=dimensions, random_state=0).fit(matrix)

    def _transform(self, texts: list[str]) -> np.ndarray:
        return _normalise(self.svd.transform(self.vectorizer.transform(texts)).astype(np.float32))

    def documents(self, texts: list[str]) -> np.ndarray:
        return self._transform(texts)

    def query(self, text: str) -> np.ndarray:
        return self._transform([text])[0]


def get_embedder():
    choice = os.environ.get("AIDS_EMBED_PROVIDER", "auto").strip().lower()
    if choice == "lsa":
        return LsaEmbedder()
    ollama = OllamaEmbedder()
    if choice == "ollama" or ollama.available():
        return ollama
    return LsaEmbedder()
