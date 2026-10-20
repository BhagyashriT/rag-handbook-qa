"""Pluggable embeddings.

Default is TF-IDF: no heavy dependencies, deterministic, and fast enough to
rebuild at startup on a corpus this size (see trade-offs in the README).
A sentence-transformers backend is available as an opt-in extra when the
``sentence-transformers`` package is installed.
"""

from __future__ import annotations

import numpy as np


class Embedder:
    """Minimal interface every embedding backend implements."""

    name: str = "base"

    def fit(self, texts: list[str]) -> None:
        raise NotImplementedError

    def embed(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


class TfidfEmbedder(Embedder):
    """Lexical baseline. Honest trade-off: no semantic understanding, but
    zero model downloads, deterministic, and deployable on any free tier."""

    name = "tfidf"

    def __init__(self, max_features: int = 20000) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=max_features,
            sublinear_tf=True,
        )

    def fit(self, texts: list[str]) -> None:
        self._vectorizer.fit(texts)

    def embed(self, texts: list[str]):
        return self._vectorizer.transform(texts)


class SentenceTransformerEmbedder(Embedder):
    """Dense semantic embeddings. Opt-in: requires ``pip install
    sentence-transformers`` and a one-time model download (~90MB)."""

    name = "sentence-transformer"

    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. "
                "Run: pip install sentence-transformers"
            ) from exc
        self._model = SentenceTransformer(model_name)
        self.model_name = model_name

    def fit(self, texts: list[str]) -> None:
        # No fitting needed for a pretrained model; kept for interface parity.
        return None

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            self._model.encode(texts, normalize_embeddings=True), dtype=np.float32
        )


def make_embedder(name: str, st_model: str = "") -> Embedder:
    if name == "tfidf":
        return TfidfEmbedder()
    if name == "sentence-transformer":
        return SentenceTransformerEmbedder(
            st_model or "sentence-transformers/all-MiniLM-L6-v2"
        )
    raise ValueError(f"unknown embedding backend: {name!r}")
