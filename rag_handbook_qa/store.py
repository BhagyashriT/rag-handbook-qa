"""File-based vector index.

The index is a single JSON file: chunk texts + metadata + embedder name.
Embeddings are rebuilt in memory at startup (fast for a corpus this size),
which keeps the index file small and human-inspectable and avoids
serializing sklearn internals.
"""

from __future__ import annotations

import json
from dataclasses import asdict

from .chunking import Chunk
from .embeddings import Embedder, make_embedder

INDEX_VERSION = 1


def save_index(path: str, chunks: list[Chunk], embedder: Embedder) -> None:
    payload = {
        "version": INDEX_VERSION,
        "embedder": embedder.name,
        "chunks": [asdict(c) for c in chunks],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)


def load_index(path: str, st_model: str = "") -> tuple[list[Chunk], Embedder]:
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    if payload.get("version") != INDEX_VERSION:
        raise ValueError(f"unsupported index version: {payload.get('version')}")
    chunks = [Chunk(**c) for c in payload["chunks"]]
    embedder = make_embedder(payload["embedder"], st_model)
    embedder.fit([c.text for c in chunks])
    return chunks, embedder
