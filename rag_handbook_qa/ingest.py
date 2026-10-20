"""Ingestion CLI: walk a handbook checkout, chunk it, and write the index.

Example:
    python -m rag_handbook_qa.ingest \\
        --corpus ~/workspace/handbook-corpus \\
        --out index/chunks.json
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .chunking import chunk_markdown
from .config import Settings
from .embeddings import make_embedder
from .store import save_index


def ingest(
    corpus_dir: str,
    out_path: str,
    max_chars: int = 1500,
    overlap: int = 200,
    embedding: str = "tfidf",
    st_model: str = "",
) -> tuple[int, int]:
    corpus = Path(corpus_dir)
    md_files = sorted(p for p in corpus.rglob("*.md") if p.is_file())
    if not md_files:
        raise SystemExit(f"no markdown files found under {corpus_dir}")

    chunks = []
    for path in md_files:
        rel = str(path.relative_to(corpus))
        text = path.read_text(encoding="utf-8")
        chunks.extend(chunk_markdown(text, rel, max_chars=max_chars, overlap=overlap))

    embedder = make_embedder(embedding, st_model)
    embedder.fit([c.text for c in chunks])  # validates the backend up front

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    save_index(out_path, chunks, embedder)
    return len(md_files), len(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest handbook markdown into a RAG index.")
    parser.add_argument("--corpus", required=True, help="handbook checkout directory")
    parser.add_argument("--out", default="index/chunks.json", help="index output path")
    parser.add_argument("--max-chars", type=int, default=1500)
    parser.add_argument("--overlap", type=int, default=200)
    parser.add_argument("--embedding", default="tfidf",
                        choices=["tfidf", "sentence-transformer"])
    args = parser.parse_args()

    settings = Settings.from_env()
    n_files, n_chunks = ingest(
        args.corpus, args.out, args.max_chars, args.overlap,
        args.embedding, st_model=settings.st_model,
    )
    print(f"ingested {n_files} files -> {n_chunks} chunks -> {args.out}")


if __name__ == "__main__":
    main()
