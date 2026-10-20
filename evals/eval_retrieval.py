"""Retrieval eval: hit-rate@k and MRR over a hand-written golden set.

Example:
    python evals/eval_retrieval.py --index index/chunks.json --golden evals/golden.json

A "hit" is file-level (the expected note file appears in the top-k); the
section check is reported separately as a stricter secondary metric.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_handbook_qa.retrieve import top_k
from rag_handbook_qa.store import load_index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="index/chunks.json")
    parser.add_argument("--golden", default="evals/golden.json")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    chunks, embedder = load_index(args.index)
    matrix = embedder.embed([c.text for c in chunks])
    golden = json.loads(Path(args.golden).read_text(encoding="utf-8"))

    file_hits = 0
    section_hits = 0
    mrr_sum = 0.0
    for i, item in enumerate(golden, 1):
        hits = top_k(embedder.embed_query(item["question"]), matrix, chunks, args.top_k)
        files = [c.file for c, _ in hits]
        sections = [c.section.lower() for c, _ in hits]
        file_hit = item["expected_file"] in files
        section_hit = any(item["expected_section"].lower() in s for s in sections)
        rank = next((r + 1 for r, f in enumerate(files) if f == item["expected_file"]), 0)
        file_hits += file_hit
        section_hits += section_hit
        mrr_sum += 1.0 / rank if rank else 0.0
        mark = "PASS" if file_hit else "FAIL"
        top = files[0] if files else "(none)"
        print(f"[{mark}] Q{i}: expected {item['expected_file']} | top hit: {top}")

    n = len(golden)
    print(f"\nfile hit-rate@{args.top_k}:    {file_hits}/{n} = {file_hits / n:.2f}")
    print(f"section hit-rate@{args.top_k}: {section_hits}/{n} = {section_hits / n:.2f}")
    print(f"MRR@{args.top_k}:                {mrr_sum / n:.3f}")


if __name__ == "__main__":
    main()
