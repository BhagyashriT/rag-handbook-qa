"""Cosine-similarity top-k retrieval over the chunk matrix."""

from __future__ import annotations

import numpy as np

from .chunking import Chunk


def top_k(
    query_vec: np.ndarray, doc_matrix, chunks: list[Chunk], k: int
) -> list[tuple[Chunk, float]]:
    """Return the k highest cosine-similarity (chunk, score) pairs, desc."""
    if hasattr(query_vec, "todense"):  # scipy sparse row
        q = np.asarray(query_vec.todense()).ravel()
    else:
        q = np.asarray(query_vec).ravel()
    q_norm = float(np.linalg.norm(q))
    if q_norm == 0.0:
        return []
    q = q / q_norm

    if hasattr(doc_matrix, "toarray"):  # scipy sparse
        sims = np.asarray(doc_matrix @ q).ravel()
    else:
        d = np.asarray(doc_matrix, dtype=np.float64)
        norms = np.linalg.norm(d, axis=1)
        norms[norms == 0.0] = 1.0
        sims = (d / norms[:, None]) @ q

    order = np.argsort(-sims, kind="stable")[: max(k, 0)]
    return [(chunks[i], float(sims[i])) for i in order if sims[i] > 0.0]
