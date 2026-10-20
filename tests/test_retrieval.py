import numpy as np

from rag_handbook_qa.chunking import Chunk
from rag_handbook_qa.embeddings import TfidfEmbedder, make_embedder
from rag_handbook_qa.retrieve import top_k


def _chunks():
    return [
        Chunk("a#0", "concurrency.md", "Deadlock checklist", "deadlock mutual exclusion circular wait locks"),
        Chunk("b#0", "caching.md", "Four patterns", "cache aside read through write behind thundering herd"),
    ]


def test_tfidf_retrieval_ranks_relevant_chunk_first():
    chunks = _chunks()
    emb = TfidfEmbedder()
    emb.fit([c.text for c in chunks])
    hits = top_k(emb.embed_query("how do I avoid deadlock with locks?"), emb.embed([c.text for c in chunks]), chunks, 2)
    assert hits[0][0].file == "concurrency.md"
    assert 0.0 < hits[0][1] <= 1.0


def test_query_with_no_vocabulary_overlap_returns_nothing():
    chunks = _chunks()
    emb = TfidfEmbedder()
    emb.fit([c.text for c in chunks])
    hits = top_k(emb.embed_query("zyxwqv bloop"), emb.embed([c.text for c in chunks]), chunks, 2)
    assert hits == []


def test_scores_descend():
    chunks = _chunks()
    emb = TfidfEmbedder()
    emb.fit([c.text for c in chunks])
    hits = top_k(emb.embed_query("cache locks"), emb.embed([c.text for c in chunks]), chunks, 2)
    scores = [s for _, s in hits]
    assert scores == sorted(scores, reverse=True)


def test_unknown_embedder_name_rejected():
    try:
        make_embedder("nope")
    except ValueError:
        return
    raise AssertionError("expected ValueError")
