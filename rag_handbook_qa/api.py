"""FastAPI service.

Contract (frozen — an external eval harness targets this shape):
  GET  /health -> {"status": "ok", "index_loaded": bool, "chunks": int,
                   "embedder": str, "llm": str}
  POST /query  {"question": "...", "top_k": 4?}
        -> {"answer": "...",
            "sources": [{"file": "...", "section": "...", "score": 0.1234}]}
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, field_validator

from .chunking import Chunk
from .config import Settings
from .embeddings import Embedder
from .llm import ContextChunk, make_llm
from .retrieve import top_k
from .store import load_index


class QueryRequest(BaseModel):
    question: str
    top_k: int = Field(default=4, ge=1, le=10)

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("question must not be blank")
        return v.strip()


class Source(BaseModel):
    file: str
    section: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]


class HealthResponse(BaseModel):
    status: str
    index_loaded: bool
    chunks: int
    embedder: str
    llm: str


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    state: dict[str, Any] = {
        "chunks": [],
        "embedder": None,
        "matrix": None,
        "llm": None,
        "settings": settings,
    }

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        import os

        if os.path.exists(settings.index_path):
            chunks, embedder = load_index(settings.index_path, settings.st_model)
            state["chunks"] = chunks
            state["embedder"] = embedder
            state["matrix"] = embedder.embed([c.text for c in chunks])
        state["llm"] = make_llm(settings)
        yield

    app = FastAPI(title="rag-handbook-qa", version="0.1.0", lifespan=lifespan)
    app.state.rag = state

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index():
        return """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>rag-handbook-qa</title>
<style>body{font-family:system-ui,sans-serif;max-width:640px;margin:3rem auto;
padding:0 1rem;line-height:1.6;color:#1a1a1a}
code{background:#f2f2f2;padding:.15rem .4rem;border-radius:4px}
pre{background:#f2f2f2;padding:1rem;border-radius:6px;overflow-x:auto}</style>
</head><body>
<h1>rag-handbook-qa</h1>
<p>RAG Q&amp;A over my Java backend interview-handbook notes
(220 chunks). Ask a question, get an answer quoted from the notes
with file/section citations.</p>
<ul>
<li><a href="/docs">Interactive API docs</a> (try <code>POST /query</code> there)</li>
<li><a href="/health">Health check</a></li>
</ul>
<h2>Example</h2>
<pre>curl -X POST /query \\
  -H 'Content-Type: application/json' \\
  -d '{"question": "what is a deadlock?"}'</pre>
<p><a href="https://github.com/BhagyashriT/rag-handbook-qa">Source on GitHub</a></p>
</body></html>"""

    @app.get("/health", response_model=HealthResponse)
    def health():
        embedder: Embedder | None = state["embedder"]
        return HealthResponse(
            status="ok",
            index_loaded=embedder is not None,
            chunks=len(state["chunks"]),
            embedder=embedder.name if embedder else "none",
            llm=state["llm"].name if state["llm"] else "none",
        )

    @app.post("/query", response_model=QueryResponse)
    def query(req: QueryRequest):
        embedder: Embedder | None = state["embedder"]
        if embedder is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "index not built yet — run: python -m rag_handbook_qa.ingest "
                    f"--corpus <handbook-dir> --out {settings.index_path}"
                ),
            )
        chunks: list[Chunk] = state["chunks"]
        q = embedder.embed_query(req.question)
        hits = top_k(q, state["matrix"], chunks, req.top_k)
        if not hits:
            return QueryResponse(
                answer="No matching sections found in the handbook notes.",
                sources=[],
            )
        contexts = [
            ContextChunk(file=c.file, section=c.section, text=c.text, score=s)
            for c, s in hits
        ]
        answer = state["llm"].generate(req.question, contexts)
        return QueryResponse(
            answer=answer,
            sources=[
                Source(file=c.file, section=c.section, score=round(c.score, 4))
                for c in contexts
            ],
        )

    return app


app = create_app()
