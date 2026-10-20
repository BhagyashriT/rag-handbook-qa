"""Pluggable answer generation.

- ``stub`` (default): deterministic extractive mode. Quotes the retrieved
  sections with citations. No API key, no network, no hallucination risk
  beyond what the notes themselves say.
- ``openai_compatible``: calls any OpenAI-style ``/chat/completions``
  endpoint (OpenAI, Ollama, LM Studio, ...), instructed to answer ONLY
  from the provided context.
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass

from .config import Settings


@dataclass
class ContextChunk:
    file: str
    section: str
    text: str
    score: float


class StubLlm:
    """Deterministic extractive answers. Safe default for tests and demos."""

    name = "stub"

    def generate(self, question: str, contexts: list[ContextChunk]) -> str:
        lines = [
            "Based on the matching sections from the handbook notes "
            f"(extractive stub mode — quoting notes, not generating):"
        ]
        for c in contexts[:3]:
            excerpt = c.text[:700].strip()
            lines.append(
                f"\n### {c.section}\n"
                f"Source: `{c.file}` (retrieval score {c.score:.3f})\n\n"
                f"{excerpt}"
            )
        return "\n".join(lines)


class OpenAICompatibleLlm:
    """Answers via an OpenAI-style chat completions endpoint, grounded in
    retrieved context. Refuses to answer when the context is insufficient."""

    name = "openai_compatible"

    SYSTEM_PROMPT = (
        "You are a backend-interview study assistant. Answer the user's "
        "question using ONLY the context excerpts below. Cite the source "
        "file for each claim, like [java/concurrency.md]. If the excerpts "
        "do not contain the answer, say so explicitly instead of guessing."
    )

    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        if not api_key:
            raise RuntimeError(
                "LLM_PROVIDER=openai_compatible requires OPENAI_API_KEY "
                "in the environment (never commit it)."
            )
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model

    def generate(self, question: str, contexts: list[ContextChunk]) -> str:
        context_block = "\n\n".join(
            f"[source: {c.file} | section: {c.section}]\n{c.text}"
            for c in contexts
        )
        body = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"Context:\n{context_block}\n\nQuestion: {question}",
                    },
                ],
                "temperature": 0.2,
            }
        ).encode()
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                payload = json.loads(resp.read().decode())
        except Exception as exc:
            raise RuntimeError(f"LLM request failed: {type(exc).__name__}") from exc
        try:
            return payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise RuntimeError("LLM returned an unexpected response shape") from exc


def make_llm(settings: Settings):
    if settings.llm_provider == "openai_compatible":
        return OpenAICompatibleLlm(
            settings.openai_base_url, settings.openai_api_key, settings.openai_model
        )
    if settings.llm_provider == "stub":
        return StubLlm()
    raise ValueError(f"unknown LLM provider: {settings.llm_provider!r}")
