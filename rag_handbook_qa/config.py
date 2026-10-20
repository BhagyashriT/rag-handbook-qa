"""Environment-driven configuration. No secrets are ever committed;
LLM keys are read from the environment at runtime only."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    index_path: str = "index/chunks.json"
    embedding: str = "tfidf"  # tfidf | sentence-transformer
    st_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_provider: str = "stub"  # stub | openai_compatible
    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    default_top_k: int = 4

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            index_path=os.getenv("INDEX_PATH", "index/chunks.json"),
            embedding=os.getenv("EMBEDDING", "tfidf"),
            st_model=os.getenv(
                "ST_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
            ),
            llm_provider=os.getenv("LLM_PROVIDER", "stub"),
            openai_base_url=os.getenv(
                "OPENAI_BASE_URL", "https://api.openai.com/v1"
            ),
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            default_top_k=int(os.getenv("DEFAULT_TOP_K", "4")),
        )
