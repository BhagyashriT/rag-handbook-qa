"""Shared fixtures: build a tiny real index in a tmp dir and serve it."""

import os

import pytest

from rag_handbook_qa.api import create_app
from rag_handbook_qa.config import Settings
from rag_handbook_qa.ingest import ingest


@pytest.fixture()
def app_client(tmp_path):
    corpus = str(__import__("pathlib").Path(__file__).parent / "fixtures")
    index_path = str(tmp_path / "chunks.json")
    n_files, n_chunks = ingest(corpus, index_path)
    assert n_files == 2 and n_chunks >= 4

    settings = Settings(index_path=index_path, llm_provider="stub")
    app = create_app(settings)
    # Lifespan does not run under TestClient without a context manager in
    # older starlette; use the context manager form to be explicit.
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        yield client


@pytest.fixture()
def no_index_client(tmp_path):
    settings = Settings(
        index_path=str(tmp_path / "missing.json"), llm_provider="stub"
    )
    from fastapi.testclient import TestClient

    with TestClient(create_app(settings)) as client:
        yield client
