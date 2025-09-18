from typing import Any, Set

"""
Unit tests for FastAPI app wiring in askme.api.main
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _build_patched_app() -> FastAPI:
    # Patch heavy components referenced inside lifespan
    with (
        patch("askme.api.main.BGEEmbeddingService", autospec=True) as _emb,
        patch("askme.api.main.MilvusRetriever", autospec=True),
        patch("askme.api.main.WeaviateRetriever", autospec=True),
        patch("askme.api.main.RerankingService", autospec=True) as _rr,
        patch("askme.api.main.IngestionService", autospec=True) as _ing,
        patch("askme.api.main.SimpleTemplateGenerator", autospec=True) as _tmpl,
        patch("askme.api.main.LocalOllamaGenerator", autospec=True),
        patch("askme.api.main.OpenAIChatGenerator", autospec=True),
        patch("askme.api.main.setup_logging", autospec=True),
    ):
        # Use simple template generator by default
        _tmpl.return_value.cleanup = AsyncMock()
        _rr.return_value.initialize = AsyncMock()
        _rr.return_value.cleanup = AsyncMock()
        _ing.return_value.initialize = AsyncMock()
        _ing.return_value.shutdown = AsyncMock()
        _emb.return_value.cleanup = AsyncMock()

        # Import module and construct app
        from askme.api.main import create_app

        app = create_app()
        return app


def test_create_app_routes() -> None:
    app = _build_patched_app()
    # Verify router prefixes registered
    paths: Set[str] = {
        path
        for route in app.routes
        for path in [getattr(route, "path", None)]
        if isinstance(path, str)
    }
    assert "/health/ping" in paths or any("/health" in p for p in paths)
    assert any("/query" in p for p in paths)
    assert any("/ingest" in p for p in paths)
    assert any("/eval" in p for p in paths)


def test_lifespan_startup_skips_heavy_init(monkeypatch: Any) -> None:
    monkeypatch.setenv("ASKME_SKIP_HEAVY_INIT", "1")
    monkeypatch.setenv("ASKME_ENABLE_OLLAMA", "0")
    app = _build_patched_app()

    # Add a simple route to ensure app runs
    @app.get("/ping")
    async def _ping() -> Any:  # pragma: no cover - trivial
        return {"ok": True}

    with TestClient(app) as client:
        # Lifespan should have attached mocked services to app.state
        assert hasattr(app.state, "embedding_service")
        assert hasattr(app.state, "retriever")
        assert hasattr(app.state, "reranking_service")
        assert hasattr(app.state, "ingestion_service")
        assert hasattr(app.state, "generator")
        assert client.get("/ping").status_code == 200


def test_global_exception_handler(monkeypatch: Any) -> None:
    app = _build_patched_app()

    @app.get("/boom")
    async def boom() -> Any:  # pragma: no cover - exception path tested
        raise ValueError("kaboom")

    # Avoid server exceptions being re-raised by the test client
    with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.get("/boom")
        assert resp.status_code == 500
        data = resp.json()
        assert data.get("error") == "Internal server error"
        assert "kaboom" in data.get("detail", "")
