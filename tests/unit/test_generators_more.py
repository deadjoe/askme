"""
Additional generator tests to raise coverage: SimpleTemplate and Ollama paths.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from askme.core.config import GenerationConfig
from askme.generation.generator import (
    LocalOllamaGenerator,
    Passage,
    SimpleTemplateGenerator,
)


@pytest.mark.asyncio
async def test_simple_template_no_passages() -> None:
    cfg = GenerationConfig(provider="simple")
    gen = SimpleTemplateGenerator(cfg)
    out = await gen.generate("what?", [])
    assert "could not find" in out


@pytest.mark.asyncio
async def test_simple_template_with_passages() -> None:
    cfg = GenerationConfig(provider="simple", max_tokens=400)
    gen = SimpleTemplateGenerator(cfg)
    passages = [
        Passage(doc_id="d1", title="t1", content="c1", score=0.9),
        Passage(doc_id="d2", title="t2", content="c2", score=0.8),
    ]
    out = await gen.generate("why?", passages)
    assert "- [d1: t1]" in out and "Sources:" in out and "Summary:" in out


@pytest.mark.asyncio
async def test_ollama_fallback_on_error(monkeypatch: Any) -> None:
    cfg = GenerationConfig(provider="ollama", ollama_endpoint="http://127.0.0.1:9")
    gen = LocalOllamaGenerator(cfg)

    class _Client:
        async def post(self, *args: Any, **kwargs: Any):
            raise RuntimeError("boom")

    async def _client_get(self) -> Any:
        return _Client()

    monkeypatch.setattr(LocalOllamaGenerator, "_client_get", _client_get, raising=False)
    out = await gen.generate(
        "q", [Passage(doc_id="d", title="t", content="c", score=1.0)]
    )
    assert "Answer (constructed" in out  # fell back to template


@pytest.mark.asyncio
async def test_ollama_success(monkeypatch: Any) -> None:
    cfg = GenerationConfig(provider="ollama", ollama_endpoint="http://localhost:11434")
    gen = LocalOllamaGenerator(cfg)

    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"response": "ok from ollama"}

    class _Client:
        async def post(self, *args: Any, **kwargs: Any):
            return _Resp()

        async def aclose(self):  # to support cleanup
            return None

    async def _client_get(self) -> Any:
        # set the private client so cleanup path closes it
        self._client = _Client()
        return self._client

    monkeypatch.setattr(LocalOllamaGenerator, "_client_get", _client_get, raising=False)
    out = await gen.generate(
        "q", [Passage(doc_id="d", title="t", content="c", score=1.0)]
    )
    assert out == "ok from ollama"
    await gen.cleanup()  # ensure no exceptions
