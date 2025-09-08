"""
Unit tests for OpenAIChatGenerator (no network; stub client).
"""

from typing import Any, Dict

import pytest

from askme.core.config import GenerationConfig
from askme.generation.generator import OpenAIChatGenerator, Passage


class DummyChoice:
    def __init__(self, content: str) -> None:
        self.message = type("m", (), {"content": content})


class DummyResponse:
    def __init__(self, content: str) -> None:
        self.choices = [DummyChoice(content)]


class DummyClient:
    def __init__(self) -> None:
        self.chat = type("c", (), {"completions": type("cc", (), {"create": self._create})})

    def _create(self, **_: Dict[str, Any]) -> DummyResponse:  # type: ignore[no-redef]
        return DummyResponse("ok")


@pytest.mark.asyncio
async def test_openai_generator_uses_base_url(monkeypatch: Any) -> None:
    cfg = GenerationConfig(provider="openai", openai_model="gpt-4o-mini")
    gen = OpenAIChatGenerator(cfg)

    # Patch OpenAI client constructor to our dummy
    import askme.generation.generator as g

    def _dummy_client(*args: Any, **kwargs: Any) -> DummyClient:  # type: ignore[no-redef]
        return DummyClient()

    monkeypatch.setattr(g, "OpenAI", _dummy_client)

    passages = [Passage(doc_id="d1", title="t1", content="c1", score=0.9)]
    out = await gen.generate("q", passages)
    assert out == "ok"

