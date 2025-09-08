"""
Tests for generation module.
"""

import pytest

from askme.core.config import GenerationConfig
from askme.generation.generator import Passage, SimpleTemplateGenerator


class TestSimpleTemplateGenerator:
    @pytest.mark.asyncio
    async def test_generate_with_passages(self) -> None:
        cfg = GenerationConfig(provider="simple", max_tokens=200)
        gen = SimpleTemplateGenerator(cfg)

        passages = [
            Passage(doc_id="doc1", title="Title 1", content="content 1", score=0.9),
            Passage(doc_id="doc2", title="Title 2", content="content 2", score=0.8),
        ]

        out = await gen.generate("What is RAG?", passages)
        assert "Question: What is RAG?" in out
        assert "[doc1: Title 1]" in out
        assert "Sources:" in out

    @pytest.mark.asyncio
    async def test_generate_without_passages(self) -> None:
        cfg = GenerationConfig(provider="simple", max_tokens=200)
        gen = SimpleTemplateGenerator(cfg)
        out = await gen.generate("What is RAG?", [])
        assert "could not find relevant" in out.lower()
