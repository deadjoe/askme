"""
Unit tests to raise coverage for askme.evals.pipeline_runner.
Covers fallback (no services), single-query path, and multi-query (HyDE+RAG-Fusion) with RRF.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from askme.core.config import Settings
from askme.evals.pipeline_runner import PipelineResult, run_pipeline_once
from askme.retriever.base import Document, RetrievalResult


class _App:
    def __init__(self) -> None:
        self.state = SimpleNamespace()


def _mk_results(ids: List[str]) -> List[RetrievalResult]:
    out: List[RetrievalResult] = []
    for i, did in enumerate(ids, 1):
        out.append(
            RetrievalResult(
                document=Document(id=did, content=f"content {did}", metadata={"title": did}),
                score=1.0 - i * 0.01,
                rank=i,
                retrieval_method="hybrid",
            )
        )
    return out


@pytest.mark.asyncio
async def test_run_pipeline_once_fallback_when_services_missing() -> None:
    app = _App()
    settings = Settings()
    result = await run_pipeline_once(app, "what is RAG?", settings)
    assert isinstance(result, PipelineResult)
    assert result.contexts == [] and result.citations == []
    assert "No services available" in result.answer


@pytest.mark.asyncio
async def test_run_pipeline_once_basic_single_query_path() -> None:
    app = _App()
    settings = Settings()
    # Disable enhancements to keep single result list
    settings.enhancer.hyde.enabled = False
    settings.enhancer.rag_fusion.enabled = False

    # Mocks
    import asyncio

    class _Emb:
        async def encode_query(self, q: str) -> Dict[str, Any]:  # type: ignore[override]
            return {"dense_embedding": [0.1, 0.2], "sparse_embedding": {}}

    class _Ret:
        async def hybrid_search(self, *_args, **_kwargs):  # type: ignore[override]
            await asyncio.sleep(0)
            return _mk_results(["d1", "d2"])  # single list path

    class _RR:
        async def rerank(self, *_args, **_kwargs):  # type: ignore[override]
            # Minimal objects with required attributes for the route code
            doc = Document(id="d1", content="content d1", metadata={"title": "d1"})
            return [SimpleNamespace(document=doc, rerank_score=0.9)]

    class _Gen:
        async def generate(self, *_args, **_kwargs):  # type: ignore[override]
            return "final answer"

    app.state.embedding_service = _Emb()
    app.state.retriever = _Ret()
    app.state.reranking_service = _RR()
    app.state.generator = _Gen()

    result = await run_pipeline_once(app, "what is RAG?", settings)
    assert result.answer == "final answer"
    assert result.contexts and result.citations
    assert result.citations[0]["doc_id"] == "d1"


@pytest.mark.asyncio
async def test_run_pipeline_once_with_hyde_and_rag_fusion_rrf() -> None:
    app = _App()
    settings = Settings()
    settings.enhancer.hyde.enabled = True
    settings.enhancer.rag_fusion.enabled = True
    settings.enhancer.rag_fusion.num_queries = 3
    settings.hybrid.mode = "rrf"

    # Mocks returning different lists to exercise RRF branch (len(result_lists) > 1)
    import asyncio

    class _Emb:
        async def encode_query(self, q: str) -> Dict[str, Any]:  # type: ignore[override]
            return {"dense_embedding": [0.1, 0.2], "sparse_embedding": {}}

    calls: List[int] = []

    class _Ret:
        async def hybrid_search(self, *_args, **_kwargs):  # type: ignore[override]
            await asyncio.sleep(0)
            calls.append(1)
            return _mk_results([f"d{len(calls)}"])  # produce different ids

    class _RR:
        async def rerank(self, *_args, **_kwargs):  # type: ignore[override]
            # Consolidate one reranked result
            doc = Document(id="d1", content="ccc", metadata={})
            return [SimpleNamespace(document=doc, rerank_score=0.8)]

    class _Gen:
        async def generate(self, *_args, **_kwargs):  # type: ignore[override]
            return "answer with rrf"

    app.state.embedding_service = _Emb()
    app.state.retriever = _Ret()
    app.state.reranking_service = _RR()
    app.state.generator = _Gen()

    result = await run_pipeline_once(app, "pipeline test", settings)
    # Should have executed multiple queries and still produce an answer
    assert result.answer == "answer with rrf"
    assert result.contexts and isinstance(result.citations, list)

