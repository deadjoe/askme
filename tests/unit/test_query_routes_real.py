"""
Additional tests for askme.api.routes.query covering the real execution path
with app.state services present to raise coverage beyond fallback branches.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from askme.api.routes.query import (
    QueryRequest,
    RetrievalRequest,
    query_documents,
    retrieve_documents,
)
from askme.core.config import Settings
from askme.retriever.base import Document, RetrievalResult


class _Req:
    def __init__(self) -> None:
        self.app = SimpleNamespace(state=SimpleNamespace())


def _mk_results(ids: List[str]) -> List[RetrievalResult]:
    out: List[RetrievalResult] = []
    for i, did in enumerate(ids, 1):
        out.append(
            RetrievalResult(
                document=Document(
                    id=did, content=f"content {did}", metadata={"title": did}
                ),
                score=1.0 - i * 0.01,
                rank=i,
                retrieval_method="hybrid",
            )
        )
    return out


@pytest.mark.asyncio
async def test_query_documents_real_path_debug_alpha() -> None:
    # Build request with services on app.state
    req = _Req()

    class _Emb:
        async def encode_query(self, q: str) -> Dict[str, Any]:  # type: ignore[override]
            return {"dense_embedding": [0.2, 0.3], "sparse_embedding": {}}

    class _Ret:
        async def hybrid_search(self, *_args, **_kwargs):  # type: ignore[override]
            return _mk_results(["h1", "h2"])  # main results

        async def dense_search(self, *_args, **_kwargs):  # type: ignore[override]
            return _mk_results(["d1", "d2"])  # for debug stats

        async def sparse_search(self, *_args, **_kwargs):  # type: ignore[override]
            return []  # trigger alpha-extreme fallback path

    class _RR:
        async def rerank(self, *_args, **_kwargs):  # type: ignore[override]
            doc1 = Document(id="h1", content="content h1", metadata={"title": "h1"})
            doc2 = Document(id="h2", content="content h2", metadata={"title": "h2"})
            return [
                SimpleNamespace(document=doc1, rerank_score=0.9),
                SimpleNamespace(document=doc2, rerank_score=0.7),
            ]

    class _Gen:
        async def generate(self, *_args, **_kwargs):  # type: ignore[override]
            return "generated answer"

    req.app.state.embedding_service = _Emb()
    req.app.state.retriever = _Ret()
    req.app.state.reranking_service = _RR()
    req.app.state.generator = _Gen()

    settings = Settings()
    q = QueryRequest(q="what is AI?", include_debug=True, use_rrf=False, alpha=0.7)
    resp = await query_documents(q, req, settings)

    assert resp.answer == "generated answer"
    assert len(resp.citations) == 2
    assert resp.retrieval_debug is not None
    assert resp.retrieval_debug.fusion_method == "alpha"
    assert resp.retrieval_debug.rrf_k is None  # when use_rrf=False
    assert resp.retrieval_debug.alpha == 0.7


@pytest.mark.asyncio
async def test_retrieve_documents_real_path_rrf_enabled_hybrid_fallbacks() -> None:
    req = _Req()

    class _Emb:
        async def encode_query(self, q: str) -> Dict[str, Any]:  # type: ignore[override]
            return {"dense_embedding": [0.5, 0.6], "sparse_embedding": {}}

    class _Ret:
        async def hybrid_search(self, *_args, **_kwargs):  # type: ignore[override]
            return _mk_results(["x1", "x2", "x3"])  # used for main + alpha extremes

        async def dense_search(self, *_args, **_kwargs):  # type: ignore[override]
            return _mk_results(["x1"])  # some overlap

        async def sparse_search(self, *_args, **_kwargs):  # type: ignore[override]
            return []  # force fallback to alpha extremes

    req.app.state.embedding_service = _Emb()
    req.app.state.retriever = _Ret()

    settings = Settings()
    r = RetrievalRequest(
        q="test", use_rrf=True, rrf_k=70, use_hyde=True, use_rag_fusion=True
    )
    resp = await retrieve_documents(r, req, settings)

    assert len(resp.documents) > 0
    assert resp.retrieval_debug is not None
    assert resp.retrieval_debug.fusion_method == "rrf"
    assert resp.retrieval_debug.rrf_k == 70
