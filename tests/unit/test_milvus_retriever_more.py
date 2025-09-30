"""
Additional tests to raise coverage for MilvusRetriever: RRF fallback and update_document.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from askme.retriever.base import Document, HybridSearchParams, RetrievalResult
from askme.retriever.milvus_retriever import MilvusRetriever


def _mk_res(doc_id: str, score: float, method: str, rank: int) -> RetrievalResult:
    return RetrievalResult(
        document=Document(id=doc_id, content=f"c-{doc_id}", metadata={}),
        score=score,
        rank=rank,
        retrieval_method=method,
    )


@pytest.mark.asyncio
async def test_hybrid_search_rrf_forces_fallback_when_native_unavailable() -> None:
    cfg = {"collection_name": "c"}
    r = MilvusRetriever(cfg)
    r.collection = MagicMock()  # pretend connected
    r.has_sparse_vector = True  # enable sparse branch for fallback

    # Force native path to be skipped
    with patch("askme.retriever.milvus_retriever.HYBRID_SEARCH_AVAILABLE", False):
        # Patch separate searches used by fallback
        r.dense_search = AsyncMock(return_value=[_mk_res("d1", 0.9, "dense", 1)])  # type: ignore
        r.sparse_search = AsyncMock(return_value=[_mk_res("s1", 0.8, "sparse", 1)])  # type: ignore

        out = await r.hybrid_search(
            [0.1], {1: 0.5}, HybridSearchParams(use_rrf=True, rrf_k=10, topk=5)
        )

        assert out and all(isinstance(x, RetrievalResult) for x in out)
        # Fallback marks retrieval_method accordingly
        assert any(x.retrieval_method == "hybrid_rrf_fallback" for x in out)


@pytest.mark.asyncio
async def test_update_document_success_path() -> None:
    cfg = {"collection_name": "c"}
    r = MilvusRetriever(cfg)

    doc = Document(
        id="u1", content="x", metadata={}, embedding=[0.1], sparse_embedding={0: 1.0}
    )

    with patch.object(r, "delete_document", return_value=True) as pdel:
        with patch.object(r, "insert_documents", return_value=["u1"]) as pins:
            ok = await r.update_document("u1", doc)
            assert ok is True
            pdel.assert_called_once_with("u1")
            pins.assert_called_once()
