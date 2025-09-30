"""
More tests for MilvusRetriever to push coverage over 85%.
"""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from askme.retriever.base import Document, HybridSearchParams, RetrievalResult
from askme.retriever.milvus_retriever import MilvusRetriever


@pytest.mark.asyncio
async def test_sparse_search_with_filters() -> None:
    r = MilvusRetriever({"collection_name": "c"})
    r.collection = MagicMock()
    r.has_sparse_vector = True  # ensure sparse path is available for the test
    # one hit
    hit = MagicMock()
    hit.entity.get.side_effect = lambda k, d=None: {
        "id": "i",
        "content": "c",
        "metadata": {},
    }.get(k, d)
    hit.score = 1.2
    r.collection.search.return_value = [[hit]]
    out = await r.sparse_search({1: 0.5}, topk=3, filters={"tag": ["a", "b"]})
    assert out and out[0].retrieval_method == "sparse"


@pytest.mark.asyncio
async def test_hybrid_search_rrf_native_error_fallback() -> None:
    r = MilvusRetriever({"collection_name": "c"})
    r.collection = MagicMock()
    r.has_sparse_vector = True  # allow hybrid path to use sparse fallback
    # Force native path but make hybrid_search raise so it falls back
    with patch("askme.retriever.milvus_retriever.HYBRID_SEARCH_AVAILABLE", True):
        r.collection.hybrid_search.side_effect = RuntimeError("boom")
        cast(Any, r).dense_search = AsyncMock(
            return_value=[
                RetrievalResult(
                    document=Document(id="d", content="", metadata={}),
                    score=0.9,
                    rank=1,
                    retrieval_method="dense",
                )
            ]
        )
        cast(Any, r).sparse_search = AsyncMock(
            return_value=[
                RetrievalResult(
                    document=Document(id="s", content="", metadata={}),
                    score=0.8,
                    rank=1,
                    retrieval_method="sparse",
                )
            ]
        )
        res = await r.hybrid_search(
            [0.1], {0: 1.0}, HybridSearchParams(use_rrf=True, rrf_k=50, topk=2)
        )
        assert res and any(x.retrieval_method.startswith("hybrid_rrf_") for x in res)


@pytest.mark.asyncio
async def test_create_collection_when_exists_early_return() -> None:
    r = MilvusRetriever({"collection_name": "c"})
    with patch("askme.retriever.milvus_retriever.utility") as u:
        with patch("askme.retriever.milvus_retriever.connections"):
            u.has_collection.return_value = True
            # Should early-return without creating
            await r.create_collection(128)


@pytest.mark.asyncio
async def test_disconnect_releases_collection() -> None:
    r = MilvusRetriever({"collection_name": "c"})
    r.collection = MagicMock()
    r.has_sparse_vector = True  # exercise sparse error path
    with patch("askme.retriever.milvus_retriever.connections") as c:
        await r.disconnect()
        r.collection.release.assert_called_once()
        c.disconnect.assert_called_once()


@pytest.mark.asyncio
async def test_get_collection_stats_includes_indexes() -> None:
    r = MilvusRetriever({"collection_name": "c"})
    col = MagicMock()
    col.num_entities = 10
    # stub index objects
    idx = MagicMock()
    idx.field_name = "dense_vector"
    idx.index_name = "hnsw_idx"
    idx.params = {"M": 16}
    col.indexes = [idx]
    r.collection = col
    stats = await r.get_collection_stats()
    assert stats["indexes"][0]["field_name"] == "dense_vector"


@pytest.mark.asyncio
async def test_operations_without_collection_errors() -> None:
    r = MilvusRetriever({"collection_name": "c"})
    with pytest.raises(RuntimeError):
        await r.dense_search([0.1])
    with pytest.raises(RuntimeError):
        await r.sparse_search({0: 1.0})
    with pytest.raises(RuntimeError):
        await r.get_document("x")


@pytest.mark.asyncio
async def test_connect_failure_logs_and_raises() -> None:
    r = MilvusRetriever({"collection_name": "c"})
    with patch("askme.retriever.milvus_retriever.connections") as c:
        c.connect.side_effect = RuntimeError("bad")
        with pytest.raises(RuntimeError):
            await r.connect()


@pytest.mark.asyncio
async def test_dense_sparse_and_hybrid_exceptions_propagate() -> None:
    r = MilvusRetriever({"collection_name": "c"})
    r.collection = MagicMock()
    r.has_sparse_vector = True  # make sparse search exercise the error path
    r.collection.search.side_effect = RuntimeError("search fail")
    with pytest.raises(RuntimeError):
        await r.dense_search([0.1])
    # reset side effect for sparse
    r.collection.search.side_effect = RuntimeError("search fail")
    with pytest.raises(RuntimeError):
        await r.sparse_search({0: 1.0})
    # hybrid path: make AnnSearchRequest available but hybrid_search raise
    with patch("askme.retriever.milvus_retriever.HYBRID_SEARCH_AVAILABLE", True):
        r.collection.hybrid_search.side_effect = RuntimeError("hs fail")
        # Patch fallback search to also raise to hit raise path
        with patch.object(
            r, "_hybrid_search_rrf_fallback", side_effect=RuntimeError("fallback fail")
        ):
            with pytest.raises(RuntimeError):
                await r.hybrid_search([0.1], {0: 1.0}, HybridSearchParams(use_rrf=True))
