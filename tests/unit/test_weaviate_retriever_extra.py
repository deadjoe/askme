"""
More tests for WeaviateRetriever to push coverage: stats dict path, sync batch, get_by_id fallback, update failure, exists path.
"""

from __future__ import annotations

from typing import Any

import pytest
from unittest.mock import MagicMock, patch

from askme.retriever.base import Document
from askme.retriever.weaviate_retriever import WeaviateRetriever


@pytest.mark.asyncio
async def test_create_collection_already_exists():
    r = WeaviateRetriever({"class_name": "C"})
    client = MagicMock()
    cols = MagicMock()
    client.collections = cols
    cols.list_all.return_value = ["C"]
    cols.get.return_value = MagicMock()
    r.client = client
    await r.create_collection(128)
    assert r.collection is not None


@pytest.mark.asyncio
async def test_get_collection_stats_total_count_dict_path():
    r = WeaviateRetriever({"class_name": "C"})
    col = MagicMock()
    info = MagicMock()
    info.vector_index_config = MagicMock()
    col.config.get.return_value = info
    agg = {"total_count": 123}
    col.aggregate.over_all.return_value = agg
    async def _ensure():
        return col
    r._ensure_collection = _ensure  # type: ignore[assignment]
    out = await r.get_collection_stats()
    assert out["num_entities"] == 123


class _SyncBatch:
    def __enter__(self):  # type: ignore[override]
        return self

    def __exit__(self, exc_type, exc, tb):  # type: ignore[override]
        return False

    def add_object(self, **kwargs: Any):  # non-awaitable
        return None


class _SyncBatcher:
    def dynamic(self):
        return _SyncBatch()


@pytest.mark.asyncio
async def test_insert_documents_sync_batch_path():
    r = WeaviateRetriever({"class_name": "C"})
    class _Col:
        def __init__(self) -> None:
            self.batch = _SyncBatcher()
    c = _Col()
    async def _ensure():
        return c
    r._ensure_collection = _ensure  # type: ignore[assignment]
    ids = await r.insert_documents([
        Document(id="a", content="x", metadata={"views": 3, "flag": True, "note": "n"}, embedding=[0.1]),
    ])
    assert len(ids) == 1


@pytest.mark.asyncio
async def test_get_document_fallback_get_by_id():
    r = WeaviateRetriever({"class_name": "C"})
    col = MagicMock()
    col.query.fetch_objects.return_value = MagicMock(objects=[])
    o = MagicMock()
    o.uuid = "u"
    o.properties = {"content": "c", "doc_id": "d"}
    col.data.objects.get_by_id.return_value = o
    async def _ensure2():
        return col
    r._ensure_collection = _ensure2  # type: ignore[assignment]
    doc = await r.get_document("d")
    assert doc and doc.id == "d"


@pytest.mark.asyncio
async def test_update_document_failure_on_insert():
    r = WeaviateRetriever({"class_name": "C"})
    with patch.object(r, "delete_document", return_value=True):
        with patch.object(r, "insert_documents", side_effect=Exception("boom")):
            ok = await r.update_document("d", Document(id="d", content="c", metadata={}, embedding=[0.1]))
            assert ok is False


@pytest.mark.asyncio
async def test__ensure_collection_creates_when_get_fails():
    r = WeaviateRetriever({"class_name": "C"})
    client = MagicMock()
    cols = MagicMock()
    client.collections = cols
    cols.get.side_effect = [Exception("missing"), MagicMock()]
    r.client = client
    # Patch create_collection to set r.collection
    async def _create(dim: int):  # type: ignore[no-redef]
        r.collection = MagicMock()
        return None
    r.create_collection = _create  # type: ignore[assignment]
    col = await r._ensure_collection()
    assert col is not None


@pytest.mark.asyncio
async def test_delete_document_multiple_matches():
    r = WeaviateRetriever({"class_name": "C"})
    col = MagicMock()
    res = MagicMock()
    o1 = MagicMock(); o1.uuid = "u1"
    o2 = MagicMock(); o2.uuid = "u2"
    res.objects = [o1, o2]
    col.query.fetch_objects.return_value = res
    col.data.objects.delete_by_id.return_value = None
    async def _ensure():
        return col
    r._ensure_collection = _ensure  # type: ignore[assignment]
    ok = await r.delete_document("d")
    assert ok is True
    assert col.data.objects.delete_by_id.call_count == 2


@pytest.mark.asyncio
async def test_connect_local_error_path():
    r = WeaviateRetriever({"url": "http://localhost:8080"})
    with patch("askme.retriever.weaviate_retriever.weaviate") as w:
        w.connect_to_custom.side_effect = RuntimeError("conn fail")
        with pytest.raises(RuntimeError):
            await r.connect()


@pytest.mark.asyncio
async def test_connect_local_heuristic_grpc_port_plus_one():
    r = WeaviateRetriever({"url": "http://localhost:8081"})
    with patch("askme.retriever.weaviate_retriever.weaviate") as w:
        w.connect_to_custom.return_value = MagicMock()
        await r.connect()
        w.connect_to_custom.assert_called_once()
        kwargs = w.connect_to_custom.call_args.kwargs
        assert kwargs["grpc_port"] == 8082
