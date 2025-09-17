"""
Tests the insert_documents batch path with async context manager and awaitable add_object.
"""

from __future__ import annotations

from typing import Any

import pytest

from askme.retriever.base import Document
from askme.retriever.weaviate_retriever import WeaviateRetriever


class _AsyncBatch:
    def __init__(self) -> None:
        self.add_calls = 0

    async def __aenter__(self):  # type: ignore[override]
        return self

    async def __aexit__(self, exc_type, exc, tb):  # type: ignore[override]
        return False

    async def add_object(self, **kwargs: Any):  # type: ignore[no-redef]
        self.add_calls += 1
        # simulate awaitable return
        return None


class _AsyncBatcher:
    def dynamic(self):  # returns async context manager
        return _AsyncBatch()


@pytest.mark.asyncio
async def test_insert_documents_async_batch_path() -> None:
    r = WeaviateRetriever({"class_name": "C"})

    # minimal fake collection that exposes .batch.dynamic()
    class _Col:
        def __init__(self) -> None:
            self.batch = _AsyncBatcher()

    col = _Col()

    # Patch _ensure_collection to our fake collection (async)
    async def _ensure() -> Any:
        return col

    r._ensure_collection = _ensure  # type: ignore[assignment]

    docs = [
        Document(id="a", content="x", metadata={}, embedding=[0.1] * 4),
        Document(id="b", content="y", metadata={}, embedding=[0.2] * 4),
    ]

    ids = await r.insert_documents(docs)
    assert len(ids) == 2
