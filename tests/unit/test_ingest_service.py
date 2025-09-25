from typing import Any

"""
Unit tests for IngestionService covering initialization, ingest flows,
batching, stats, cancellation and cleanup helpers.
"""

import asyncio
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from askme.core.config import Settings
from askme.ingest.ingest_service import (
    IngestionService,
    IngestionStats,
    IngestionTask,
    TaskStatus,
)
from askme.retriever.base import Document


@pytest.fixture
def svc(monkeypatch: Any) -> Any:
    settings = Settings()
    # Keep batches small to exercise batching code
    settings.performance.batch.embedding_batch_size = 2

    # Mock vector retriever with async methods
    retriever = MagicMock()
    retriever.connect = AsyncMock()
    retriever.disconnect = AsyncMock()
    retriever.create_collection = AsyncMock()
    retriever.insert_documents = AsyncMock(return_value=["id1", "id2"])  # ids
    retriever.get_collection_stats = AsyncMock(return_value={"num_entities": 3})
    retriever.delete_document = AsyncMock(return_value=True)

    # Mock embedding service manager later via replacement
    emb_svc = MagicMock()

    service = IngestionService(retriever, emb_svc, settings)

    # Replace processing pipeline with a mock
    service.processing_pipeline = MagicMock()

    # Replace embedding manager with a stub that returns vectors
    service.embedding_manager = MagicMock()
    service.embedding_manager.get_document_embeddings = AsyncMock(
        side_effect=lambda texts, **_: [
            {"dense_embedding": [0.1, 0.2], "sparse_embedding": {}} for _ in texts
        ]
    )
    # Initialize path uses embedding_manager.embedding_service.initialize()
    service.embedding_manager.embedding_service = MagicMock()
    service.embedding_manager.embedding_service.initialize = AsyncMock()

    return service


@pytest.mark.asyncio
async def test_initialize_calls_dependencies(svc: Any) -> None:
    await svc.initialize()
    svc.vector_retriever.connect.assert_awaited()
    svc.embedding_manager.embedding_service.initialize.assert_awaited()
    svc.vector_retriever.create_collection.assert_awaited()


@pytest.mark.asyncio
async def test_ingest_file_happy_path(svc: Any, tmp_path: Any) -> None:
    # Create actual file for file stats collection
    test_file = tmp_path / "sample.txt"
    test_content = "This is test content for file processing"
    test_file.write_text(test_content)

    # Prepare processor to return two documents
    docs = [
        Document(id="d1", content="hello"),
        Document(id="d2", content="world"),
    ]
    svc.processing_pipeline.process_file = AsyncMock(return_value=docs)

    task_id = await svc.ingest_file(test_file)

    # Wait for background processing to complete
    await svc._running_tasks[task_id]

    task = await svc.get_task_status(task_id)
    assert task is not None
    assert task.status == TaskStatus.COMPLETED
    assert task.total_chunks == 2
    assert task.processed_files == 1
    # Test new statistics fields
    assert task.total_size_bytes == len(test_content)
    assert task.files_by_type is not None
    assert ".txt" in task.files_by_type
    assert task.files_by_type[".txt"] == 1
    assert task.processing_stages is not None
    assert "document_processing" in task.processing_stages
    assert "embedding_and_ingestion" in task.processing_stages
    svc.vector_retriever.insert_documents.assert_awaited()


@pytest.mark.asyncio
async def test_ingest_directory_happy_path(svc: Any, tmp_path: Any) -> None:
    # Emulate directory processing returning one Document
    docs = [Document(id="d3", content="alpha")]
    svc.processing_pipeline.process_directory = AsyncMock(return_value=docs)

    task_id = await svc.ingest_directory(tmp_path)
    await svc._running_tasks[task_id]

    task = await svc.get_task_status(task_id)
    assert task is not None and task.status == TaskStatus.COMPLETED
    assert task.total_chunks == 1
    assert task.processed_files >= 0


@pytest.mark.asyncio
async def test__ingest_documents_batches_and_overwrite(svc: Any) -> None:
    docs = [
        Document(id=f"d{i}", content=f"text {i}") for i in range(5)
    ]  # 3 batches of size 2,2,1

    # Exercise overwrite path
    await svc._ingest_documents(
        IngestionTask(
            task_id="t1",
            source_type="file",
            source_path="/tmp/x",
            status=TaskStatus.PROCESSING,
            created_at=datetime.utcnow(),
        ),
        docs,
        overwrite=True,
    )

    # Insert called at least once
    assert svc.vector_retriever.insert_documents.await_count >= 1


@pytest.mark.asyncio
async def test_cancel_task_and_cleanup(svc: Any) -> None:
    # Create a pending task
    async def sleeper() -> Any:
        await asyncio.sleep(0.1)

    t = asyncio.create_task(sleeper())
    svc._running_tasks["tid"] = t
    svc._tasks["tid"] = IngestionTask(
        task_id="tid",
        source_type="file",
        source_path="/tmp/x",
        status=TaskStatus.PROCESSING,
        created_at=datetime.utcnow(),
    )

    cancelled = await svc.cancel_task("tid")
    assert cancelled is True


@pytest.mark.asyncio
async def test_get_ingestion_stats_and_cleanup_completed(svc: Any) -> None:
    # Populate tasks representing completed and failed ones
    now = datetime.utcnow()
    svc._tasks = {
        "a": IngestionTask(
            task_id="a",
            source_type="file",
            source_path="/tmp/a",
            status=TaskStatus.COMPLETED,
            created_at=now - timedelta(hours=48),
            started_at=now - timedelta(hours=48, minutes=5),
            completed_at=now - timedelta(hours=48, minutes=1),
            processed_files=2,
        ),
        "b": IngestionTask(
            task_id="b",
            source_type="dir",
            source_path="/tmp/b",
            status=TaskStatus.FAILED,
            created_at=now - timedelta(hours=2),
            error_message="boom",
        ),
    }

    stats = await svc.get_ingestion_stats()
    assert isinstance(stats, IngestionStats)
    assert stats.total_chunks == 3

    # Cleanup old tasks
    cleaned = await svc.cleanup_completed_tasks(older_than_hours=24)
    assert cleaned >= 1
