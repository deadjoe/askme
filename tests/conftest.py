from typing import Any

"""
Pytest configuration and shared fixtures.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from askme.core.config import Settings
from askme.retriever.base import Document, RetrievalResult


@pytest.fixture
def mock_settings() -> Any:
    """Mock settings for testing."""
    settings = Settings()
    settings.vector_backend = "weaviate"
    settings.database.host = "localhost"
    settings.database.port = 19530
    settings.embedding.model = "Qwen/Qwen3-Embedding-8B"
    settings.embedding.dimension = 4096
    settings.rerank.local_enabled = True
    settings.rerank.local_backend = "qwen_local"
    settings.rerank.local_model = "Qwen/Qwen3-Reranker-8B"
    return settings


@pytest.fixture
def sample_document() -> Any:
    """Sample document for testing."""
    return Document(
        id="doc_001",
        content="This is a sample document for testing purposes.",
        metadata={
            "source": "test.txt",
            "title": "Test Document",
            "tags": ["test", "sample"],
        },
        embedding=[0.1] * 1024,
        sparse_embedding={1: 0.5, 5: 0.3, 10: 0.8},
    )


@pytest.fixture
def sample_retrieval_result() -> Any:
    """Sample retrieval result for testing."""
    doc = Document(
        id="doc_001",
        content="Sample document content for retrieval testing.",
        metadata={"source": "test.txt"},
    )
    return RetrievalResult(
        document=doc,
        score=0.95,
        rank=1,
        retrieval_method="hybrid",
        debug_info={"fusion": "rrf"},
    )


@pytest.fixture
def mock_vector_retriever() -> Any:
    """Mock vector retriever for testing."""
    mock = AsyncMock()
    mock.connect.return_value = None
    mock.disconnect.return_value = None
    mock.create_collection.return_value = None
    mock.insert_documents.return_value = ["doc_001", "doc_002"]
    return mock


@pytest.fixture
def mock_embedding_service() -> Any:
    """Mock embedding service for testing."""
    mock = AsyncMock()
    mock.initialize.return_value = None
    mock.get_dense_embedding.return_value = [0.1] * 1024
    mock.get_sparse_embedding.return_value = {1: 0.5, 5: 0.3}
    mock.cleanup.return_value = None
    return mock
