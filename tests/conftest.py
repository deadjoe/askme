"""
Pytest configuration and shared fixtures.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from askme.core.config import Settings
from askme.retriever.base import Document, RetrievalResult


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    return Settings(
        vector_backend="weaviate",
        database={"host": "localhost", "port": 19530},
        embedding={"model": "BAAI/bge-m3", "dimension": 1024},
        rerank={"local_enabled": True, "local_model": "BAAI/bge-reranker-v2.5"},
        llm={"provider": "local", "model": "llama2"},
    )


@pytest.fixture
def sample_document():
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
def sample_retrieval_result():
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
def mock_vector_retriever():
    """Mock vector retriever for testing."""
    mock = AsyncMock()
    mock.connect.return_value = None
    mock.disconnect.return_value = None
    mock.create_collection.return_value = None
    mock.insert_documents.return_value = ["doc_001", "doc_002"]
    return mock


@pytest.fixture
def mock_embedding_service():
    """Mock embedding service for testing."""
    mock = AsyncMock()
    mock.initialize.return_value = None
    mock.get_dense_embedding.return_value = [0.1] * 1024
    mock.get_sparse_embedding.return_value = {1: 0.5, 5: 0.3}
    mock.cleanup.return_value = None
    return mock
