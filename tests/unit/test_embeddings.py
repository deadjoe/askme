"""
Unit tests for embedding service.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from askme.core.config import EmbeddingConfig
from askme.core.embeddings import BGEEmbeddingService, EmbeddingManager


class TestBGEEmbeddingService:
    """Test BGE embedding service."""

    def test_service_creation(self):
        """Test creating embedding service."""
        config = EmbeddingConfig(model="BAAI/bge-m3", dimension=1024, batch_size=16)
        service = BGEEmbeddingService(config)

        assert service.config.model == "BAAI/bge-m3"
        assert service.config.dimension == 1024
        assert service.config.batch_size == 16
        assert service.model is None
        assert service._is_initialized is False

    @patch("askme.core.embeddings.BGEM3FlagModel")
    async def test_initialization(self, mock_model_class):
        """Test service initialization."""
        # Mock the BGE model
        mock_model = MagicMock()
        mock_model_class.return_value = mock_model

        config = EmbeddingConfig()
        service = BGEEmbeddingService(config)

        await service.initialize()

        assert service._is_initialized is True
        mock_model_class.assert_called_once()

    @patch("askme.core.embeddings.BGEM3FlagModel")
    async def test_get_dense_embedding(self, mock_model_class):
        """Test getting dense embeddings."""
        # Mock the BGE model
        mock_model = MagicMock()
        mock_model.encode.return_value = {"dense_vecs": [[0.1, 0.2, 0.3]]}
        mock_model_class.return_value = mock_model

        config = EmbeddingConfig()
        service = BGEEmbeddingService(config)
        service.model = mock_model
        service._is_initialized = True

        embedding = await service.get_dense_embedding("test text")

        assert embedding == [0.1, 0.2, 0.3]
        mock_model.encode.assert_called_once()

    @patch("askme.core.embeddings.BGEM3FlagModel")
    async def test_get_sparse_embedding(self, mock_model_class):
        """Test getting sparse embeddings."""
        # Mock the BGE model
        mock_model = MagicMock()
        mock_model.encode.return_value = {
            "lexical_weights": [{"token_1": 0.5, "token_2": 0.3}]
        }
        mock_model_class.return_value = mock_model

        config = EmbeddingConfig()
        service = BGEEmbeddingService(config)
        service.model = mock_model
        service._is_initialized = True

        embedding = await service.get_sparse_embedding("test text")

        assert embedding == {"token_1": 0.5, "token_2": 0.3}


class TestEmbeddingManager:
    """Test embedding manager with caching."""

    def test_manager_creation(self):
        """Test creating embedding manager."""
        config = EmbeddingConfig()
        service = BGEEmbeddingService(config)
        manager = EmbeddingManager(service)

        assert manager.embedding_service == service
        assert isinstance(manager._cache, dict)
        assert len(manager._cache) == 0

    async def test_cache_behavior(self):
        """Test embedding caching behavior."""
        config = EmbeddingConfig()
        service = BGEEmbeddingService(config)
        service.get_dense_embedding = AsyncMock(return_value=[0.1, 0.2])
        service.get_sparse_embedding = AsyncMock(return_value={1: 0.5})

        manager = EmbeddingManager(service)

        # First call should hit the service
        result1 = await manager.get_document_embeddings(["test text"], use_cache=True)

        # Second call should use cache
        result2 = await manager.get_document_embeddings(["test text"], use_cache=True)

        assert result1 == result2
        # Service should only be called once due to caching
        service.get_dense_embedding.assert_called_once()
        service.get_sparse_embedding.assert_called_once()

    async def test_batch_processing(self):
        """Test batch embedding processing."""
        config = EmbeddingConfig()
        service = BGEEmbeddingService(config)
        service.get_dense_embedding = AsyncMock(return_value=[0.1, 0.2])
        service.get_sparse_embedding = AsyncMock(return_value={1: 0.5})

        manager = EmbeddingManager(service)

        texts = ["text 1", "text 2", "text 3"]
        results = await manager.get_document_embeddings(
            texts, batch_size=2, use_cache=False
        )

        assert len(results) == 3
        for result in results:
            assert "dense_embedding" in result
            assert "sparse_embedding" in result
            assert result["dense_embedding"] == [0.1, 0.2]
            assert result["sparse_embedding"] == {1: 0.5}
