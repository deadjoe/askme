from typing import Any, cast

"""
Unit tests for embedding service.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from askme.core.config import EmbeddingConfig
from askme.core.embeddings import (
    BGEEmbeddingService,
    EmbeddingManager,
    HybridEmbeddingService,
    Qwen3EmbeddingService,
    create_embedding_backend,
)


class TestEmbeddingFactory:
    """Test embedding backend factory."""

    def test_default_returns_hybrid(self: Any) -> None:
        config = EmbeddingConfig()
        backend = create_embedding_backend(config)
        assert isinstance(backend, HybridEmbeddingService)
        assert backend.supports_sparse is True

    def test_explicit_bge_backend(self: Any) -> None:
        config = EmbeddingConfig(
            backend="bge_m3",
            model="BAAI/bge-m3",
            model_name="bge-m3",
            dimension=1024,
        )
        backend = create_embedding_backend(config)
        assert isinstance(backend, BGEEmbeddingService)
        assert backend.supports_sparse is True

    def test_qwen3_backend_resolution(self: Any) -> None:
        config = EmbeddingConfig(
            backend="qwen3-embedding-small",
            model="Qwen/Qwen3-Embedding-Small",
            dimension=1536,
        )
        backend = create_embedding_backend(config)
        assert isinstance(backend, HybridEmbeddingService)
        assert backend.supports_sparse is True

    def test_qwen3_dense_only_backend(self: Any) -> None:
        config = EmbeddingConfig(
            backend="qwen3_dense_only",
            model="Qwen/Qwen3-Embedding-Small",
            dimension=1536,
        )
        backend = create_embedding_backend(config)
        assert isinstance(backend, Qwen3EmbeddingService)
        assert backend.supports_sparse is False


class TestBGEEmbeddingService:
    """Test BGE embedding service."""

    def _bge_config(self: Any) -> EmbeddingConfig:
        return EmbeddingConfig(
            backend="bge_m3",
            model="BAAI/bge-m3",
            model_name="bge-m3",
            dimension=1024,
            batch_size=16,
        )

    def test_service_creation(self: Any) -> None:
        """Test creating embedding service."""
        config = self._bge_config()
        service = BGEEmbeddingService(config)

        assert service.config.model == "BAAI/bge-m3"
        assert service.config.dimension == 1024
        assert service.config.batch_size == 16
        assert service.model is None
        assert service._is_initialized is False

    @patch("askme.core.embeddings.BGEM3FlagModel")
    async def test_initialization(self: Any, mock_model_class: MagicMock) -> None:
        """Test service initialization."""
        # Mock the BGE model
        mock_model = MagicMock()
        mock_model_class.return_value = mock_model

        config = self._bge_config()
        service = BGEEmbeddingService(config)

        await service.initialize()

        assert service._is_initialized is True
        mock_model_class.assert_called_once()

    @patch("askme.core.embeddings.BGEM3FlagModel")
    async def test_get_dense_embedding(self: Any, mock_model_class: MagicMock) -> None:
        """Test getting dense embeddings."""
        # Mock the BGE model
        mock_model = MagicMock()
        mock_model.encode.return_value = {"dense_vecs": [[0.1, 0.2, 0.3]]}
        mock_model_class.return_value = mock_model

        config = self._bge_config()
        service = BGEEmbeddingService(config)
        service.model = mock_model
        service._is_initialized = True

        embedding = await service.get_dense_embedding("test text")

        assert embedding == [0.1, 0.2, 0.3]
        mock_model.encode.assert_called_once()

    @patch("askme.core.embeddings.BGEM3FlagModel")
    async def test_get_sparse_embedding(self: Any, mock_model_class: MagicMock) -> None:
        """Test getting sparse embeddings."""
        # Mock the BGE model
        mock_model = MagicMock()
        mock_model.encode.return_value = {
            "lexical_weights": [{"token_1": 0.5, "token_2": 0.3}]
        }
        mock_model_class.return_value = mock_model

        config = self._bge_config()
        service = BGEEmbeddingService(config)
        service.model = mock_model
        service._is_initialized = True

        embedding = await service.get_sparse_embedding("test text")

        assert embedding == {"token_1": 0.5, "token_2": 0.3}


class TestEmbeddingManager:
    """Test embedding manager with caching."""

    def _bge_service(self: Any) -> BGEEmbeddingService:
        config = EmbeddingConfig(
            backend="bge_m3",
            model="BAAI/bge-m3",
            model_name="bge-m3",
            dimension=1024,
            batch_size=16,
        )
        return BGEEmbeddingService(config)

    def test_manager_creation(self: Any) -> None:
        """Test creating embedding manager."""
        service = self._bge_service()
        manager = EmbeddingManager(service)

        assert manager.embedding_service == service
        assert isinstance(manager._cache, dict)
        assert len(manager._cache) == 0

    async def test_cache_behavior(self: Any) -> None:
        """Test embedding caching behavior."""
        service = self._bge_service()
        # Mock the new batch processing method instead of individual methods
        mock_embedding = {
            "text": "test text",
            "dense_embedding": [0.1, 0.2],
            "sparse_embedding": {1: 0.5},
            "embedding_dim": 2,
        }
        cast(Any, service).encode_documents = AsyncMock(return_value=[mock_embedding])

        manager = EmbeddingManager(service)

        # First call should hit the service
        result1 = await manager.get_document_embeddings(["test text"], use_cache=True)

        # Second call should use cache
        result2 = await manager.get_document_embeddings(["test text"], use_cache=True)

        assert result1 == result2
        # Service should only be called once due to caching
        assert isinstance(service.encode_documents, AsyncMock)
        service.encode_documents.assert_called_once_with(["test text"], batch_size=None)

    async def test_batch_processing(self: Any) -> None:
        """Test batch embedding processing."""
        service = self._bge_service()
        cast(Any, service).get_dense_embedding = AsyncMock(return_value=[0.1, 0.2])
        cast(Any, service).get_sparse_embedding = AsyncMock(return_value={1: 0.5})

        manager = EmbeddingManager(service)

        # Mock batch processing with proper return format
        mock_embeddings = [
            {
                "text": "text 1",
                "dense_embedding": [0.1, 0.2],
                "sparse_embedding": {1: 0.5},
                "embedding_dim": 2,
            },
            {
                "text": "text 2",
                "dense_embedding": [0.3, 0.4],
                "sparse_embedding": {2: 0.6},
                "embedding_dim": 2,
            },
            {
                "text": "text 3",
                "dense_embedding": [0.5, 0.6],
                "sparse_embedding": {3: 0.7},
                "embedding_dim": 2,
            },
        ]
        cast(Any, service).encode_documents = AsyncMock(return_value=mock_embeddings)

        texts = ["text 1", "text 2", "text 3"]
        results = await manager.get_document_embeddings(
            texts, batch_size=2, use_cache=False
        )

        assert len(results) == 3
        for i, result in enumerate(results):
            assert "dense_embedding" in result
            assert "sparse_embedding" in result
            assert result["text"] == texts[i]

        # Check that batch service was called once with the batch
        assert isinstance(service.encode_documents, AsyncMock)
        service.encode_documents.assert_called_once_with(texts, batch_size=2)


class TestBGEEmbeddingServiceAdvanced:
    """Advanced tests for BGE embedding service covering uncovered code paths."""

    async def test_initialize_already_initialized(self: Any) -> None:
        """Test that initialize doesn't reload when already initialized."""
        config = EmbeddingConfig()
        service = BGEEmbeddingService(config)
        service._is_initialized = True

        # This should return immediately without loading the model
        await service.initialize()

        # Model should still be None since we didn't actually initialize
        assert service.model is None
        assert service._is_initialized is True

    @patch("askme.core.embeddings.BGEM3FlagModel")
    async def test_initialize_import_error(
        self: Any, mock_model_class: MagicMock
    ) -> None:
        """Test initialization with import failure."""
        # Mock import failure
        mock_model_class.side_effect = ImportError("FlagEmbedding not available")

        config = EmbeddingConfig()
        service = BGEEmbeddingService(config)

        with pytest.raises(ImportError):
            await service.initialize()

    @patch("askme.core.embeddings.BGEM3FlagModel")
    async def test_encode_documents_with_batches(
        self: Any, mock_model_class: MagicMock
    ) -> None:
        """Test encode_documents with batch processing."""
        # Mock the BGE model
        mock_model = MagicMock()

        # Create mock sparse objects with indices and values attributes
        mock_sparse1 = MagicMock()
        mock_sparse1.indices = [1, 3]
        mock_sparse1.values = [0.5, 0.3]
        mock_sparse2 = MagicMock()
        mock_sparse2.indices = [2, 4]
        mock_sparse2.values = [0.7, 0.2]

        mock_model.encode.return_value = {
            "dense_vecs": [[0.1, 0.2], [0.3, 0.4]],
            "sparse_vecs": [mock_sparse1, mock_sparse2],
        }
        mock_model_class.return_value = mock_model

        config = EmbeddingConfig(batch_size=2)
        service = BGEEmbeddingService(config)
        service.model = mock_model
        service._is_initialized = True

        texts = ["text1", "text2", "text3", "text4"]
        results = await service.encode_documents(texts)

        # Should process in 2 batches of 2
        assert len(results) == 4
        assert mock_model.encode.call_count == 2

    @patch("askme.core.embeddings.BGEM3FlagModel")
    async def test_encode_documents_not_initialized(
        self: Any, mock_model_class: MagicMock
    ) -> None:
        """Test encode_documents automatically initializes."""
        mock_model = MagicMock()

        # Create mock sparse object with indices and values attributes
        mock_sparse = MagicMock()
        mock_sparse.indices = [1, 2]
        mock_sparse.values = [0.5, 0.3]

        mock_model.encode.return_value = {
            "dense_vecs": [[0.1, 0.2]],
            "sparse_vecs": [mock_sparse],
        }
        mock_model_class.return_value = mock_model

        config = EmbeddingConfig()
        service = BGEEmbeddingService(config)

        results = await service.encode_documents(["test"])

        assert service._is_initialized is True
        assert len(results) == 1

    async def test_encode_documents_empty_list(self: Any) -> None:
        """Test encode_documents with empty text list."""
        config = EmbeddingConfig()
        service = BGEEmbeddingService(config)
        service._is_initialized = True

        results = await service.encode_documents([])

        assert results == []

    @patch("askme.core.embeddings.BGEM3FlagModel")
    async def test_get_dense_embedding_not_initialized(
        self: Any, mock_model_class: MagicMock
    ) -> None:
        """Test get_dense_embedding auto-initializes."""
        mock_model = MagicMock()
        mock_model.encode.return_value = {"dense_vecs": [[0.1, 0.2, 0.3]]}
        mock_model_class.return_value = mock_model

        config = EmbeddingConfig()
        service = BGEEmbeddingService(config)

        result = await service.get_dense_embedding("test")

        assert service._is_initialized is True
        assert result == [0.1, 0.2, 0.3]

    async def test_get_dense_embedding_model_not_initialized(self: Any) -> None:
        """Test get_dense_embedding raises error when model is None."""
        config = EmbeddingConfig()
        service = BGEEmbeddingService(config)
        service._is_initialized = True
        service.model = None

        with pytest.raises(RuntimeError, match="Embedding model not initialized"):
            await service.get_dense_embedding("test")

    @patch("askme.core.embeddings.BGEM3FlagModel")
    async def test_get_dense_embedding_numpy_array(
        self: Any, mock_model_class: MagicMock
    ) -> None:
        """Test get_dense_embedding with numpy array response."""
        import numpy as np

        # Mock numpy array response
        mock_array = np.array([[0.1, 0.2, 0.3]])
        mock_model = MagicMock()
        mock_model.encode.return_value = {"dense_vecs": mock_array}
        mock_model_class.return_value = mock_model

        config = EmbeddingConfig()
        service = BGEEmbeddingService(config)
        service.model = mock_model
        service._is_initialized = True

        result = await service.get_dense_embedding("test")

        assert result == [0.1, 0.2, 0.3]

    @patch("askme.core.embeddings.BGEM3FlagModel")
    async def test_get_dense_embedding_dense_vectors_key(
        self: Any, mock_model_class: MagicMock
    ) -> None:
        """Test get_dense_embedding with 'dense_vectors' key."""
        mock_model = MagicMock()
        mock_model.encode.return_value = {"dense_vectors": [[0.1, 0.2, 0.3]]}
        mock_model_class.return_value = mock_model

        config = EmbeddingConfig()
        service = BGEEmbeddingService(config)
        service.model = mock_model
        service._is_initialized = True

        result = await service.get_dense_embedding("test")

        assert result == [0.1, 0.2, 0.3]

    @patch("askme.core.embeddings.BGEM3FlagModel")
    async def test_get_dense_embedding_no_dense_key(
        self: Any, mock_model_class: MagicMock
    ) -> None:
        """Test get_dense_embedding raises error when no dense embeddings returned."""
        mock_model = MagicMock()
        mock_model.encode.return_value = {"other_key": "value"}
        mock_model_class.return_value = mock_model

        config = EmbeddingConfig()
        service = BGEEmbeddingService(config)
        service.model = mock_model
        service._is_initialized = True

        with pytest.raises(RuntimeError, match="Dense embedding not returned by model"):
            await service.get_dense_embedding("test")

    @patch("askme.core.embeddings.BGEM3FlagModel")
    async def test_get_sparse_embedding_not_initialized(
        self: Any, mock_model_class: MagicMock
    ) -> None:
        """Test get_sparse_embedding auto-initializes."""
        mock_model = MagicMock()
        mock_model.encode.return_value = {"lexical_weights": [{"token": 0.5}]}
        mock_model_class.return_value = mock_model

        config = EmbeddingConfig()
        service = BGEEmbeddingService(config)

        result = await service.get_sparse_embedding("test")

        assert service._is_initialized is True
        assert result == {"token": 0.5}

    async def test_get_sparse_embedding_model_not_initialized(self: Any) -> None:
        """Test get_sparse_embedding raises error when model is None."""
        config = EmbeddingConfig()
        service = BGEEmbeddingService(config)
        service._is_initialized = True
        service.model = None

        with pytest.raises(RuntimeError, match="Embedding model not initialized"):
            await service.get_sparse_embedding("test")

    @patch("askme.core.embeddings.BGEM3FlagModel")
    async def test_get_sparse_embedding_sparse_vecs_key(
        self: Any, mock_model_class: MagicMock
    ) -> None:
        """Test get_sparse_embedding with 'sparse_vecs' key."""
        mock_model = MagicMock()
        mock_sparse_vec = MagicMock()
        mock_model.encode.return_value = {"sparse_vecs": [mock_sparse_vec]}
        mock_model_class.return_value = mock_model

        config = EmbeddingConfig()
        service = BGEEmbeddingService(config)
        service.model = mock_model
        service._is_initialized = True

        # Mock _convert_sparse_embedding
        with patch.object(
            service, "_convert_sparse_embedding", return_value={1: 0.5}
        ) as mock_convert:
            result = await service.get_sparse_embedding("test")

            assert result == {1: 0.5}
            mock_convert.assert_called_once_with(mock_sparse_vec)

    @patch("askme.core.embeddings.BGEM3FlagModel")
    async def test_get_sparse_embedding_fallback(
        self: Any, mock_model_class: MagicMock
    ) -> None:
        """Test get_sparse_embedding fallback to empty dict."""
        mock_model = MagicMock()
        mock_model.encode.return_value = {"other_key": "value"}
        mock_model_class.return_value = mock_model

        config = EmbeddingConfig()
        service = BGEEmbeddingService(config)
        service.model = mock_model
        service._is_initialized = True

        result = await service.get_sparse_embedding("test")

        assert result == {}

    @patch("askme.core.embeddings.BGEM3FlagModel")
    async def test_encode_query(self: Any, mock_model_class: MagicMock) -> None:
        """Test encode_query method."""
        import numpy as np

        mock_model = MagicMock()

        # Create mock sparse object with indices and values attributes
        mock_sparse = MagicMock()
        mock_sparse.indices = [1, 2]
        mock_sparse.values = [0.8, 0.6]

        mock_model.encode.return_value = {
            "dense_vecs": [np.array([0.1, 0.2])],
            "sparse_vecs": [mock_sparse],
        }
        mock_model_class.return_value = mock_model

        config = EmbeddingConfig(normalize_embeddings=False)
        service = BGEEmbeddingService(config)
        service.model = mock_model
        service._is_initialized = True

        result = await service.encode_query("search query")

        assert "dense_embedding" in result
        assert "sparse_embedding" in result
        assert result["dense_embedding"] == [0.1, 0.2]
        assert result["sparse_embedding"] == {1: 0.8, 2: 0.6}

    @patch("askme.core.embeddings.BGEM3FlagModel")
    async def test_encode_query_with_instruction(
        self: Any, mock_model_class: MagicMock
    ) -> None:
        """Test encode_query with instruction prefix."""
        mock_model = MagicMock()
        mock_model.encode.return_value = {
            "dense_vecs": [[0.1, 0.2]],
            "lexical_weights": [{"search": 0.8}],
        }
        mock_model_class.return_value = mock_model

        config = EmbeddingConfig()
        service = BGEEmbeddingService(config)
        service.model = mock_model
        service._is_initialized = True

        await service.encode_query(
            "machine learning",
            "Represent this sentence for searching relevant passages:",
        )

        # Verify that instruction was prepended
        mock_model.encode.assert_called_once()
        call_args = mock_model.encode.call_args[0]
        assert (
            call_args[0][0]
            == "Represent this sentence for searching relevant passages: "
            "machine learning"
        )

    @patch("askme.core.embeddings.BGEM3FlagModel")
    async def test_get_model_info(self: Any, mock_model_class: MagicMock) -> None:
        """Test get_model_info method."""
        mock_model = MagicMock()
        mock_model_class.return_value = mock_model

        config = EmbeddingConfig(model="custom/model", dimension=768, batch_size=32)
        service = BGEEmbeddingService(config)

        info = await service.get_model_info()

        assert info["model_name"] == "custom/model"
        assert info["embedding_dim"] == 768
        assert info["batch_size"] == 32
        assert info["supports_sparse"] is True
        assert info["device"] == service.device

    def test_device_selection_cuda(self: Any) -> None:
        """Test CUDA device selection."""
        with patch("torch.cuda.is_available", return_value=True):
            config = EmbeddingConfig()
            service = BGEEmbeddingService(config)
            assert "cuda" in service.device

    def test_device_selection_cpu(self: Any) -> None:
        """Test CPU device selection when CUDA unavailable."""
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.backends.mps.is_available", return_value=False),
        ):
            config = EmbeddingConfig()
            service = BGEEmbeddingService(config)
            assert service.device == "cpu"


class TestHybridEmbeddingService:
    """Test hybrid embedding service with Qwen3 + BGE-M3."""

    @patch("askme.core.embeddings.BGEEmbeddingService.initialize")
    @patch("askme.core.embeddings.Qwen3EmbeddingService.initialize")
    async def test_initialization(
        self: Any, mock_qwen_init: AsyncMock, mock_bge_init: AsyncMock
    ) -> None:
        """Test hybrid service initializes both backends."""
        config = EmbeddingConfig()
        service = HybridEmbeddingService(config)

        await service.initialize()

        assert service._is_initialized is True
        mock_qwen_init.assert_called_once()
        mock_bge_init.assert_called_once()

    @patch("askme.core.embeddings.BGEEmbeddingService.encode_query")
    @patch("askme.core.embeddings.Qwen3EmbeddingService.encode_query")
    async def test_encode_query_combines_embeddings(
        self: Any, mock_qwen_encode: AsyncMock, mock_bge_encode: AsyncMock
    ) -> None:
        """Test query encoding combines dense (Qwen3) + sparse (BGE-M3)."""
        config = EmbeddingConfig()
        service = HybridEmbeddingService(config)
        service._is_initialized = True

        # Mock responses
        mock_qwen_encode.return_value = {
            "query": "test query",
            "dense_embedding": [0.1, 0.2, 0.3],
            "sparse_embedding": {},
            "embedding_dim": 3,
        }
        mock_bge_encode.return_value = {
            "query": "test query",
            "dense_embedding": [0.4, 0.5, 0.6],  # ignored
            "sparse_embedding": {1: 0.7, 2: 0.8},
            "embedding_dim": 3,
        }

        result = await service.encode_query("test query")

        assert result["query"] == "test query"
        assert result["dense_embedding"] == [0.1, 0.2, 0.3]  # from Qwen3
        assert result["sparse_embedding"] == {1: 0.7, 2: 0.8}  # from BGE-M3
        assert result["embedding_dim"] == 3

    @patch("askme.core.embeddings.BGEEmbeddingService.encode_documents")
    @patch("askme.core.embeddings.Qwen3EmbeddingService.encode_documents")
    async def test_encode_documents_combines_embeddings(
        self: Any, mock_qwen_encode: AsyncMock, mock_bge_encode: AsyncMock
    ) -> None:
        """Test document encoding combines dense + sparse embeddings."""
        config = EmbeddingConfig()
        service = HybridEmbeddingService(config)
        service._is_initialized = True

        # Mock responses
        mock_qwen_encode.return_value = [
            {
                "text": "doc1",
                "dense_embedding": [0.1, 0.2],
                "sparse_embedding": {},
                "embedding_dim": 2,
            }
        ]
        mock_bge_encode.return_value = [
            {
                "text": "doc1",
                "dense_embedding": [0.3, 0.4],  # ignored
                "sparse_embedding": {1: 0.5},
                "embedding_dim": 2,
            }
        ]

        result = await service.encode_documents(["doc1"])

        assert len(result) == 1
        assert result[0]["text"] == "doc1"
        assert result[0]["dense_embedding"] == [0.1, 0.2]  # from Qwen3
        assert result[0]["sparse_embedding"] == {1: 0.5}  # from BGE-M3

    def test_supports_sparse(self: Any) -> None:
        """Test hybrid service supports sparse vectors."""
        config = EmbeddingConfig()
        service = HybridEmbeddingService(config)
        assert service.supports_sparse is True

    @patch("askme.core.embeddings.BGEEmbeddingService.cleanup")
    @patch("askme.core.embeddings.Qwen3EmbeddingService.cleanup")
    async def test_cleanup_both_services(
        self: Any, mock_qwen_cleanup: AsyncMock, mock_bge_cleanup: AsyncMock
    ) -> None:
        """Test cleanup calls both services."""
        config = EmbeddingConfig()
        service = HybridEmbeddingService(config)
        service._is_initialized = True

        await service.cleanup()

        mock_qwen_cleanup.assert_called_once()
        mock_bge_cleanup.assert_called_once()
        assert service._is_initialized is False
