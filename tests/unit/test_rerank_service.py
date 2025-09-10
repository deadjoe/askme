"""
Unit tests for reranking service with BGE local and Cohere fallback.
"""

from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from askme.core.config import RerankConfig
from askme.rerank.rerank_service import (
    BGEReranker,
    CohereReranker,
    RerankingService,
    RerankResult,
)
from askme.retriever.base import Document, RetrievalResult


class TestRerankConfig:
    """Test reranking configuration."""

    def test_default_values(self):
        """Test default reranking configuration values."""
        config = RerankConfig()
        assert config.local_enabled is True
        assert config.cohere_enabled is False
        assert config.top_n == 8
        assert config.score_threshold == 0.0
        assert config.local_model == "BAAI/bge-reranker-v2.5-gemma2-lightweight"
        assert config.local_batch_size == 16
        assert config.local_max_length == 1024

    def test_custom_values(self):
        """Test custom reranking configuration."""
        config = RerankConfig(
            local_enabled=False,
            cohere_enabled=True,
            top_n=5,
            score_threshold=0.7,
            local_model="custom-model",
            cohere_model="rerank-english-v3.0",
        )
        assert config.local_enabled is False
        assert config.cohere_enabled is True
        assert config.top_n == 5
        assert config.score_threshold == 0.7
        assert config.local_model == "custom-model"
        assert config.cohere_model == "rerank-english-v3.0"


class TestBGEReranker:
    """Test BGE local reranker."""

    @pytest.fixture
    def sample_documents(self) -> List[RetrievalResult]:
        """Create sample retrieval results for testing."""
        docs = [
            Document(
                id="doc1",
                content="Machine learning is a subset of artificial intelligence.",
            ),
            Document(
                id="doc2",
                content="Natural language processing deals with text analysis.",
            ),
            Document(
                id="doc3", content="Computer vision focuses on image recognition tasks."
            ),
        ]

        return [
            RetrievalResult(
                document=docs[0], score=0.9, rank=1, retrieval_method="dense"
            ),
            RetrievalResult(
                document=docs[1], score=0.8, rank=2, retrieval_method="dense"
            ),
            RetrievalResult(
                document=docs[2], score=0.7, rank=3, retrieval_method="sparse"
            ),
        ]

    def test_initialization(self):
        """Test BGE reranker initialization."""
        config = RerankConfig()
        reranker = BGEReranker(config)

        assert reranker.config == config
        assert reranker.model is None
        assert reranker._is_initialized is False
        assert reranker.device in ["cpu", "cuda"]

    @pytest.mark.asyncio
    async def test_initialize_success(self):
        """Test successful BGE reranker model initialization."""
        config = RerankConfig(local_model="BAAI/bge-reranker-base")
        reranker = BGEReranker(config)

        # Mock FlagReranker
        mock_model = MagicMock()

        # Need to patch both the import and the class itself
        with patch("FlagEmbedding.FlagReranker", create=True) as mock_flag:
            mock_flag.return_value = mock_model

            await reranker.initialize()

            assert reranker._is_initialized is True
            assert reranker.model == mock_model
            mock_flag.assert_called_once_with(
                "BAAI/bge-reranker-base",
                use_fp16=(reranker.device != "cpu"),
                device=reranker.device,
            )

    @pytest.mark.asyncio
    async def test_initialize_trust_remote_code_fallback(self):
        """Test BGE reranker fallback when trust_remote_code is required."""
        config = RerankConfig(local_model="untrusted-model")
        reranker = BGEReranker(config)

        mock_model = MagicMock()

        def flag_reranker_side_effect(*args, **kwargs):
            if args[0] == "untrusted-model":
                raise Exception("trust_remote_code required")
            elif args[0] == "BAAI/bge-reranker-base":
                return mock_model
            else:
                raise Exception("Unexpected model")

        with patch(
            "FlagEmbedding.FlagReranker",
            side_effect=flag_reranker_side_effect,
            create=True,
        ):
            await reranker.initialize()

            assert reranker._is_initialized is True
            assert reranker.model == mock_model

    @pytest.mark.asyncio
    async def test_initialize_failure(self):
        """Test BGE reranker initialization failure."""
        config = RerankConfig()
        reranker = BGEReranker(config)

        # Use a specific error that won't trigger fallback logic
        def flag_reranker_side_effect(*args, **kwargs):
            raise Exception("Model load failed - not trust_remote_code related")

        with patch(
            "FlagEmbedding.FlagReranker",
            side_effect=flag_reranker_side_effect,
            create=True,
        ):
            with pytest.raises(Exception, match="Model load failed"):
                await reranker.initialize()

            assert reranker._is_initialized is False
            assert reranker.model is None

    @pytest.mark.asyncio
    async def test_rerank_success(self, sample_documents):
        """Test successful BGE reranking."""
        config = RerankConfig(local_batch_size=2, score_threshold=0.0)
        reranker = BGEReranker(config)

        # Mock initialized model
        mock_model = MagicMock()
        mock_model.compute_score.side_effect = [
            [0.95, 0.85],  # First batch
            [0.75],  # Second batch
        ]

        reranker.model = mock_model
        reranker._is_initialized = True

        results = await reranker.rerank(
            "machine learning query", sample_documents, top_n=3
        )

        assert len(results) == 3
        assert all(isinstance(r, RerankResult) for r in results)

        # Results should be sorted by rerank score (descending)
        assert (
            results[0].rerank_score
            >= results[1].rerank_score
            >= results[2].rerank_score
        )
        assert results[0].rerank_score == 0.95
        assert results[1].rerank_score == 0.85
        assert results[2].rerank_score == 0.75

        # Check ranks are updated
        assert results[0].new_rank == 1
        assert results[1].new_rank == 2
        assert results[2].new_rank == 3

        # Check reranker metadata
        for result in results:
            assert result.reranker_used == "bge_local"
            assert result.debug_info["model"] == config.local_model
            assert result.debug_info["device"] == reranker.device

    @pytest.mark.asyncio
    async def test_rerank_with_score_threshold(self, sample_documents):
        """Test BGE reranking with score threshold filtering."""
        config = RerankConfig(score_threshold=0.8)
        reranker = BGEReranker(config)

        # Mock model with varying scores
        mock_model = MagicMock()
        mock_model.compute_score.return_value = [0.9, 0.85, 0.5]  # One below threshold

        reranker.model = mock_model
        reranker._is_initialized = True

        results = await reranker.rerank("query", sample_documents, top_n=3)

        # Should filter out the result with score 0.5
        assert len(results) == 2
        assert all(r.rerank_score >= 0.8 for r in results)

    @pytest.mark.asyncio
    async def test_rerank_with_long_content(self, sample_documents):
        """Test BGE reranking with content truncation."""
        config = RerankConfig(local_max_length=128)
        reranker = BGEReranker(config)

        # Create document with very long content
        long_doc = Document(
            id="long_doc", content="Very long content. " * 200  # ~4000 characters
        )
        long_result = RetrievalResult(
            document=long_doc, score=0.8, rank=1, retrieval_method="dense"
        )

        mock_model = MagicMock()
        mock_model.compute_score.return_value = [0.85]

        reranker.model = mock_model
        reranker._is_initialized = True

        results = await reranker.rerank("query", [long_result], top_n=1)

        assert len(results) == 1
        # Verify model was called with truncated content
        call_args = mock_model.compute_score.call_args[0][0]
        assert len(call_args[0][1]) <= config.local_max_length * 4  # Character limit

    @pytest.mark.asyncio
    async def test_rerank_empty_documents(self):
        """Test BGE reranking with empty document list."""
        config = RerankConfig()
        reranker = BGEReranker(config)

        results = await reranker.rerank("query", [], top_n=5)
        assert results == []

    @pytest.mark.asyncio
    async def test_rerank_not_initialized(self, sample_documents):
        """Test BGE reranking auto-initializes if needed."""
        config = RerankConfig()
        reranker = BGEReranker(config)

        mock_model = MagicMock()
        mock_model.compute_score.return_value = [0.9, 0.8, 0.7]

        with patch(
            "askme.rerank.rerank_service.FlagReranker",
            return_value=mock_model,
            create=True,
        ):
            results = await reranker.rerank("query", sample_documents, top_n=3)

            assert reranker._is_initialized is True
            assert len(results) == 3

    def test_get_model_info(self):
        """Test getting BGE reranker model information."""
        config = RerankConfig(
            local_model="test-model",
            local_batch_size=4,
            local_max_length=256,
            score_threshold=0.5,
        )
        reranker = BGEReranker(config)

        info = reranker.get_model_info()

        assert info["type"] == "bge_local"
        assert info["model_name"] == "test-model"
        assert info["device"] == reranker.device
        assert info["batch_size"] == 4
        assert info["max_length"] == 256
        assert info["score_threshold"] == 0.5
        assert info["initialized"] is False

    @pytest.mark.asyncio
    async def test_cleanup(self):
        """Test BGE reranker resource cleanup."""
        config = RerankConfig()
        reranker = BGEReranker(config)

        # Set up initialized state
        reranker.model = MagicMock()
        reranker._is_initialized = True

        with patch("askme.rerank.rerank_service._torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            mock_torch.cuda.empty_cache = MagicMock()

            await reranker.cleanup()

            assert reranker.model is None
            assert reranker._is_initialized is False
            mock_torch.cuda.empty_cache.assert_called_once()


class TestCohereReranker:
    """Test Cohere reranker."""

    @pytest.fixture
    def sample_documents(self) -> List[RetrievalResult]:
        """Create sample retrieval results for testing."""
        docs = [
            Document(
                id="doc1", content="Machine learning algorithms for data analysis."
            ),
            Document(
                id="doc2", content="Natural language processing with transformers."
            ),
        ]

        return [
            RetrievalResult(
                document=docs[0], score=0.9, rank=1, retrieval_method="dense"
            ),
            RetrievalResult(
                document=docs[1], score=0.8, rank=2, retrieval_method="sparse"
            ),
        ]

    def test_initialization_with_api_key(self):
        """Test Cohere reranker initialization with API key."""
        config = RerankConfig()
        mock_client = MagicMock()

        with patch("cohere.Client", return_value=mock_client) as mock_cohere:
            reranker = CohereReranker(config, api_key="test-api-key")

            assert reranker.config == config
            assert reranker.api_key == "test-api-key"
            assert reranker.client == mock_client
            mock_cohere.assert_called_once_with("test-api-key")

    def test_initialization_without_api_key(self):
        """Test Cohere reranker initialization without API key."""
        config = RerankConfig()
        reranker = CohereReranker(config, api_key=None)

        assert reranker.config == config
        assert reranker.api_key is None
        assert reranker.client is None

    @pytest.mark.asyncio
    async def test_rerank_success(self, sample_documents):
        """Test successful Cohere reranking."""
        config = RerankConfig(
            cohere_model="rerank-english-v3.0",
            cohere_return_documents=True,
            cohere_max_chunks_per_doc=1,
        )

        # Mock Cohere API response
        mock_result1 = MagicMock()
        mock_result1.index = 1  # Second document ranks higher
        mock_result1.relevance_score = 0.92

        mock_result2 = MagicMock()
        mock_result2.index = 0  # First document ranks lower
        mock_result2.relevance_score = 0.88

        mock_response = MagicMock()
        mock_response.results = [mock_result1, mock_result2]

        mock_client = MagicMock()
        mock_client.rerank.return_value = mock_response

        reranker = CohereReranker(config, api_key="test-key")
        reranker.client = mock_client

        results = await reranker.rerank(
            "machine learning query", sample_documents, top_n=2
        )

        assert len(results) == 2

        # Check API call
        mock_client.rerank.assert_called_once_with(
            model="rerank-english-v3.0",
            query="machine learning query",
            documents=[
                "Machine learning algorithms for data analysis.",
                "Natural language processing with transformers.",
            ],
            top_n=2,
            return_documents=True,
            max_chunks_per_doc=1,
        )

        # Check results mapping
        assert results[0].document.id == "doc2"  # Index 1, higher score
        assert results[0].rerank_score == 0.92
        assert results[0].new_rank == 1
        assert results[0].reranker_used == "cohere"

        assert results[1].document.id == "doc1"  # Index 0, lower score
        assert results[1].rerank_score == 0.88
        assert results[1].new_rank == 2

    @pytest.mark.asyncio
    async def test_rerank_no_client(self, sample_documents):
        """Test Cohere reranking without initialized client."""
        config = RerankConfig()
        reranker = CohereReranker(config, api_key=None)

        with pytest.raises(RuntimeError, match="Cohere API client not initialized"):
            await reranker.rerank("query", sample_documents)

    @pytest.mark.asyncio
    async def test_rerank_empty_documents(self):
        """Test Cohere reranking with empty document list."""
        config = RerankConfig()
        reranker = CohereReranker(config, api_key="test-key")
        reranker.client = MagicMock()

        results = await reranker.rerank("query", [])
        assert results == []

    @pytest.mark.asyncio
    async def test_rerank_with_long_content(self, sample_documents):
        """Test Cohere reranking with content truncation."""
        config = RerankConfig()

        # Create document with very long content
        long_doc = Document(
            id="long_doc", content="Very long content. " * 500
        )  # >4000 chars
        long_result = RetrievalResult(
            document=long_doc, score=0.8, rank=1, retrieval_method="dense"
        )

        mock_result = MagicMock()
        mock_result.index = 0
        mock_result.relevance_score = 0.85

        mock_response = MagicMock()
        mock_response.results = [mock_result]

        mock_client = MagicMock()
        mock_client.rerank.return_value = mock_response

        reranker = CohereReranker(config, api_key="test-key")
        reranker.client = mock_client

        results = await reranker.rerank("query", [long_result], top_n=1)

        assert len(results) == 1
        # Verify content was truncated
        call_args = mock_client.rerank.call_args[1]["documents"]
        assert len(call_args[0]) <= 4003  # 4000 chars + "..."

    def test_get_model_info(self):
        """Test getting Cohere reranker model information."""
        config = RerankConfig(
            cohere_model="rerank-english-v3.0",
            cohere_max_chunks_per_doc=2,
            cohere_return_documents=False,
        )
        reranker = CohereReranker(config, api_key="test-key")
        reranker.client = MagicMock()

        info = reranker.get_model_info()

        assert info["type"] == "cohere"
        assert info["model_name"] == "rerank-english-v3.0"
        assert info["max_chunks_per_doc"] == 2
        assert info["return_documents"] is False
        assert info["api_key_configured"] is True


class TestRerankingService:
    """Test main reranking service."""

    @pytest.fixture
    def sample_documents(self) -> List[RetrievalResult]:
        """Create sample retrieval results for testing."""
        docs = [
            Document(id="doc1", content="First document content."),
            Document(id="doc2", content="Second document content."),
        ]

        return [
            RetrievalResult(
                document=docs[0], score=0.9, rank=1, retrieval_method="dense"
            ),
            RetrievalResult(
                document=docs[1], score=0.8, rank=2, retrieval_method="sparse"
            ),
        ]

    def test_initialization_bge_only(self):
        """Test service initialization with BGE reranker only."""
        config = RerankConfig(local_enabled=True, cohere_enabled=False)
        service = RerankingService(config, cohere_api_key=None)

        assert service.bge_reranker is not None
        assert service.cohere_reranker is None
        assert service._fallback_enabled is False

    def test_initialization_cohere_only(self):
        """Test service initialization with Cohere reranker only."""
        config = RerankConfig(local_enabled=False, cohere_enabled=True)
        service = RerankingService(config, cohere_api_key="test-key")

        assert service.bge_reranker is None
        assert service.cohere_reranker is not None
        assert service._fallback_enabled is True

    def test_initialization_both_enabled(self):
        """Test service initialization with both rerankers enabled."""
        config = RerankConfig(local_enabled=True, cohere_enabled=True)
        service = RerankingService(config, cohere_api_key="test-key")

        assert service.bge_reranker is not None
        assert service.cohere_reranker is not None
        assert service._fallback_enabled is True

    @pytest.mark.asyncio
    async def test_initialize_service(self):
        """Test service initialization."""
        config = RerankConfig(local_enabled=True, cohere_enabled=True)
        service = RerankingService(config, cohere_api_key="test-key")

        # Mock BGE initialization
        service.bge_reranker = MagicMock()
        service.bge_reranker.initialize = AsyncMock()

        await service.initialize()

        service.bge_reranker.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_rerank_prefer_local_success(self, sample_documents):
        """Test reranking preferring local reranker (success case)."""
        config = RerankConfig(local_enabled=True, cohere_enabled=True, top_n=5)
        service = RerankingService(config, cohere_api_key="test-key")

        # Mock successful BGE reranking
        mock_bge_results = [
            RerankResult(
                document=sample_documents[0].document,
                original_score=0.9,
                rerank_score=0.95,
                original_rank=1,
                new_rank=1,
                reranker_used="bge_local",
            )
        ]

        service.bge_reranker = MagicMock()
        service.bge_reranker.rerank = AsyncMock(return_value=mock_bge_results)
        service.cohere_reranker = MagicMock()

        results = await service.rerank("query", sample_documents, prefer_local=True)

        assert results == mock_bge_results
        service.bge_reranker.rerank.assert_called_once_with(
            "query", sample_documents, 5
        )
        service.cohere_reranker.rerank.assert_not_called()

    @pytest.mark.asyncio
    async def test_rerank_prefer_local_fallback(self, sample_documents):
        """Test reranking with local failure and Cohere fallback."""
        config = RerankConfig(local_enabled=True, cohere_enabled=True, top_n=5)
        service = RerankingService(config, cohere_api_key="test-key")

        # Mock BGE failure and Cohere success
        mock_cohere_results = [
            RerankResult(
                document=sample_documents[0].document,
                original_score=0.9,
                rerank_score=0.92,
                original_rank=1,
                new_rank=1,
                reranker_used="cohere",
            )
        ]

        service.bge_reranker = MagicMock()
        service.bge_reranker.rerank = AsyncMock(side_effect=Exception("BGE failed"))
        service.cohere_reranker = MagicMock()
        service.cohere_reranker.rerank = AsyncMock(return_value=mock_cohere_results)

        results = await service.rerank("query", sample_documents, prefer_local=True)

        assert results == mock_cohere_results
        service.bge_reranker.rerank.assert_called_once_with(
            "query", sample_documents, 5
        )
        service.cohere_reranker.rerank.assert_called_once_with(
            "query", sample_documents, 5
        )

    @pytest.mark.asyncio
    async def test_rerank_no_rerankers_available(self, sample_documents):
        """Test reranking with no rerankers available."""
        config = RerankConfig(local_enabled=False, cohere_enabled=False)
        service = RerankingService(config, cohere_api_key=None)

        with pytest.raises(RuntimeError, match="No rerankers available"):
            await service.rerank("query", sample_documents)

    @pytest.mark.asyncio
    async def test_rerank_by_method_bge(self, sample_documents):
        """Test reranking by specific method (BGE)."""
        config = RerankConfig(local_enabled=True, top_n=3)
        service = RerankingService(config)

        mock_results = [MagicMock()]
        service.bge_reranker = MagicMock()
        service.bge_reranker.rerank = AsyncMock(return_value=mock_results)

        results = await service.rerank_by_method(
            "query", sample_documents, method="bge_local"
        )

        assert results == mock_results
        service.bge_reranker.rerank.assert_called_once_with(
            "query", sample_documents, 3
        )

    @pytest.mark.asyncio
    async def test_rerank_by_method_cohere(self, sample_documents):
        """Test reranking by specific method (Cohere)."""
        config = RerankConfig(cohere_enabled=True, top_n=4)
        service = RerankingService(config, cohere_api_key="test-key")

        mock_results = [MagicMock()]
        service.cohere_reranker = MagicMock()
        service.cohere_reranker.rerank = AsyncMock(return_value=mock_results)

        results = await service.rerank_by_method(
            "query", sample_documents, method="cohere"
        )

        assert results == mock_results
        service.cohere_reranker.rerank.assert_called_once_with(
            "query", sample_documents, 4
        )

    @pytest.mark.asyncio
    async def test_rerank_by_method_unavailable(self, sample_documents):
        """Test reranking by unavailable method."""
        config = RerankConfig(local_enabled=False)
        service = RerankingService(config)

        with pytest.raises(ValueError, match="BGE local reranker not available"):
            await service.rerank_by_method(
                "query", sample_documents, method="bge_local"
            )

    @pytest.mark.asyncio
    async def test_rerank_by_method_unknown(self, sample_documents):
        """Test reranking by unknown method."""
        config = RerankConfig()
        service = RerankingService(config)

        with pytest.raises(ValueError, match="Unknown reranking method"):
            await service.rerank_by_method("query", sample_documents, method="unknown")

    def test_get_available_methods(self):
        """Test getting available reranking methods."""
        # BGE only
        config1 = RerankConfig(local_enabled=True, cohere_enabled=False)
        service1 = RerankingService(config1)
        assert service1.get_available_methods() == ["bge_local"]

        # Cohere only
        config2 = RerankConfig(local_enabled=False, cohere_enabled=True)
        service2 = RerankingService(config2, cohere_api_key="test-key")
        assert service2.get_available_methods() == ["cohere"]

        # Both
        config3 = RerankConfig(local_enabled=True, cohere_enabled=True)
        service3 = RerankingService(config3, cohere_api_key="test-key")
        assert set(service3.get_available_methods()) == {"bge_local", "cohere"}

    def test_get_service_info(self):
        """Test getting service information."""
        config = RerankConfig(
            local_enabled=True,
            cohere_enabled=True,
            top_n=5,
            score_threshold=0.7,
        )
        service = RerankingService(config, cohere_api_key="test-key")

        # Mock reranker info
        service.bge_reranker = MagicMock()
        service.bge_reranker.get_model_info.return_value = {"type": "bge_local"}
        service.cohere_reranker = MagicMock()
        service.cohere_reranker.get_model_info.return_value = {"type": "cohere"}

        info = service.get_service_info()

        assert info["available_methods"] == service.get_available_methods()
        assert info["fallback_enabled"] is True
        assert info["config"]["top_n"] == 5
        assert info["config"]["score_threshold"] == 0.7
        assert info["bge_reranker"] == {"type": "bge_local"}
        assert info["cohere_reranker"] == {"type": "cohere"}

    @pytest.mark.asyncio
    async def test_cleanup(self):
        """Test service cleanup."""
        config = RerankConfig(local_enabled=True)
        service = RerankingService(config)

        service.bge_reranker = MagicMock()
        service.bge_reranker.cleanup = AsyncMock()

        await service.cleanup()

        service.bge_reranker.cleanup.assert_called_once()

    @pytest.mark.asyncio
    async def test_rerank_empty_documents(self):
        """Test reranking with empty document list."""
        config = RerankConfig(local_enabled=True)
        service = RerankingService(config)

        results = await service.rerank("query", [])
        assert results == []
