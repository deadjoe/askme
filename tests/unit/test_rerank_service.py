"""Unit tests for reranking services."""

from typing import Any, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from askme.core.config import RerankConfig
from askme.rerank.rerank_service import (
    BGEReranker,
    Qwen3Reranker,
    RerankingService,
    RerankResult,
)
from askme.retriever.base import Document, RetrievalResult


class TestRerankConfig:
    """Verify configuration defaults and overrides."""

    def test_default_values(self: Any) -> None:
        config = RerankConfig()
        assert config.local_backend == "qwen_local"
        assert config.local_model == "Qwen/Qwen3-Reranker-0.6B"
        assert config.local_enabled is True
        assert config.local_batch_size == 16
        assert config.local_max_length == 1024
        assert config.local_instruction.startswith("Given a web search query")
        assert config.top_n == 8
        assert config.score_threshold == 0.0

    def test_custom_values(self: Any) -> None:
        config = RerankConfig(
            local_backend="bge_local",
            local_model="BAAI/bge-reranker-v2-m3",
            local_enabled=False,
            top_n=5,
            score_threshold=0.6,
            local_batch_size=4,
            local_max_length=2048,
        )

        assert config.local_backend == "bge_local"
        assert config.local_model == "BAAI/bge-reranker-v2-m3"
        assert config.local_enabled is False
        assert config.top_n == 5
        assert config.score_threshold == 0.6
        assert config.local_batch_size == 4
        assert config.local_max_length == 2048


class TestBGEReranker:
    """Exercise BGE-specific reranker logic with mocks."""

    @pytest.fixture
    def sample_results(self: Any) -> List[RetrievalResult]:
        docs = [
            Document(id="d1", content="Alpha content"),
            Document(id="d2", content="Beta content"),
            Document(id="d3", content="Gamma content"),
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

    def _bge_config(self: Any, **overrides: Any) -> RerankConfig:
        base = {
            "local_backend": "bge_local",
            "local_model": "BAAI/bge-reranker-v2-m3",
        }
        base.update(overrides)
        return RerankConfig(**base)

    def test_initialization_device_detection(self: Any) -> None:
        reranker = BGEReranker(self._bge_config())
        assert reranker.model is None
        assert reranker._is_initialized is False
        assert reranker.device in {"cpu", "cuda", "mps"}

    @pytest.mark.asyncio
    async def test_initialize_uses_flag_embedding(self: Any) -> None:
        reranker = BGEReranker(self._bge_config())
        mock_model = MagicMock()

        with patch(
            "FlagEmbedding.FlagReranker", return_value=mock_model, create=True
        ) as mock_cls:
            await reranker.initialize()

        mock_cls.assert_called_once()
        assert reranker.model is mock_model
        assert reranker._is_initialized is True

    @pytest.mark.asyncio
    async def test_rerank_batches_scores(
        self: Any, sample_results: List[RetrievalResult]
    ) -> None:
        config = self._bge_config(local_batch_size=2)
        reranker = BGEReranker(config)
        reranker.model = MagicMock()
        reranker._is_initialized = True
        reranker.model.compute_score.side_effect = [[0.95, 0.8], [0.6]]

        results = await reranker.rerank("sample", sample_results, top_n=3)

        assert [round(r.rerank_score, 2) for r in results] == [0.95, 0.8, 0.6]
        assert [r.new_rank for r in results] == [1, 2, 3]
        assert all(r.reranker_used == "bge_local" for r in results)

    @pytest.mark.asyncio
    async def test_rerank_applies_threshold(
        self: Any, sample_results: List[RetrievalResult]
    ) -> None:
        config = self._bge_config(score_threshold=0.8)
        reranker = BGEReranker(config)
        reranker.model = MagicMock()
        reranker._is_initialized = True
        reranker.model.compute_score.return_value = [0.85, 0.75, 0.95]

        results = await reranker.rerank("query", sample_results, top_n=3)
        assert len(results) == 2
        assert min(r.rerank_score for r in results) >= 0.8

    @pytest.mark.asyncio
    async def test_cleanup_clears_state(self: Any) -> None:
        reranker = BGEReranker(self._bge_config())
        reranker.model = MagicMock()
        reranker._is_initialized = True

        await reranker.cleanup()
        assert reranker.model is None
        assert reranker._is_initialized is False


class TestQwen3Reranker:
    """Focus on orchestration logic without loading actual weights."""

    @pytest.fixture
    def sample_results(self: Any) -> List[RetrievalResult]:
        doc = Document(id="doc", content="Qwen content")
        return [
            RetrievalResult(document=doc, score=0.5, rank=1, retrieval_method="dense"),
            RetrievalResult(document=doc, score=0.4, rank=2, retrieval_method="dense"),
        ]

    def _qwen_config(self: Any, **overrides: Any) -> RerankConfig:
        base = {
            "local_backend": "qwen_local",
            "local_model": "Qwen/Qwen3-Reranker-0.6B",
            "local_batch_size": 2,
        }
        base.update(overrides)
        return RerankConfig(**base)

    @pytest.mark.asyncio
    async def test_rerank_uses_prepared_scores(
        self: Any, sample_results: List[RetrievalResult]
    ) -> None:
        reranker = Qwen3Reranker(self._qwen_config())
        reranker._is_initialized = True
        reranker._format_instruction = MagicMock(return_value="prompt")
        reranker._prepare_inputs = MagicMock(return_value={})
        reranker._compute_scores = MagicMock(return_value=[0.8, 0.6])
        reranker._truncate_document = MagicMock(side_effect=lambda text: text)

        results = await reranker.rerank("query", sample_results, top_n=2)

        assert [round(r.rerank_score, 2) for r in results] == [0.8, 0.6]
        assert results[0].reranker_used == "qwen_local"
        reranker._prepare_inputs.assert_called_once()
        reranker._compute_scores.assert_called_once()

    def test_get_model_info_contains_metadata(self: Any) -> None:
        config = self._qwen_config()
        reranker = Qwen3Reranker(config)
        info = reranker.get_model_info()
        assert info["type"] == "qwen_local"
        assert info["model_name"] == "Qwen/Qwen3-Reranker-0.6B"
        assert info["initialized"] is False


class TestRerankingService:
    """Ensure service routes to the configured backend."""

    @pytest.fixture
    def sample_results(self: Any) -> List[RetrievalResult]:
        doc = Document(id="doc", content="content")
        return [
            RetrievalResult(document=doc, score=1.0, rank=1, retrieval_method="dense")
        ]

    def test_service_initializes_qwen_backend(self: Any) -> None:
        config = RerankConfig(local_backend="qwen_local")
        with patch("askme.rerank.rerank_service.Qwen3Reranker") as mock_qwen:
            service = RerankingService(config)
            mock_qwen.assert_called_once_with(config)
            assert service.local_backend == "qwen_local"
            assert service.local_reranker is mock_qwen.return_value

    def test_service_initializes_bge_backend(self: Any) -> None:
        config = RerankConfig(
            local_backend="bge_local", local_model="BAAI/bge-reranker-v2-m3"
        )
        with patch("askme.rerank.rerank_service.BGEReranker") as mock_bge:
            service = RerankingService(config)
            mock_bge.assert_called_once_with(config)
            assert service.local_backend == "bge_local"
            assert service.local_reranker is mock_bge.return_value

    @pytest.mark.asyncio
    async def test_rerank_calls_local_backend(
        self: Any, sample_results: List[RetrievalResult]
    ) -> None:
        config = RerankConfig(local_backend="qwen_local")
        with patch("askme.rerank.rerank_service.Qwen3Reranker") as mock_qwen:
            mock_instance = mock_qwen.return_value
            mock_instance.rerank = AsyncMock(
                return_value=[MagicMock(spec=RerankResult)]
            )
            service = RerankingService(config)

            results = await service.rerank("query", sample_results, top_n=1)

        mock_instance.rerank.assert_called_once_with("query", sample_results, 1)
        assert results == mock_instance.rerank.return_value

    @pytest.mark.asyncio
    async def test_rerank_by_method_qwen(self: Any) -> None:
        config = RerankConfig(local_backend="qwen_local")
        with patch("askme.rerank.rerank_service.Qwen3Reranker") as mock_qwen:
            mock_instance = mock_qwen.return_value
            mock_instance.rerank = AsyncMock(return_value=[])
            service = RerankingService(config)
            await service.rerank_by_method("query", [], method="qwen_local")

        mock_instance.rerank.assert_called_once()

    def test_get_available_methods(self: Any) -> None:
        config = RerankConfig(local_backend="qwen_local")
        with patch("askme.rerank.rerank_service.Qwen3Reranker"):
            service = RerankingService(config)
            assert service.get_available_methods() == ["qwen_local"]

    def test_get_service_info_contains_backend(self: Any) -> None:
        config = RerankConfig(local_backend="qwen_local")
        with patch("askme.rerank.rerank_service.Qwen3Reranker") as mock_qwen:
            mock_qwen.return_value.get_model_info.return_value = {"type": "qwen_local"}
            service = RerankingService(config)
            info = service.get_service_info()

        assert info["available_methods"] == ["qwen_local"]
        assert info["local_backend"] == "qwen_local"
        assert info["local_model"] == {"type": "qwen_local"}
