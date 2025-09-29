from typing import Any

"""
Unit tests for configuration management.
"""

import pytest

from askme.core.config import (
    DatabaseConfig,
    EmbeddingConfig,
    HybridConfig,
    RerankConfig,
    Settings,
)


class TestDatabaseConfig:
    """Test database configuration."""

    def test_default_values(self: Any) -> None:
        """Test default configuration values."""
        config = DatabaseConfig()
        assert config.milvus.host == "localhost"
        assert config.milvus.port == 19530
        assert config.milvus.collection_name == "askme_hybrid"

    def test_custom_values(self: Any) -> None:
        """Test custom configuration values."""
        config = DatabaseConfig(
            host="custom-host", port=9999, collection_name="custom_collection"
        )
        assert config.host == "custom-host"
        assert config.port == 9999
        assert config.collection_name == "custom_collection"


class TestEmbeddingConfig:
    """Test embedding configuration."""

    def test_default_values(self: Any) -> None:
        """Test default embedding configuration."""
        config = EmbeddingConfig()
        assert config.backend == "qwen3-hybrid"
        assert config.model == "Qwen/Qwen3-Embedding-8B"
        assert config.dimension == 4096
        assert config.batch_size == 16
        # Test sparse config defaults
        assert config.sparse.enabled is True
        assert config.sparse.backend == "bge_m3"
        assert config.sparse.model == "BAAI/bge-m3"

    def test_custom_values(self: Any) -> None:
        """Test custom embedding configuration."""
        config = EmbeddingConfig(
            backend="custom-backend", model="custom-model", dimension=512, batch_size=16
        )
        assert config.backend == "custom-backend"
        assert config.model == "custom-model"
        assert config.dimension == 512
        assert config.batch_size == 16


class TestHybridConfig:
    """Test hybrid search configuration."""

    def test_default_values(self: Any) -> None:
        """Test default hybrid search configuration."""
        config = HybridConfig()
        assert config.alpha == 0.5
        assert config.use_rrf is True
        assert config.rrf_k == 60
        assert config.topk == 50

    def test_alpha_validation(self: Any) -> None:
        """Test alpha value validation."""
        # Valid alpha values
        config = HybridConfig(alpha=0.0)
        assert config.alpha == 0.0

        config = HybridConfig(alpha=1.0)
        assert config.alpha == 1.0

        config = HybridConfig(alpha=0.5)
        assert config.alpha == 0.5


class TestRerankConfig:
    """Test reranking configuration."""

    def test_default_values(self: Any) -> None:
        """Test default reranking configuration."""
        config = RerankConfig()
        assert config.local_enabled is True
        assert config.top_n == 8
        assert config.score_threshold == 0.0
        assert config.local_backend == "qwen_local"
        assert config.local_model == "Qwen/Qwen3-Reranker-8B"

    def test_custom_values(self: Any) -> None:
        """Test custom reranking configuration."""
        config = RerankConfig(
            local_enabled=False,
            top_n=5,
            score_threshold=0.7,
            local_backend="bge_local",
            local_model="BAAI/bge-reranker-v2.5",
        )
        assert config.local_enabled is False
        assert config.top_n == 5
        assert config.score_threshold == 0.7
        assert config.local_backend == "bge_local"
        assert config.local_model == "BAAI/bge-reranker-v2.5"


class TestSettings:
    """Test main settings class."""

    def test_default_settings(self: Any) -> None:
        """Test default settings creation."""
        settings = Settings()
        assert settings.vector_backend == "weaviate"
        assert isinstance(settings.database, DatabaseConfig)
        assert isinstance(settings.embedding, EmbeddingConfig)
        assert isinstance(settings.hybrid, HybridConfig)
        assert isinstance(settings.rerank, RerankConfig)

    def test_nested_config_access(self: Any) -> None:
        """Test accessing nested configuration values."""
        settings = Settings()
        assert settings.database.host == "localhost"
        assert settings.embedding.model == "Qwen/Qwen3-Embedding-8B"
        assert settings.hybrid.alpha == 0.5
        assert settings.rerank.top_n == 8
