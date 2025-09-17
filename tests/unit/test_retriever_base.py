"""
Unit tests for base retriever functionality.
"""

import pytest

from askme.retriever.base import (
    Document,
    HybridSearchParams,
    RetrievalResult,
    SearchFusion,
)


class TestDocument:
    """Test Document class."""

    def test_document_creation(self) -> None:
        """Test creating a document."""
        doc = Document(
            id="test_001",
            content="Test document content",
            metadata={"source": "test.txt"},
        )
        assert doc.id == "test_001"
        assert doc.content == "Test document content"
        assert doc.metadata["source"] == "test.txt"
        assert doc.embedding is None
        assert doc.sparse_embedding is None

    def test_document_with_embeddings(self) -> None:
        """Test document with embeddings."""
        doc = Document(
            id="test_002",
            content="Test content",
            embedding=[0.1, 0.2, 0.3],
            sparse_embedding={1: 0.5, 3: 0.8},
        )
        assert doc.embedding == [0.1, 0.2, 0.3]
        assert doc.sparse_embedding == {1: 0.5, 3: 0.8}


class TestRetrievalResult:
    """Test RetrievalResult class."""

    def test_retrieval_result_creation(self) -> None:
        """Test creating a retrieval result."""
        doc = Document(id="test", content="content", metadata={})
        result = RetrievalResult(
            document=doc,
            score=0.95,
            rank=1,
            retrieval_method="dense",
        )
        assert result.document.id == "test"
        assert result.score == 0.95
        assert result.rank == 1
        assert result.retrieval_method == "dense"
        assert result.debug_info is None

    def test_retrieval_result_with_debug_info(self) -> None:
        """Test retrieval result with debug info."""
        doc = Document(id="test", content="content", metadata={})
        result = RetrievalResult(
            document=doc,
            score=0.85,
            rank=2,
            retrieval_method="hybrid",
            debug_info={"fusion": "alpha", "alpha": 0.5},
        )
        assert result.debug_info["fusion"] == "alpha"
        assert result.debug_info["alpha"] == 0.5


class TestHybridSearchParams:
    """Test HybridSearchParams class."""

    def test_default_params(self) -> None:
        """Test default hybrid search parameters."""
        params = HybridSearchParams()
        assert params.topk == 50
        assert params.alpha == 0.5
        assert params.use_rrf is True
        assert params.rrf_k == 60
        assert params.filters is None

    def test_custom_params(self) -> None:
        """Test custom hybrid search parameters."""
        filters = {"category": "tech"}
        params = HybridSearchParams(topk=100, alpha=0.7, use_rrf=False, filters=filters)
        assert params.topk == 100
        assert params.alpha == 0.7
        assert params.use_rrf is False
        assert params.filters == filters


class TestSearchFusion:
    """Test SearchFusion utility methods."""

    def create_sample_results(self):
        """Create sample retrieval results for testing."""
        docs = [
            Document(id=f"doc_{i}", content=f"Content {i}", metadata={})
            for i in range(3)
        ]

        dense_results = [
            RetrievalResult(docs[0], 0.9, 1, "dense"),
            RetrievalResult(docs[1], 0.8, 2, "dense"),
            RetrievalResult(docs[2], 0.7, 3, "dense"),
        ]

        sparse_results = [
            RetrievalResult(docs[1], 0.85, 1, "sparse"),
            RetrievalResult(docs[2], 0.75, 2, "sparse"),
            RetrievalResult(docs[0], 0.65, 3, "sparse"),
        ]

        return dense_results, sparse_results

    def test_alpha_fusion(self) -> None:
        """Test alpha-weighted fusion."""
        dense_results, sparse_results = self.create_sample_results()

        fused = SearchFusion.alpha_fusion(dense_results, sparse_results, alpha=0.5)

        assert len(fused) == 3
        # Results should be ordered by combined score
        assert fused[0].document.id == "doc_1"  # Best combined score

    def test_reciprocal_rank_fusion(self) -> None:
        """Test reciprocal rank fusion."""
        dense_results, sparse_results = self.create_sample_results()

        fused = SearchFusion.reciprocal_rank_fusion(
            [dense_results, sparse_results], k=60
        )

        assert len(fused) == 3
        # Check that RRF scores are calculated
        assert all(hasattr(result, "score") for result in fused)

    def test_empty_results_fusion(self) -> None:
        """Test fusion with empty results."""
        empty_results = []
        dense_results, _ = self.create_sample_results()

        fused = SearchFusion.alpha_fusion(dense_results, empty_results, alpha=0.5)

        assert len(fused) == len(dense_results)
