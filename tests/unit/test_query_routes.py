from typing import Any

"""
Unit tests for query API routes.
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from askme.api.routes.query import (
    Citation,
    QueryRequest,
    QueryResponse,
    RetrievalDebugInfo,
    RetrievalRequest,
    RetrievalResponse,
    explain_retrieval,
    find_similar_documents,
    query_documents,
    retrieve_documents,
    router,
)
from askme.core.config import Settings


@pytest.fixture
def mock_settings() -> Any:
    """Mock settings for testing."""
    settings = MagicMock(spec=Settings)
    settings.rerank.cohere_enabled = False
    return settings


@pytest.fixture
def mock_request() -> Any:
    """Mock FastAPI Request object."""
    mock_req = MagicMock()
    mock_req.app.state.embedding_service = None
    mock_req.app.state.retriever = None
    mock_req.app.state.reranking_service = None
    mock_req.app.state.generator = None
    return mock_req


class TestCitationModel:
    """Test Citation model validation."""

    def test_citation_creation(self: Any) -> None:
        """Test creating a citation with valid data."""
        citation = Citation(
            doc_id="doc_123",
            title="Test Document",
            content="This is test content",
            start=0,
            end=20,
            score=0.95,
            metadata={"author": "Test Author"},
        )

        assert citation.doc_id == "doc_123"
        assert citation.title == "Test Document"
        assert citation.content == "This is test content"
        assert citation.start == 0
        assert citation.end == 20
        assert citation.score == 0.95
        assert citation.metadata == {"author": "Test Author"}

    def test_citation_score_validation(self: Any) -> None:
        """Test citation score validation."""
        with pytest.raises(ValueError):
            Citation(
                doc_id="doc_123",
                title="Test",
                content="Test",
                start=0,
                end=10,
                score=1.5,  # Invalid score > 1
            )

    def test_citation_optional_metadata(self: Any) -> None:
        """Test citation with optional metadata."""
        citation = Citation(
            doc_id="doc_123", title="Test", content="Test", start=0, end=10, score=0.8
        )
        assert citation.metadata is None


class TestRetrievalDebugInfoModel:
    """Test RetrievalDebugInfo model."""

    def test_debug_info_creation(self: Any) -> None:
        """Test creating debug info with all fields."""
        debug = RetrievalDebugInfo(
            bm25_hits=25,
            dense_hits=30,
            fusion_method="rrf",
            alpha=0.5,
            rrf_k=60,
            rerank_model="bge_local",
            rerank_scores=[0.9, 0.8, 0.7],
            latency_ms=1200,
            embedding_latency_ms=150,
            search_latency_ms=800,
            rerank_latency_ms=250,
            overlap_hits=15,
        )

        assert debug.bm25_hits == 25
        assert debug.dense_hits == 30
        assert debug.fusion_method == "rrf"
        assert debug.alpha == 0.5
        assert debug.rrf_k == 60
        assert debug.rerank_model == "bge_local"
        assert debug.rerank_scores == [0.9, 0.8, 0.7]
        assert debug.latency_ms == 1200
        assert debug.overlap_hits == 15

    def test_debug_info_optional_fields(self: Any) -> None:
        """Test debug info with minimal required fields."""
        debug = RetrievalDebugInfo(
            bm25_hits=10,
            dense_hits=15,
            fusion_method="alpha",
            rerank_model="cohere",
            latency_ms=800,
            embedding_latency_ms=100,
            search_latency_ms=600,
            rerank_latency_ms=100,
        )

        assert debug.alpha is None
        assert debug.rrf_k is None
        assert debug.rerank_scores is None
        assert debug.overlap_hits is None
        assert debug.error is None


class TestQueryRequestModel:
    """Test QueryRequest model validation."""

    def test_query_request_defaults(self: Any) -> None:
        """Test query request with default values."""
        req = QueryRequest(q="test query")

        assert req.q == "test query"
        assert req.topk == 50
        assert req.alpha == 0.5
        assert req.use_rrf is True
        assert req.rrf_k == 60
        assert req.use_hyde is False
        assert req.use_rag_fusion is False
        assert req.reranker == "bge_local"
        assert req.max_passages == 8
        assert req.filters is None
        assert req.include_debug is False

    def test_query_request_validation(self: Any) -> None:
        """Test query request validation."""
        # Valid request
        req = QueryRequest(
            q="test query",
            topK=25,
            alpha=0.7,
            use_rrf=False,
            rrf_k=80,
            use_hyde=True,
            use_rag_fusion=True,
            reranker="cohere",
            max_passages=5,
            filters={"tag": "important"},
            include_debug=True,
        )

        assert req.topk == 25  # Alias handling
        assert req.alpha == 0.7
        assert req.use_rrf is False

    def test_query_request_validation_errors(self: Any) -> None:
        """Test query request validation errors."""
        # Empty query
        with pytest.raises(ValueError):
            QueryRequest(q="")

        # Invalid topK
        with pytest.raises(ValueError):
            QueryRequest(q="test", topK=0)

        with pytest.raises(ValueError):
            QueryRequest(q="test", topK=150)

        # Invalid alpha
        with pytest.raises(ValueError):
            QueryRequest(q="test", alpha=-0.1)

        with pytest.raises(ValueError):
            QueryRequest(q="test", alpha=1.1)


class TestRetrievalRequestModel:
    """Test RetrievalRequest model."""

    def test_retrieval_request_defaults(self: Any) -> None:
        """Test retrieval request defaults."""
        req = RetrievalRequest(q="test query")

        assert req.q == "test query"
        assert req.topk == 50
        assert req.use_hyde is False
        assert req.use_rag_fusion is False
        assert req.alpha == 0.5
        assert req.use_rrf is True
        assert req.rrf_k == 60
        assert req.filters is None

    def test_retrieval_request_custom_values(self: Any) -> None:
        """Test retrieval request with custom values."""
        req = RetrievalRequest(
            q="test query",
            topK=20,
            use_hyde=True,
            use_rag_fusion=True,
            alpha=0.3,
            use_rrf=False,
            rrf_k=40,
            filters={"category": "science"},
        )

        assert req.topk == 20
        assert req.use_hyde is True
        assert req.use_rag_fusion is True
        assert req.alpha == 0.3
        assert req.use_rrf is False
        assert req.rrf_k == 40
        assert req.filters == {"category": "science"}


class TestQueryDocuments:
    """Test query_documents endpoint function."""

    @pytest.mark.asyncio
    async def test_cohere_validation_error(
        self: Any, mock_request: MagicMock, mock_settings: MagicMock
    ) -> None:
        """Test Cohere reranker validation error."""
        req = QueryRequest(q="test query", reranker="cohere")
        mock_settings.rerank.cohere_enabled = False

        with pytest.raises(HTTPException) as exc_info:
            await query_documents(req, mock_request, mock_settings)

        assert exc_info.value.status_code == 400
        assert "Cohere reranker not enabled" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_cohere_enabled_passes_validation(
        self: Any, mock_request: MagicMock, mock_settings: MagicMock
    ) -> None:
        """Test that Cohere validation passes when enabled."""
        req = QueryRequest(q="test query", reranker="cohere")
        mock_settings.rerank.cohere_enabled = True

        # Should not raise HTTPException for Cohere validation
        # Will fall back to mock response due to missing services
        response = await query_documents(req, mock_request, mock_settings)

        assert isinstance(response, QueryResponse)
        assert response.answer is not None
        assert len(response.citations) > 0

    @pytest.mark.asyncio
    async def test_mock_response_fallback(
        self: Any, mock_request: MagicMock, mock_settings: MagicMock
    ) -> None:
        """Test fallback to mock response when services are not available."""
        req = QueryRequest(
            q="test query",
            include_debug=True,
            use_rrf=False,
            alpha=0.7,
            reranker="bge_local",
        )

        response = await query_documents(req, mock_request, mock_settings)

        assert isinstance(response, QueryResponse)
        assert response.answer is not None
        assert len(response.citations) == 2

        # Check citations
        assert response.citations[0].doc_id == "doc_001"
        assert response.citations[0].title == "Sample Document 1"
        assert response.citations[0].score == 0.92
        assert response.citations[1].doc_id == "doc_002"
        assert response.citations[1].score == 0.87

        # Check debug info
        assert response.retrieval_debug is not None
        assert response.retrieval_debug.bm25_hits == 25
        assert response.retrieval_debug.dense_hits == 30
        assert response.retrieval_debug.fusion_method == "alpha"
        assert response.retrieval_debug.alpha == 0.7
        assert response.retrieval_debug.rrf_k is None
        assert response.retrieval_debug.rerank_model == "bge_local"

        # Check response structure
        assert isinstance(response.timestamp, datetime)

        # Query ID should be valid UUID
        UUID(response.query_id)  # Will raise ValueError if invalid

    @pytest.mark.asyncio
    async def test_mock_response_with_rrf(
        self: Any, mock_request: MagicMock, mock_settings: MagicMock
    ) -> None:
        """Test mock response with RRF enabled."""
        req = QueryRequest(q="test query", include_debug=True, use_rrf=True, rrf_k=80)

        response = await query_documents(req, mock_request, mock_settings)

        assert response.retrieval_debug is not None
        assert response.retrieval_debug.fusion_method == "rrf"
        assert response.retrieval_debug.rrf_k == 80

    @pytest.mark.asyncio
    async def test_no_debug_info_when_disabled(
        self: Any, mock_request: MagicMock, mock_settings: MagicMock
    ) -> None:
        """Test that debug info is None when include_debug=False."""
        req = QueryRequest(q="test query", include_debug=False)

        response = await query_documents(req, mock_request, mock_settings)

        assert response.retrieval_debug is None

    @pytest.mark.asyncio
    async def test_mock_response_with_generator(
        self: Any, mock_request: MagicMock, mock_settings: MagicMock
    ) -> None:
        """Test mock response fallback with generator available."""
        # Mock generator service
        mock_generator = AsyncMock()
        mock_generator.generate.return_value = (
            "Generated answer with citations [Doc 001] [Doc 002]"
        )
        mock_request.app.state.generator = mock_generator

        req = QueryRequest(q="test query about AI")

        response = await query_documents(req, mock_request, mock_settings)

        assert response.answer == "Generated answer with citations [Doc 001] [Doc 002]"
        # Should have called generator with passages
        mock_generator.generate.assert_called_once()
        call_args = mock_generator.generate.call_args
        assert call_args[0][0] == "test query about AI"  # query
        assert len(call_args[0][1]) == 2  # passages

    @pytest.mark.asyncio
    async def test_mock_response_generator_exception(
        self: Any, mock_request: MagicMock, mock_settings: MagicMock
    ) -> None:
        """Test mock response when generator raises exception."""
        # Mock generator that raises exception
        mock_generator = AsyncMock()
        mock_generator.generate.side_effect = Exception("Generator failed")
        mock_request.app.state.generator = mock_generator

        req = QueryRequest(q="test query")

        response = await query_documents(req, mock_request, mock_settings)

        # Should fall back to default answer template
        assert "Based on the provided context" in response.answer
        assert "[Doc 001: Sample Document 1]" in response.answer
        assert "[Doc 002: Sample Document 2]" in response.answer

    @pytest.mark.asyncio
    async def test_mock_response_no_request_context(
        self: Any, mock_settings: MagicMock
    ) -> None:
        """Test mock response handling when request is None."""
        req = QueryRequest(q="test query")

        response = await query_documents(req, None, mock_settings)

        # Should fall back to default answer template
        assert "Based on the provided context" in response.answer


class TestRetrieveDocuments:
    """Test retrieve_documents endpoint function."""

    @pytest.mark.asyncio
    async def test_mock_retrieval_response(
        self: Any, mock_request: MagicMock, mock_settings: MagicMock
    ) -> None:
        """Test mock retrieval response when services unavailable."""
        req = RetrievalRequest(
            q="test retrieval query",
            topK=25,
            use_hyde=True,
            use_rag_fusion=True,
            alpha=0.3,
            use_rrf=False,
            rrf_k=40,
        )

        response = await retrieve_documents(req, mock_request, mock_settings)

        assert isinstance(response, RetrievalResponse)
        assert len(response.documents) == 1

        # Check document structure
        doc = response.documents[0]
        assert doc.doc_id == "doc_001"
        assert doc.title == "Retrieved Document 1"
        assert doc.content == "Content snippet from document 1..."
        assert doc.score == 0.95
        assert doc.metadata == {"source": "manual", "tags": ["important"]}

        # Check debug info
        assert response.retrieval_debug is not None
        assert response.retrieval_debug.bm25_hits == 20
        assert response.retrieval_debug.dense_hits == 35
        assert response.retrieval_debug.fusion_method == "alpha"
        assert response.retrieval_debug.alpha == 0.3
        assert response.retrieval_debug.rrf_k == 40
        assert response.retrieval_debug.rerank_model == "bge_local"
        assert response.retrieval_debug.rerank_latency_ms == 0

        # Check response structure
        assert isinstance(response.timestamp, datetime)
        UUID(response.query_id)  # Will raise ValueError if invalid

    @pytest.mark.asyncio
    async def test_retrieval_with_rrf(
        self: Any, mock_request: MagicMock, mock_settings: MagicMock
    ) -> None:
        """Test retrieval response with RRF enabled."""
        req = RetrievalRequest(q="test query", use_rrf=True, rrf_k=70)

        response = await retrieve_documents(req, mock_request, mock_settings)

        assert response.retrieval_debug.fusion_method == "rrf"
        assert response.retrieval_debug.rrf_k == 70


class TestFindSimilarDocuments:
    """Test find_similar_documents endpoint function."""

    @pytest.mark.asyncio
    async def test_find_similar_basic(self: Any, mock_settings: MagicMock) -> None:
        """Test basic similar documents functionality."""
        response = await find_similar_documents("test_doc_123", 5, mock_settings)

        assert isinstance(response, list)
        assert len(response) == 1

        similar_doc = response[0]
        assert similar_doc.doc_id == "similar_001"
        assert similar_doc.title == "Similar Document"
        assert similar_doc.content == "Content similar to the target document..."
        assert similar_doc.score == 0.89

    @pytest.mark.asyncio
    async def test_find_similar_with_limit(self: Any, mock_settings: MagicMock) -> None:
        """Test similar documents with custom limit."""
        response = await find_similar_documents(
            "test_doc_456", limit=15, settings=mock_settings
        )

        # Currently returns single mock document regardless of limit
        # This tests the parameter handling
        assert isinstance(response, list)
        assert len(response) == 1


class TestExplainRetrieval:
    """Test explain_retrieval endpoint function."""

    @pytest.mark.asyncio
    async def test_explain_retrieval_basic(self: Any, mock_settings: MagicMock) -> None:
        """Test basic retrieval explanation."""
        req = RetrievalRequest(q="explain this query", topK=10, alpha=0.6)

        response = await explain_retrieval(req, mock_settings)

        assert isinstance(response, dict)
        assert response["query"] == "explain this query"
        assert response["explanation"] == "Mock explanation of retrieval process"
        assert response["matching_terms"] == ["term1", "term2"]
        assert response["semantic_similarity"] == 0.85

        ranking_factors = response["ranking_factors"]
        assert ranking_factors["bm25_score"] == 0.7
        assert ranking_factors["dense_score"] == 0.9
        assert ranking_factors["final_score"] == 0.82

    @pytest.mark.asyncio
    async def test_explain_retrieval_with_filters(
        self: Any, mock_settings: MagicMock
    ) -> None:
        """Test retrieval explanation with filters."""
        req = RetrievalRequest(
            q="filtered query", filters={"category": "research", "year": 2024}
        )

        response = await explain_retrieval(req, mock_settings)

        # Mock implementation doesn't use filters yet, but tests parameter handling
        assert response["query"] == "filtered query"


class TestResponseModels:
    """Test response model validation."""

    def test_query_response_creation(self: Any) -> None:
        """Test QueryResponse model creation."""
        citations = [
            Citation(
                doc_id="doc1",
                title="Test Doc",
                content="Test content",
                start=0,
                end=12,
                score=0.9,
            )
        ]

        debug_info = RetrievalDebugInfo(
            bm25_hits=10,
            dense_hits=15,
            fusion_method="rrf",
            rerank_model="bge_local",
            latency_ms=800,
            embedding_latency_ms=100,
            search_latency_ms=600,
            rerank_latency_ms=100,
        )

        response = QueryResponse(
            answer="Test answer",
            citations=citations,
            query_id="12345-678-90",
            timestamp=datetime.now(),
            retrieval_debug=debug_info,
        )

        assert response.answer == "Test answer"
        assert len(response.citations) == 1
        assert response.query_id == "12345-678-90"
        assert response.retrieval_debug == debug_info

    def test_retrieval_response_creation(self: Any) -> None:
        """Test RetrievalResponse model creation."""
        documents = [
            Citation(
                doc_id="retrieved1",
                title="Retrieved Doc",
                content="Retrieved content",
                start=0,
                end=17,
                score=0.85,
            )
        ]

        debug_info = RetrievalDebugInfo(
            bm25_hits=20,
            dense_hits=25,
            fusion_method="alpha",
            alpha=0.5,
            rerank_model="cohere",
            latency_ms=1000,
            embedding_latency_ms=200,
            search_latency_ms=700,
            rerank_latency_ms=100,
        )

        response = RetrievalResponse(
            documents=documents,
            query_id="retrieval-123",
            timestamp=datetime.now(),
            retrieval_debug=debug_info,
        )

        assert len(response.documents) == 1
        assert response.query_id == "retrieval-123"
        assert response.retrieval_debug == debug_info


class TestRouterIntegration:
    """Test router integration."""

    def test_router_has_expected_endpoints(self: Any) -> None:
        """Test that router has all expected endpoints."""
        # Get all routes from router
        routes = [route.path for route in router.routes]

        # Check main endpoints exist
        assert "/" in routes  # query_documents
        assert "/retrieve" in routes  # retrieve_documents
        assert "/similar/{doc_id}" in routes  # find_similar_documents
        assert "/explain" in routes  # explain_retrieval

    def test_router_endpoint_methods(self: Any) -> None:
        """Test that endpoints have correct HTTP methods."""
        route_methods = {route.path: route.methods for route in router.routes}

        assert "POST" in route_methods["/"]
        assert "POST" in route_methods["/retrieve"]
        assert "GET" in route_methods["/similar/{doc_id}"]
        assert "POST" in route_methods["/explain"]
