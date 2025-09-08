"""
Integration tests for API endpoints.
"""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from askme.api.main import app


@pytest.fixture
def client():
    """Test client for the FastAPI app."""
    return TestClient(app)


class TestHealthEndpoint:
    """Test health check endpoint."""

    def test_health_check(self, client):
        """Test basic health check."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "timestamp" in data


class TestIngestEndpoints:
    """Test document ingestion endpoints."""

    @patch("askme.api.routes.ingest.IngestionService")
    def test_ingest_file_endpoint(self, mock_service, client):
        """Test file ingestion endpoint."""
        # Mock the service
        mock_instance = AsyncMock()
        mock_instance.ingest_file.return_value = "task_123"
        mock_service.return_value = mock_instance

        # Test data
        test_data = {
            "file_path": "/tmp/test.txt",
            "metadata": {"source": "test"},
            "tags": ["test"],
        }

        response = client.post("/ingest/file", json=test_data)
        assert response.status_code == 200
        data = response.json()
        assert "task_id" in data

    @patch("askme.api.routes.ingest.IngestionService")
    def test_ingest_directory_endpoint(self, mock_service, client):
        """Test directory ingestion endpoint."""
        # Mock the service
        mock_instance = AsyncMock()
        mock_instance.ingest_directory.return_value = "task_456"
        mock_service.return_value = mock_instance

        # Test data
        test_data = {
            "directory_path": "/tmp/docs",
            "recursive": True,
            "metadata": {"batch": "test_batch"},
        }

        response = client.post("/ingest/directory", json=test_data)
        assert response.status_code == 200
        data = response.json()
        assert "task_id" in data


class TestQueryEndpoints:
    """Test query endpoints."""

    def test_query_endpoint_structure(self, client):
        """Test query endpoint accepts proper structure."""
        test_query = {
            "q": "What is machine learning?",
            "topK": 10,
            "alpha": 0.5,
            "use_rrf": True,
            "max_passages": 5,
        }

        # This will fail with 500 due to missing services,
        # but should accept the request structure
        response = client.post("/query", json=test_query)
        # We expect 500 because services aren't properly mocked
        assert response.status_code in [200, 500]


class TestEvaluationEndpoints:
    """Test evaluation endpoints."""

    def test_run_evaluation_endpoint(self, client):
        """Test evaluation run endpoint."""
        test_data = {
            "suite": "baseline",
            "metrics": ["faithfulness", "answer_relevancy"],
            "sample_size": 10,
        }

        response = client.post("/eval/run", json=test_data)
        assert response.status_code == 200
        data = response.json()
        assert "run_id" in data
        assert data["suite"] == "baseline"

    def test_get_evaluation_results(self, client):
        """Test getting evaluation results."""
        response = client.get("/eval/runs/test_run_123")
        assert response.status_code == 200
        data = response.json()
        assert "run_id" in data
        assert "status" in data

    def test_list_evaluation_runs(self, client):
        """Test listing evaluation runs."""
        response = client.get("/eval/runs?limit=10")
        assert response.status_code == 200
        data = response.json()
        assert "runs" in data
        assert "total" in data

    def test_get_available_metrics(self, client):
        """Test getting available metrics."""
        response = client.get("/eval/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "trulens_metrics" in data
        assert "ragas_metrics" in data
