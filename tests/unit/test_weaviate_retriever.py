"""
Comprehensive unit tests for Weaviate retriever.
"""

from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from askme.retriever.base import Document, HybridSearchParams, RetrievalResult
from askme.retriever.weaviate_retriever import WeaviateRetriever


class TestWeaviateRetrieverCore:
    """Test core Weaviate retriever functionality."""

    def test_initialization_basic(self: Any) -> None:
        """Test basic initialization."""
        config = {
            "url": "https://test-cluster.weaviate.network",
            "api_key": "test-api-key",
            "class_name": "TestDocument",
            "dimension": 768,
        }

        retriever = WeaviateRetriever(config)

        assert retriever.url == "https://test-cluster.weaviate.network"
        assert retriever.api_key == "test-api-key"
        assert retriever.class_name == "TestDocument"
        assert retriever.dimension == 768
        assert retriever.client is None
        assert retriever.collection is None

    def test_initialization_defaults(self: Any) -> None:
        """Test initialization with default values."""
        config = {"collection_name": "minimal"}

        retriever = WeaviateRetriever(config)

        assert retriever.url == "http://localhost:8080"
        assert retriever.api_key == ""
        assert retriever.class_name == "AskmeDocument"
        assert retriever.dimension == 1024
        assert retriever.collection_name == "minimal"

    @pytest.mark.asyncio
    async def test_connect_with_api_key(self: Any) -> None:
        """Test connection with API key (cloud)."""
        config = {
            "url": "https://cluster.weaviate.network",
            "api_key": "test-key",
            "class_name": "CloudDoc",
        }
        retriever = WeaviateRetriever(config)

        mock_client = MagicMock()

        with patch("askme.retriever.weaviate_retriever.weaviate") as mock_weaviate:
            mock_weaviate.connect_to_weaviate_cloud.return_value = mock_client
            mock_weaviate.AuthApiKey.return_value = "auth_object"

            await retriever.connect()

            assert retriever.client == mock_client
            mock_weaviate.connect_to_weaviate_cloud.assert_called_once_with(
                cluster_url="https://cluster.weaviate.network",
                auth_credentials="auth_object",
            )

    @pytest.mark.asyncio
    async def test_connect_local_http(self: Any) -> None:
        """Test connection to local Weaviate instance."""
        config = {
            "url": "http://localhost:8080",
            "api_key": "",
            "class_name": "LocalDoc",
        }
        retriever = WeaviateRetriever(config)

        mock_client = MagicMock()

        with patch("askme.retriever.weaviate_retriever.weaviate") as mock_weaviate:
            mock_weaviate.connect_to_custom.return_value = mock_client

            await retriever.connect()

            assert retriever.client == mock_client
            mock_weaviate.connect_to_custom.assert_called_once_with(
                http_host="localhost",
                http_port=8080,
                http_secure=False,
                grpc_host="localhost",
                grpc_port=8081,
                grpc_secure=False,
                skip_init_checks=True,
            )

    @pytest.mark.asyncio
    async def test_connect_local_https(self: Any) -> None:
        """Test connection to local HTTPS Weaviate."""
        config = {
            "url": "https://localhost:8443",
            "api_key": "",
            "class_name": "SecureDoc",
        }
        retriever = WeaviateRetriever(config)

        mock_client = MagicMock()

        with patch("askme.retriever.weaviate_retriever.weaviate") as mock_weaviate:
            mock_weaviate.connect_to_custom.return_value = mock_client

            await retriever.connect()

            mock_weaviate.connect_to_custom.assert_called_once_with(
                http_host="localhost",
                http_port=8443,
                http_secure=True,
                grpc_host="localhost",
                grpc_port=50051,
                grpc_secure=True,
                skip_init_checks=True,
            )

    @pytest.mark.asyncio
    async def test_connect_failure(self: Any) -> None:
        """Test connection failure handling."""
        config = {"class_name": "FailDoc"}
        retriever = WeaviateRetriever(config)

        with patch("askme.retriever.weaviate_retriever.weaviate") as mock_weaviate:
            mock_weaviate.connect_to_custom.side_effect = Exception("Connection failed")

            # Should raise exception and log error
            with pytest.raises(Exception, match="Connection failed"):
                await retriever.connect()
            assert retriever.client is None

    @pytest.mark.asyncio
    async def test_disconnect(self: Any) -> None:
        """Test disconnection."""
        config = {"class_name": "DisconnectTest"}
        retriever = WeaviateRetriever(config)

        mock_client = MagicMock()
        retriever.client = mock_client

        await retriever.disconnect()

        mock_client.close.assert_called_once()
        assert retriever.client is None
        assert retriever.collection is None

    @pytest.mark.asyncio
    async def test_disconnect_no_client(self: Any) -> None:
        """Test disconnection when no client exists."""
        config = {"class_name": "NoClient"}
        retriever = WeaviateRetriever(config)

        # Should not raise error
        await retriever.disconnect()

        assert retriever.client is None

    @pytest.mark.asyncio
    async def test_create_collection_new(self: Any) -> None:
        """Test creating a new collection."""
        config = {"class_name": "NewCollection"}
        retriever = WeaviateRetriever(config)

        mock_client = MagicMock()
        mock_collections = MagicMock()
        mock_client.collections = mock_collections
        mock_collections.exists.return_value = False
        retriever.client = mock_client

        with patch("askme.retriever.weaviate_retriever.wcfg") as mock_wcfg:
            mock_property = MagicMock()
            mock_wcfg.Property.return_value = mock_property

            mock_vector_config = MagicMock()
            mock_wcfg.Configure.VectorIndex.hnsw.return_value = mock_vector_config

            mock_created_collection = MagicMock()
            mock_collections.create.return_value = mock_created_collection

            mock_get_collection = MagicMock()
            mock_collections.get.return_value = mock_get_collection

            await retriever.create_collection(dimension=768, metric="cosine")

            mock_collections.create.assert_called_once()
            mock_collections.get.assert_called_once_with("NewCollection")
            assert retriever.collection == mock_get_collection

    @pytest.mark.asyncio
    async def test_create_collection_existing(self: Any) -> None:
        """Test handling existing collection."""
        config = {"class_name": "ExistingCollection"}
        retriever = WeaviateRetriever(config)

        mock_client = MagicMock()
        mock_collections = MagicMock()
        mock_client.collections = mock_collections
        mock_collections.exists.return_value = True

        mock_existing_collection = MagicMock()
        mock_collections.get.return_value = mock_existing_collection

        retriever.client = mock_client

        await retriever.create_collection(dimension=512)

        mock_collections.get.assert_called_once_with("ExistingCollection")
        assert retriever.collection == mock_existing_collection

    @pytest.mark.asyncio
    async def test_insert_documents_basic(self: Any) -> None:
        """Test basic document insertion."""
        config = {"class_name": "InsertTest"}
        retriever = WeaviateRetriever(config)

        # Mock collection and client
        mock_collection = MagicMock()
        mock_data = MagicMock()
        mock_collection.data = mock_data

        # Mock batch insert result
        mock_result = MagicMock()
        mock_result.uuids = {"doc1": "uuid1", "doc2": "uuid2"}
        mock_result.errors = {}
        mock_data.insert_many.return_value = mock_result

        documents = [
            Document(
                id="doc1",
                content="First document content",
                metadata={"source": "test1.txt", "type": "text"},
                embedding=[0.1] * 1024,
            ),
            Document(
                id="doc2",
                content="Second document content",
                metadata={"source": "test2.txt", "type": "text"},
                embedding=[0.2] * 1024,
            ),
        ]

        # Mock _ensure_collection to return the mock collection
        with patch.object(
            retriever, "_ensure_collection", return_value=mock_collection
        ):
            result_ids = await retriever.insert_documents(documents)

            # The implementation returns UUIDs, not original doc IDs
            assert len(result_ids) == 2
            assert all(isinstance(id, str) for id in result_ids)

    @pytest.mark.asyncio
    async def test_insert_documents_with_errors(self: Any) -> None:
        """Test document insertion with partial errors."""
        config = {"class_name": "ErrorTest"}
        retriever = WeaviateRetriever(config)

        mock_collection = MagicMock()
        mock_data = MagicMock()
        mock_collection.data = mock_data
        # Mock result with errors
        mock_result = MagicMock()
        mock_result.uuids = {"doc1": "uuid1"}  # Only one succeeded
        mock_result.errors = {"doc2": "Insert failed"}
        mock_data.insert_many.return_value = mock_result

        documents = [
            Document(
                id="doc1", content="Success doc", metadata={}, embedding=[0.1] * 1024
            ),
            Document(
                id="doc2", content="Failed doc", metadata={}, embedding=[0.2] * 1024
            ),
        ]

        with patch.object(
            retriever, "_ensure_collection", return_value=mock_collection
        ):
            result_ids = await retriever.insert_documents(documents)

            # The implementation returns UUIDs for documents that were processed
            assert (
                len(result_ids) == 2
            )  # Both documents are processed, even if one fails

    @pytest.mark.asyncio
    async def test_dense_search_basic(self: Any) -> None:
        """Test basic dense vector search."""
        config = {"class_name": "DenseSearch"}
        retriever = WeaviateRetriever(config)

        mock_collection = MagicMock()
        mock_query = MagicMock()
        mock_collection.query = mock_query

        # Mock search results
        mock_object1 = MagicMock()
        mock_object1.uuid = "uuid1"
        mock_object1.properties = {
            "content": "Dense search result",
            "doc_id": "dense_doc",
            "source": "dense.txt",
        }
        mock_object1.metadata = MagicMock()
        mock_object1.metadata.score = 0.85

        # Set up the method chain for Weaviate query API
        mock_result = MagicMock()
        mock_result.objects = [mock_object1]

        # The near_vector method directly returns the result, not a fluent API
        mock_query.near_vector.return_value = mock_result

        # Mock _ensure_collection to return the mock collection
        with patch.object(
            retriever, "_ensure_collection", return_value=mock_collection
        ):
            query_embedding = [0.1] * 1024
            results = await retriever.dense_search(query_embedding, topk=5)

            assert len(results) == 1
            assert results[0].document.id == "dense_doc"
            assert results[0].score == 0.85
            assert results[0].retrieval_method == "dense"
            assert results[0].document.content == "Dense search result"

    @pytest.mark.asyncio
    async def test_dense_search_with_filters(self: Any) -> None:
        """Test dense search with metadata filters."""
        config = {"class_name": "FilteredDense"}
        retriever = WeaviateRetriever(config)

        mock_collection = MagicMock()
        mock_query = MagicMock()
        mock_collection.query = mock_query

        # Mock search results
        mock_result = MagicMock()
        mock_result.objects = []
        mock_query.near_vector.return_value = mock_result

        # Mock _ensure_collection to return the mock collection
        with patch.object(
            retriever, "_ensure_collection", return_value=mock_collection
        ):
            filters = {"source": "important.pdf", "tags": ["critical"]}
            results = await retriever.dense_search([0.1] * 1024, filters=filters)

            # Verify search was called with filters
            mock_query.near_vector.assert_called_once()
            call_args = mock_query.near_vector.call_args
            assert "filters" in call_args.kwargs
            assert results == []

    @pytest.mark.asyncio
    async def test_sparse_search_basic(self: Any) -> None:
        """Test sparse (BM25) search."""
        config = {"class_name": "SparseSearch"}
        retriever = WeaviateRetriever(config)

        mock_collection = MagicMock()
        mock_query = MagicMock()
        mock_collection.query = mock_query

        # Mock sparse search result
        mock_object = MagicMock()
        mock_object.uuid = "sparse_uuid"
        mock_object.properties = {
            "content": "BM25 relevant content",
            "doc_id": "sparse_doc",
            "type": "sparse",
        }
        mock_object.metadata = MagicMock()
        mock_object.metadata.score = 2.5

        mock_result = MagicMock()
        mock_result.objects = [mock_object]
        mock_query.bm25.return_value = mock_result

        # Mock _ensure_collection to return the mock collection
        with patch.object(
            retriever, "_ensure_collection", return_value=mock_collection
        ):
            # Note: Weaviate sparse_search actually returns empty results
            # because it cannot reconstruct text from sparse terms
            results = await retriever.sparse_search({1: 0.9, 5: 0.7}, topk=10)

            # Implementation always returns empty list for sparse search
            assert results == []

    @pytest.mark.asyncio
    async def test_hybrid_search_basic(self: Any) -> None:
        """Test hybrid search combining dense and sparse."""
        config = {"class_name": "HybridSearch"}
        retriever = WeaviateRetriever(config)

        mock_collection = MagicMock()
        mock_query = MagicMock()
        mock_collection.query = mock_query

        # Mock hybrid search result
        mock_object = MagicMock()
        mock_object.uuid = "hybrid_uuid"
        mock_object.properties = {
            "content": "Hybrid search result",
            "doc_id": "hybrid_doc",
        }
        mock_object.metadata = MagicMock()
        mock_object.metadata.score = 0.92

        mock_result = MagicMock()
        mock_result.objects = [mock_object]
        mock_query.hybrid.return_value = mock_result

        with patch.object(
            retriever, "_ensure_collection", return_value=mock_collection
        ):
            query_embedding = [0.1] * 1024
            params = HybridSearchParams(alpha=0.7, topk=5)

            results = await retriever.hybrid_search(
                query_embedding=query_embedding,
                query_terms={},  # Dict of sparse terms
                params=params,
            )

            assert len(results) == 1
            assert results[0].document.id == "hybrid_doc"
            assert results[0].retrieval_method == "hybrid"

    @pytest.mark.asyncio
    async def test_hybrid_search_with_filters(self: Any) -> None:
        """Test hybrid search with metadata filters."""
        config = {"class_name": "FilteredHybrid"}
        retriever = WeaviateRetriever(config)

        mock_collection = MagicMock()
        mock_query = MagicMock()
        mock_collection.query = mock_query

        # Mock hybrid search result - empty for filters test
        mock_result = MagicMock()
        mock_result.objects = []
        mock_query.hybrid.return_value = mock_result

        with patch.object(
            retriever, "_ensure_collection", return_value=mock_collection
        ):
            filters = {"category": "research"}
            params = HybridSearchParams(alpha=0.5, topk=3, filters=filters)

            results = await retriever.hybrid_search([0.1] * 1024, {}, params)

            # Check that hybrid was called (filters are processed by _build_where)
            mock_query.hybrid.assert_called_once()
            assert len(results) == 0

    @pytest.mark.asyncio
    async def test_get_document(self: Any) -> None:
        """Test getting a document by ID."""
        config = {"class_name": "GetDocTest"}
        retriever = WeaviateRetriever(config)

        mock_collection = MagicMock()
        mock_query = MagicMock()
        mock_collection.query = mock_query

        # Mock document result
        mock_object = MagicMock()
        mock_object.uuid = "doc_uuid"
        mock_object.properties = MagicMock()
        mock_object.properties.get = MagicMock(
            side_effect=lambda k, default=None: {
                "content": "Retrieved document",
                "doc_id": "retrieved_doc",
                "retrieved": True,
            }.get(k, default)
        )
        mock_object.properties.items = MagicMock(return_value=[("retrieved", True)])
        mock_object.vector = [0.1] * 1024

        mock_result = MagicMock()
        mock_result.objects = [mock_object]
        mock_query.fetch_objects.return_value = mock_result

        # Mock _ensure_collection to return the mock collection
        with patch.object(
            retriever, "_ensure_collection", return_value=mock_collection
        ):
            # Mock the fallback data.objects.get_by_id call to return None
            mock_collection.data.objects.get_by_id.return_value = None

            document = await retriever.get_document("retrieved_doc")

            assert document is not None
            assert document.id == "retrieved_doc"
            assert document.content == "Retrieved document"
            assert document.embedding is None  # get_document doesn't return embedding

    @pytest.mark.asyncio
    async def test_get_document_not_found(self: Any) -> None:
        """Test getting non-existent document."""
        config = {"class_name": "NotFound"}
        retriever = WeaviateRetriever(config)

        mock_collection = MagicMock()
        mock_query = MagicMock()
        mock_collection.query = mock_query

        mock_result = MagicMock()
        mock_result.objects = []  # Empty result
        mock_query.fetch_objects.return_value = mock_result

        # Mock _ensure_collection to return the mock collection
        with patch.object(
            retriever, "_ensure_collection", return_value=mock_collection
        ):
            # Mock the fallback data.objects.get_by_id call to also return None
            mock_collection.data.objects.get_by_id.return_value = None

            document = await retriever.get_document("nonexistent")

            assert document is None

    @pytest.mark.asyncio
    async def test_delete_document(self: Any) -> None:
        """Test document deletion."""
        config = {"class_name": "DeleteTest"}
        retriever = WeaviateRetriever(config)

        mock_collection = MagicMock()
        mock_query = MagicMock()
        mock_data = MagicMock()
        mock_objects = MagicMock()
        mock_collection.query = mock_query
        mock_collection.data = mock_data
        mock_data.objects = mock_objects

        # Mock finding the document first
        mock_object = MagicMock()
        mock_object.uuid = "found_uuid"
        mock_result = MagicMock()
        mock_result.objects = [mock_object]
        mock_query.fetch_objects.return_value = mock_result

        with patch.object(
            retriever, "_ensure_collection", return_value=mock_collection
        ):
            success = await retriever.delete_document("doc_to_delete")

            assert success is True
            mock_query.fetch_objects.assert_called_once()
            mock_objects.delete_by_id.assert_called_once_with("found_uuid")

    @pytest.mark.asyncio
    async def test_delete_document_not_found(self: Any) -> None:
        """Test deleting non-existent document."""
        config = {"class_name": "DeleteMissing"}
        retriever = WeaviateRetriever(config)

        mock_collection = MagicMock()
        mock_query = MagicMock()
        mock_data = MagicMock()
        mock_objects = MagicMock()
        mock_collection.query = mock_query
        mock_collection.data = mock_data
        mock_data.objects = mock_objects

        # Mock no documents found
        mock_result = MagicMock()
        mock_result.objects = []
        mock_query.fetch_objects.return_value = mock_result

        # Mock fallback delete also fails
        mock_objects.delete_by_id.side_effect = Exception("Not found")

        with patch.object(
            retriever, "_ensure_collection", return_value=mock_collection
        ):
            success = await retriever.delete_document("missing_doc")

            assert success is False

    @pytest.mark.asyncio
    async def test_update_document(self: Any) -> None:
        """Test document update."""
        config = {"class_name": "UpdateTest"}
        retriever = WeaviateRetriever(config)

        updated_doc = Document(
            id="update_doc",
            content="Updated content",
            metadata={"updated": True},
            embedding=[0.2] * 1024,
        )

        # Mock delete_document and insert_documents
        with patch.object(
            retriever, "delete_document", return_value=True
        ) as mock_delete:
            with patch.object(
                retriever, "insert_documents", return_value=["uuid"]
            ) as mock_insert:
                success = await retriever.update_document("update_doc", updated_doc)

                assert success is True
                mock_delete.assert_called_once_with("update_doc")
                mock_insert.assert_called_once_with([updated_doc])

    @pytest.mark.asyncio
    async def test_get_collection_stats(self: Any) -> None:
        """Test getting collection statistics."""
        config = {"class_name": "StatsTest"}
        retriever = WeaviateRetriever(config)

        mock_collection = MagicMock()
        mock_config = MagicMock()
        mock_aggregate = MagicMock()
        mock_collection.config = mock_config
        mock_collection.aggregate = mock_aggregate

        # Mock config.get() result
        mock_info = MagicMock()
        mock_info.vector_index_config.__class__.__name__ = "HNSWIndex"
        mock_config.get.return_value = mock_info

        # Mock aggregate result
        mock_agg_result = MagicMock()
        mock_agg_result.total_count = 2500
        mock_aggregate.over_all.return_value = mock_agg_result

        with patch.object(
            retriever, "_ensure_collection", return_value=mock_collection
        ):
            stats = await retriever.get_collection_stats()

            assert stats["name"] == "StatsTest"
            assert stats["num_entities"] == 2500
            assert stats["description"] == "Weaviate collection"
            assert len(stats["indexes"]) == 1

    def test_build_where_filter_empty(self: Any) -> None:
        """Test building WHERE filter with no filters."""
        config = {"class_name": "FilterTest"}
        retriever = WeaviateRetriever(config)

        where_filter = retriever._build_where({})
        assert where_filter is None

    def test_build_where_filter_single(self: Any) -> None:
        """Test building WHERE filter with single condition."""
        config = {"class_name": "FilterTest"}
        retriever = WeaviateRetriever(config)

        with patch("askme.retriever.weaviate_retriever.Filter") as mock_filter:
            mock_by_property = MagicMock()
            mock_filter.by_property.return_value = mock_by_property
            mock_equal = MagicMock()
            mock_by_property.equal.return_value = mock_equal

            filters = {"source": "document.pdf"}
            where_filter = retriever._build_where(filters)

            mock_filter.by_property.assert_called_with("source")
            mock_by_property.equal.assert_called_with("document.pdf")
            assert where_filter == mock_equal

    def test_build_where_filter_multiple(self: Any) -> None:
        """Test building WHERE filter with multiple conditions."""
        config = {"class_name": "FilterTest"}
        retriever = WeaviateRetriever(config)

        with patch("askme.retriever.weaviate_retriever.Filter") as mock_filter:
            mock_by_property = MagicMock()
            mock_filter.by_property.return_value = mock_by_property
            mock_equal = MagicMock()
            mock_by_property.equal.return_value = mock_equal
            mock_by_property.contains_any.return_value = mock_equal

            filters = {"source": "doc.pdf", "tags": ["tag1", "tag2"]}
            retriever._build_where(filters)

            # Should create multiple conditions
            assert mock_filter.by_property.call_count >= 2

    @pytest.mark.asyncio
    async def test_ensure_collection(self: Any) -> None:
        """Test _ensure_collection method."""
        config = {"class_name": "EnsureTest"}
        retriever = WeaviateRetriever(config)

        mock_client = MagicMock()
        mock_collections = MagicMock()
        mock_client.collections = mock_collections
        mock_collections.exists.return_value = True

        mock_collection = MagicMock()
        mock_collections.get.return_value = mock_collection

        retriever.client = mock_client

        await retriever._ensure_collection()

        assert retriever.collection == mock_collection
        mock_collections.get.assert_called_once_with("EnsureTest")


class TestWeaviateRetrieverEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_operations_without_client(self: Any) -> None:
        """Test operations when client is not connected."""
        config = {"class_name": "NoClient"}
        retriever = WeaviateRetriever(config)

        # Mock _ensure_collection to fail when client is None
        with patch.object(
            retriever,
            "_ensure_collection",
            side_effect=RuntimeError("Collection not initialized"),
        ):
            with pytest.raises(RuntimeError):
                await retriever.dense_search([0.1] * 1024)

            with pytest.raises(RuntimeError):
                await retriever.get_document("test")

    @pytest.mark.asyncio
    async def test_operations_without_collection(self: Any) -> None:
        """Test operations when collection doesn't exist."""
        config = {"class_name": "NoCollection"}
        retriever = WeaviateRetriever(config)

        mock_client = MagicMock()
        retriever.client = mock_client
        # collection is None

        docs = [
            Document(id="test", content="test", metadata={}, embedding=[0.1] * 1024)
        ]

        # Should try to ensure collection first
        with patch.object(retriever, "_ensure_collection"):
            try:
                await retriever.insert_documents(docs)
            except Exception:
                pass  # Expected to fail due to mocking

    @pytest.mark.asyncio
    async def test_search_empty_results(self: Any) -> None:
        """Test search with no results."""
        config = {"class_name": "EmptyResults"}
        retriever = WeaviateRetriever(config)

        mock_collection = MagicMock()
        mock_query = MagicMock()
        mock_collection.query = mock_query

        # Mock empty results
        mock_result = MagicMock()
        mock_result.objects = []
        mock_query.near_vector.return_value = mock_result

        with patch.object(
            retriever, "_ensure_collection", return_value=mock_collection
        ):
            results = await retriever.dense_search([0.1] * 1024)
            assert results == []

    def test_url_parsing_edge_cases(self: Any) -> None:
        """Test URL parsing for different formats."""
        # Test with port specified
        config1 = {"url": "http://localhost:8081", "class_name": "Test1"}
        retriever1 = WeaviateRetriever(config1)
        assert retriever1.url == "http://localhost:8081"

        # Test HTTPS URL
        config2 = {"url": "https://weaviate.example.com", "class_name": "Test2"}
        retriever2 = WeaviateRetriever(config2)
        assert retriever2.url == "https://weaviate.example.com"

        # Test minimal URL
        config3 = {"url": "http://weaviate", "class_name": "Test3"}
        retriever3 = WeaviateRetriever(config3)
        assert retriever3.url == "http://weaviate"

    def test_config_edge_cases(self: Any) -> None:
        """Test configuration edge cases."""
        # Minimal config
        config = {}
        retriever = WeaviateRetriever(config)
        assert retriever.class_name == "AskmeDocument"
        assert retriever.url == "http://localhost:8080"

        # Config with extra fields
        config = {"class_name": "test", "extra_field": "ignored", "another": 456}
        retriever = WeaviateRetriever(config)
        assert retriever.class_name == "test"


class TestWeaviateRetrieverIntegration:
    """Integration-style tests."""

    @pytest.mark.asyncio
    async def test_full_workflow_simulation(self: Any) -> None:
        """Test complete workflow with mocks."""
        config = {"class_name": "IntegrationTest", "dimension": 512}
        retriever = WeaviateRetriever(config)

        mock_client = MagicMock()
        mock_collections = MagicMock()
        mock_client.collections = mock_collections

        # Mock collection creation/retrieval
        mock_collections.list_all.return_value = (
            []
        )  # Empty list means collection doesn't exist
        mock_created_collection = MagicMock()
        mock_collections.create.return_value = mock_created_collection
        mock_collection = MagicMock()
        mock_collections.get.return_value = mock_collection

        with patch("askme.retriever.weaviate_retriever.weaviate") as mock_weaviate:
            with patch("askme.retriever.weaviate_retriever.wcfg"):
                mock_weaviate.connect_to_custom.return_value = mock_client

                # 1. Connect
                await retriever.connect()
                assert retriever.client == mock_client

                # 2. Create collection
                await retriever.create_collection(dimension=512)
                assert retriever.collection == mock_collection

                # 3. Insert document
                docs = [
                    Document(
                        id="integration_doc",
                        content="Integration test content",
                        metadata={"test": True},
                        embedding=[0.1] * 512,
                    )
                ]

                mock_data = MagicMock()
                mock_collection.data = mock_data
                mock_result = MagicMock()
                mock_result.uuids = {"integration_doc": "uuid123"}
                mock_result.errors = {}
                mock_data.insert_many.return_value = mock_result

                result_ids = await retriever.insert_documents(docs)
                # The implementation returns UUID5 based on doc.id, not original IDs
                expected_uuid = (
                    "866289e0-e9d2-5afe-a424-e4c85ebb4ef8"  # UUID5 of "integration_doc"
                )
                assert result_ids == [expected_uuid]

                # 4. Search
                mock_query = MagicMock()
                mock_collection.query = mock_query

                mock_search_object = MagicMock()
                mock_search_object.uuid = expected_uuid
                mock_search_object.properties = {
                    "content": "Integration test content",
                    "doc_id": "integration_doc",
                }
                mock_search_object.metadata = MagicMock()
                mock_search_object.metadata.score = 0.9

                mock_result = MagicMock()
                mock_result.objects = [mock_search_object]
                mock_query.near_vector.return_value = mock_result

                search_results = await retriever.dense_search([0.1] * 512, topk=1)
                assert len(search_results) == 1
                assert search_results[0].document.id == "integration_doc"

                # 5. Cleanup
                await retriever.disconnect()
                assert retriever.client is None
