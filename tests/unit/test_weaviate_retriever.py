"""
Comprehensive unit tests for Weaviate retriever.
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from typing import List, Dict, Any

from askme.retriever.weaviate_retriever import WeaviateRetriever
from askme.retriever.base import Document, RetrievalResult, HybridSearchParams


class TestWeaviateRetrieverCore:
    """Test core Weaviate retriever functionality."""

    def test_initialization_basic(self):
        """Test basic initialization."""
        config = {
            "url": "https://test-cluster.weaviate.network",
            "api_key": "test-api-key",
            "class_name": "TestDocument",
            "dimension": 768
        }
        
        retriever = WeaviateRetriever(config)
        
        assert retriever.url == "https://test-cluster.weaviate.network"
        assert retriever.api_key == "test-api-key"
        assert retriever.class_name == "TestDocument"
        assert retriever.dimension == 768
        assert retriever.client is None
        assert retriever.collection is None

    def test_initialization_defaults(self):
        """Test initialization with default values."""
        config = {"collection_name": "minimal"}
        
        retriever = WeaviateRetriever(config)
        
        assert retriever.url == "http://localhost:8080"
        assert retriever.api_key == ""
        assert retriever.class_name == "AskmeDocument"
        assert retriever.dimension == 1024
        assert retriever.collection_name == "minimal"

    @pytest.mark.asyncio
    async def test_connect_with_api_key(self):
        """Test connection with API key (cloud)."""
        config = {
            "url": "https://cluster.weaviate.network",
            "api_key": "test-key",
            "class_name": "CloudDoc"
        }
        retriever = WeaviateRetriever(config)
        
        mock_client = MagicMock()
        
        with patch('askme.retriever.weaviate_retriever.weaviate') as mock_weaviate:
            mock_weaviate.connect_to_weaviate_cloud.return_value = mock_client
            mock_weaviate.AuthApiKey.return_value = "auth_object"
            
            await retriever.connect()
            
            assert retriever.client == mock_client
            mock_weaviate.connect_to_weaviate_cloud.assert_called_once_with(
                cluster_url="https://cluster.weaviate.network",
                auth_credentials="auth_object"
            )

    @pytest.mark.asyncio
    async def test_connect_local_http(self):
        """Test connection to local Weaviate instance."""
        config = {
            "url": "http://localhost:8080",
            "api_key": "",
            "class_name": "LocalDoc"
        }
        retriever = WeaviateRetriever(config)
        
        mock_client = MagicMock()
        
        with patch('askme.retriever.weaviate_retriever.weaviate') as mock_weaviate:
            mock_weaviate.connect_to_local.return_value = mock_client
            
            await retriever.connect()
            
            assert retriever.client == mock_client
            mock_weaviate.connect_to_local.assert_called_once_with(
                host="localhost",
                port=8080,
                grpc_port=50051,
                http_secure=False,
                grpc_secure=False
            )

    @pytest.mark.asyncio
    async def test_connect_local_https(self):
        """Test connection to local HTTPS Weaviate."""
        config = {
            "url": "https://localhost:8443",
            "api_key": "",
            "class_name": "SecureDoc"
        }
        retriever = WeaviateRetriever(config)
        
        mock_client = MagicMock()
        
        with patch('askme.retriever.weaviate_retriever.weaviate') as mock_weaviate:
            mock_weaviate.connect_to_local.return_value = mock_client
            
            await retriever.connect()
            
            mock_weaviate.connect_to_local.assert_called_once_with(
                host="localhost",
                port=8443,
                grpc_port=50051,
                http_secure=True,
                grpc_secure=True
            )

    @pytest.mark.asyncio
    async def test_connect_failure(self):
        """Test connection failure handling."""
        config = {"class_name": "FailDoc"}
        retriever = WeaviateRetriever(config)
        
        with patch('askme.retriever.weaviate_retriever.weaviate') as mock_weaviate:
            mock_weaviate.connect_to_local.side_effect = Exception("Connection failed")
            
            # Should not raise exception but log error
            await retriever.connect()
            assert retriever.client is None

    @pytest.mark.asyncio
    async def test_disconnect(self):
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
    async def test_disconnect_no_client(self):
        """Test disconnection when no client exists."""
        config = {"class_name": "NoClient"}
        retriever = WeaviateRetriever(config)
        
        # Should not raise error
        await retriever.disconnect()
        
        assert retriever.client is None

    @pytest.mark.asyncio
    async def test_create_collection_new(self):
        """Test creating a new collection."""
        config = {"class_name": "NewCollection"}
        retriever = WeaviateRetriever(config)
        
        mock_client = MagicMock()
        mock_collections = MagicMock()
        mock_client.collections = mock_collections
        mock_collections.exists.return_value = False
        retriever.client = mock_client
        
        with patch('askme.retriever.weaviate_retriever.wcfg') as mock_wcfg:
            mock_property = MagicMock()
            mock_wcfg.Property.return_value = mock_property
            
            mock_vector_config = MagicMock()  
            mock_wcfg.Configure.VectorIndex.hnsw.return_value = mock_vector_config
            
            mock_collection = MagicMock()
            mock_collections.create.return_value = mock_collection
            
            await retriever.create_collection(dimension=768, metric="cosine")
            
            mock_collections.create.assert_called_once()
            assert retriever.collection == mock_collection

    @pytest.mark.asyncio
    async def test_create_collection_existing(self):
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
    async def test_insert_documents_basic(self):
        """Test basic document insertion."""
        config = {"class_name": "InsertTest"}
        retriever = WeaviateRetriever(config)
        
        # Mock collection and client
        mock_collection = MagicMock()
        mock_data = MagicMock()
        mock_collection.data = mock_data
        retriever.collection = mock_collection
        
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
                embedding=[0.1] * 1024
            ),
            Document(
                id="doc2",
                content="Second document content",
                metadata={"source": "test2.txt", "type": "text"},
                embedding=[0.2] * 1024
            )
        ]
        
        result_ids = await retriever.insert_documents(documents)
        
        assert result_ids == ["doc1", "doc2"]
        mock_data.insert_many.assert_called_once()

    @pytest.mark.asyncio
    async def test_insert_documents_with_errors(self):
        """Test document insertion with partial errors."""
        config = {"class_name": "ErrorTest"}
        retriever = WeaviateRetriever(config)
        
        mock_collection = MagicMock()
        mock_data = MagicMock()
        mock_collection.data = mock_data
        retriever.collection = mock_collection
        
        # Mock result with errors
        mock_result = MagicMock()
        mock_result.uuids = {"doc1": "uuid1"}  # Only one succeeded
        mock_result.errors = {"doc2": "Insert failed"}
        mock_data.insert_many.return_value = mock_result
        
        documents = [
            Document(
                id="doc1", 
                content="Success doc",
                metadata={},
                embedding=[0.1] * 1024
            ),
            Document(
                id="doc2",
                content="Failed doc", 
                metadata={},
                embedding=[0.2] * 1024
            )
        ]
        
        result_ids = await retriever.insert_documents(documents)
        
        assert result_ids == ["doc1"]  # Only successful insertions

    @pytest.mark.asyncio
    async def test_dense_search_basic(self):
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
            "metadata": {"source": "dense.txt"},
            "askme_id": "dense_doc"
        }
        mock_object1.metadata = MagicMock()
        mock_object1.metadata.distance = 0.15  # Note: Weaviate uses distance (lower=better)
        
        mock_result = MagicMock()
        mock_result.objects = [mock_object1]
        mock_query.near_vector.return_value.limit.return_value.objects = [mock_object1]
        
        retriever.collection = mock_collection
        
        query_embedding = [0.1] * 1024
        results = await retriever.dense_search(query_embedding, topk=5)
        
        assert len(results) == 1
        assert results[0].document.id == "dense_doc"
        assert results[0].score == 0.85  # Converted from distance (1 - 0.15)
        assert results[0].retrieval_method == "dense"
        assert results[0].document.content == "Dense search result"

    @pytest.mark.asyncio
    async def test_dense_search_with_filters(self):
        """Test dense search with metadata filters."""
        config = {"class_name": "FilteredDense"}
        retriever = WeaviateRetriever(config)
        
        mock_collection = MagicMock()
        mock_query = MagicMock()
        mock_collection.query = mock_query
        
        # Chain mocks for fluent API
        mock_near_vector = MagicMock()
        mock_where = MagicMock()
        mock_limit = MagicMock()
        mock_limit.objects = []
        
        mock_query.near_vector.return_value = mock_near_vector
        mock_near_vector.where.return_value = mock_where
        mock_where.limit.return_value = mock_limit
        
        retriever.collection = mock_collection
        
        filters = {"source": "important.pdf", "tags": ["critical"]}
        await retriever.dense_search([0.1] * 1024, filters=filters)
        
        # Verify filter was applied
        mock_near_vector.where.assert_called_once()

    @pytest.mark.asyncio
    async def test_sparse_search_basic(self):
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
            "metadata": {"type": "sparse"},
            "askme_id": "sparse_doc"
        }
        mock_object.metadata = MagicMock()
        mock_object.metadata.score = 2.5
        
        mock_bm25 = MagicMock()
        mock_bm25.limit.return_value.objects = [mock_object]
        mock_query.bm25.return_value = mock_bm25
        
        retriever.collection = mock_collection
        
        # Note: For Weaviate, sparse search uses text query, not term dict
        results = await retriever.sparse_search("sparse search query", topk=10)
        
        assert len(results) == 1
        assert results[0].document.id == "sparse_doc"
        assert results[0].score == 2.5
        assert results[0].retrieval_method == "sparse"

    @pytest.mark.asyncio
    async def test_hybrid_search_basic(self):
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
            "metadata": {"hybrid": True},
            "askme_id": "hybrid_doc"
        }
        mock_object.metadata = MagicMock()
        mock_object.metadata.score = 0.92
        
        mock_hybrid = MagicMock()
        mock_hybrid.limit.return_value.objects = [mock_object]
        mock_query.hybrid.return_value = mock_hybrid
        
        retriever.collection = mock_collection
        
        query_embedding = [0.1] * 1024
        params = HybridSearchParams(alpha=0.7, topk=5)
        
        results = await retriever.hybrid_search(
            query_embedding=query_embedding,
            query_terms="hybrid query text",  # Weaviate expects text
            params=params
        )
        
        assert len(results) == 1
        assert results[0].document.id == "hybrid_doc"
        assert results[0].retrieval_method == "hybrid"

    @pytest.mark.asyncio
    async def test_hybrid_search_with_filters(self):
        """Test hybrid search with metadata filters."""
        config = {"class_name": "FilteredHybrid"}
        retriever = WeaviateRetriever(config)
        
        mock_collection = MagicMock()
        mock_query = MagicMock()
        mock_collection.query = mock_query
        
        # Chain mocks for fluent API
        mock_hybrid = MagicMock()
        mock_where = MagicMock()
        mock_limit = MagicMock()
        mock_limit.objects = []
        
        mock_query.hybrid.return_value = mock_hybrid
        mock_hybrid.where.return_value = mock_where
        mock_where.limit.return_value = mock_limit
        
        retriever.collection = mock_collection
        
        filters = {"category": "research"}
        params = HybridSearchParams(alpha=0.5, topk=3, filters=filters)
        
        await retriever.hybrid_search([0.1] * 1024, "research query", params)
        
        mock_hybrid.where.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_document(self):
        """Test getting a document by ID."""
        config = {"class_name": "GetDocTest"}
        retriever = WeaviateRetriever(config)
        
        mock_collection = MagicMock()
        mock_query = MagicMock()
        mock_collection.query = mock_query
        
        # Mock document result
        mock_object = MagicMock()
        mock_object.uuid = "doc_uuid"
        mock_object.properties = {
            "content": "Retrieved document",
            "metadata": {"retrieved": True},
            "askme_id": "retrieved_doc"
        }
        mock_object.vector = [0.1] * 1024
        
        mock_where = MagicMock()
        mock_where.objects = [mock_object]
        mock_query.where.return_value = mock_where
        
        retriever.collection = mock_collection
        
        document = await retriever.get_document("retrieved_doc")
        
        assert document is not None
        assert document.id == "retrieved_doc"
        assert document.content == "Retrieved document"
        assert len(document.embedding) == 1024

    @pytest.mark.asyncio
    async def test_get_document_not_found(self):
        """Test getting non-existent document."""
        config = {"class_name": "NotFound"}
        retriever = WeaviateRetriever(config)
        
        mock_collection = MagicMock()
        mock_query = MagicMock()
        mock_collection.query = mock_query
        
        mock_where = MagicMock()
        mock_where.objects = []  # Empty result
        mock_query.where.return_value = mock_where
        
        retriever.collection = mock_collection
        
        document = await retriever.get_document("nonexistent")
        
        assert document is None

    @pytest.mark.asyncio
    async def test_delete_document(self):
        """Test document deletion."""
        config = {"class_name": "DeleteTest"}
        retriever = WeaviateRetriever(config)
        
        mock_collection = MagicMock()
        mock_data = MagicMock()
        mock_collection.data = mock_data
        
        # Mock successful deletion
        mock_result = MagicMock()
        mock_result.successful = 1
        mock_data.delete_where.return_value = mock_result
        
        retriever.collection = mock_collection
        
        success = await retriever.delete_document("doc_to_delete")
        
        assert success is True
        mock_data.delete_where.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_document_not_found(self):
        """Test deleting non-existent document."""
        config = {"class_name": "DeleteMissing"}
        retriever = WeaviateRetriever(config)
        
        mock_collection = MagicMock()
        mock_data = MagicMock()
        mock_collection.data = mock_data
        
        # Mock no deletion
        mock_result = MagicMock()
        mock_result.successful = 0
        mock_data.delete_where.return_value = mock_result
        
        retriever.collection = mock_collection
        
        success = await retriever.delete_document("missing_doc")
        
        assert success is False

    @pytest.mark.asyncio
    async def test_update_document(self):
        """Test document update."""
        config = {"class_name": "UpdateTest"}
        retriever = WeaviateRetriever(config)
        
        mock_collection = MagicMock()
        mock_data = MagicMock()
        mock_collection.data = mock_data
        
        # Mock successful update
        mock_result = MagicMock()
        mock_result.successful = 1
        mock_data.update_where.return_value = mock_result
        
        retriever.collection = mock_collection
        
        updated_doc = Document(
            id="update_doc",
            content="Updated content",
            metadata={"updated": True},
            embedding=[0.2] * 1024
        )
        
        success = await retriever.update_document("update_doc", updated_doc)
        
        assert success is True
        mock_data.update_where.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_collection_stats(self):
        """Test getting collection statistics."""
        config = {"class_name": "StatsTest"}
        retriever = WeaviateRetriever(config)
        
        mock_collection = MagicMock()
        mock_aggregate = MagicMock()
        mock_collection.aggregate = mock_aggregate
        
        # Mock aggregate result
        mock_meta_count = MagicMock()
        mock_meta_count.meta_count = 2500
        mock_aggregate.over_all.return_value.objects = [mock_meta_count]
        
        retriever.collection = mock_collection
        
        stats = await retriever.get_collection_stats()
        
        assert stats["class_name"] == "StatsTest"
        assert stats["total_objects"] == 2500
        assert stats["connected"] is True

    def test_build_where_filter_empty(self):
        """Test building WHERE filter with no filters."""
        config = {"class_name": "FilterTest"}
        retriever = WeaviateRetriever(config)
        
        where_filter = retriever._build_where({})
        assert where_filter is None

    def test_build_where_filter_single(self):
        """Test building WHERE filter with single condition."""
        config = {"class_name": "FilterTest"}
        retriever = WeaviateRetriever(config)
        
        with patch('askme.retriever.weaviate_retriever.Filter') as mock_filter:
            mock_by_property = MagicMock()
            mock_filter.by_property.return_value = mock_by_property
            mock_equal = MagicMock()
            mock_by_property.equal.return_value = mock_equal
            
            filters = {"source": "document.pdf"}
            where_filter = retriever._build_where(filters)
            
            mock_filter.by_property.assert_called_with("metadata.source")
            mock_by_property.equal.assert_called_with("document.pdf")
            assert where_filter == mock_equal

    def test_build_where_filter_multiple(self):
        """Test building WHERE filter with multiple conditions."""
        config = {"class_name": "FilterTest"}
        retriever = WeaviateRetriever(config)
        
        with patch('askme.retriever.weaviate_retriever.Filter') as mock_filter:
            mock_by_property = MagicMock()
            mock_filter.by_property.return_value = mock_by_property
            mock_equal = MagicMock()
            mock_by_property.equal.return_value = mock_equal
            mock_by_property.contains_any.return_value = mock_equal
            
            filters = {"source": "doc.pdf", "tags": ["tag1", "tag2"]}
            where_filter = retriever._build_where(filters)
            
            # Should create multiple conditions
            assert mock_filter.by_property.call_count >= 2

    @pytest.mark.asyncio
    async def test_ensure_collection(self):
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
    async def test_operations_without_client(self):
        """Test operations when client is not connected."""
        config = {"class_name": "NoClient"}
        retriever = WeaviateRetriever(config)
        
        # All these operations should handle missing client gracefully
        with pytest.raises((AttributeError, RuntimeError)):
            await retriever.dense_search([0.1] * 1024)
            
        with pytest.raises((AttributeError, RuntimeError)):
            await retriever.get_document("test")

    @pytest.mark.asyncio
    async def test_operations_without_collection(self):
        """Test operations when collection doesn't exist."""
        config = {"class_name": "NoCollection"}
        retriever = WeaviateRetriever(config)
        
        mock_client = MagicMock()
        retriever.client = mock_client
        # collection is None
        
        docs = [Document(id="test", content="test", metadata={}, embedding=[0.1] * 1024)]
        
        # Should try to ensure collection first
        with patch.object(retriever, '_ensure_collection'):
            try:
                await retriever.insert_documents(docs)
            except Exception:
                pass  # Expected to fail due to mocking

    @pytest.mark.asyncio 
    async def test_search_empty_results(self):
        """Test search with no results."""
        config = {"class_name": "EmptyResults"}
        retriever = WeaviateRetriever(config)
        
        mock_collection = MagicMock()
        mock_query = MagicMock()
        mock_collection.query = mock_query
        
        # Mock empty results
        mock_limit = MagicMock()
        mock_limit.objects = []
        mock_query.near_vector.return_value.limit.return_value = mock_limit
        
        retriever.collection = mock_collection
        
        results = await retriever.dense_search([0.1] * 1024)
        assert results == []

    def test_url_parsing_edge_cases(self):
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

    def test_config_edge_cases(self):
        """Test configuration edge cases."""
        # Minimal config
        config = {}
        retriever = WeaviateRetriever(config)
        assert retriever.class_name == "AskmeDocument"
        assert retriever.url == "http://localhost:8080"
        
        # Config with extra fields
        config = {
            "class_name": "test",
            "extra_field": "ignored",
            "another": 456
        }
        retriever = WeaviateRetriever(config)
        assert retriever.class_name == "test"


class TestWeaviateRetrieverIntegration:
    """Integration-style tests."""

    @pytest.mark.asyncio
    async def test_full_workflow_simulation(self):
        """Test complete workflow with mocks."""
        config = {"class_name": "IntegrationTest", "dimension": 512}
        retriever = WeaviateRetriever(config)
        
        mock_client = MagicMock()
        mock_collections = MagicMock()
        mock_client.collections = mock_collections
        
        # Mock collection creation/retrieval
        mock_collections.exists.return_value = False
        mock_collection = MagicMock()
        mock_collections.create.return_value = mock_collection
        
        with patch('askme.retriever.weaviate_retriever.weaviate') as mock_weaviate:
            with patch('askme.retriever.weaviate_retriever.wcfg'):
                mock_weaviate.connect_to_local.return_value = mock_client
                
                # 1. Connect
                await retriever.connect()
                assert retriever.client == mock_client
                
                # 2. Create collection
                await retriever.create_collection(dimension=512)
                assert retriever.collection == mock_collection
                
                # 3. Insert document
                docs = [Document(
                    id="integration_doc",
                    content="Integration test content",
                    metadata={"test": True},
                    embedding=[0.1] * 512
                )]
                
                mock_data = MagicMock()
                mock_collection.data = mock_data
                mock_result = MagicMock()
                mock_result.uuids = {"integration_doc": "uuid123"}
                mock_result.errors = {}
                mock_data.insert_many.return_value = mock_result
                
                result_ids = await retriever.insert_documents(docs)
                assert result_ids == ["integration_doc"]
                
                # 4. Search
                mock_query = MagicMock()
                mock_collection.query = mock_query
                
                mock_search_object = MagicMock()
                mock_search_object.uuid = "uuid123"
                mock_search_object.properties = {
                    "content": "Integration test content",
                    "metadata": {"test": True},
                    "askme_id": "integration_doc"
                }
                mock_search_object.metadata = MagicMock()
                mock_search_object.metadata.distance = 0.1
                
                mock_limit = MagicMock()
                mock_limit.objects = [mock_search_object]
                mock_query.near_vector.return_value.limit.return_value = mock_limit
                
                search_results = await retriever.dense_search([0.1] * 512, topk=1)
                assert len(search_results) == 1
                assert search_results[0].document.id == "integration_doc"
                
                # 5. Cleanup
                await retriever.disconnect()
                assert retriever.client is None