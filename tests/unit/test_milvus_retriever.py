"""
Focused unit tests for Milvus retriever covering actual implementation.
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from typing import List, Dict, Any

from askme.retriever.milvus_retriever import MilvusRetriever
from askme.retriever.base import Document, RetrievalResult, HybridSearchParams


class TestMilvusRetrieverCore:
    """Test core Milvus retriever functionality."""

    def test_initialization_basic(self):
        """Test basic initialization."""
        config = {
            "collection_name": "test_collection",
            "host": "test-host",
            "port": 19532,
            "username": "testuser",
            "password": "testpass",
            "secure": True,
            "dimension": 512
        }
        
        retriever = MilvusRetriever(config)
        
        assert retriever.collection_name == "test_collection"
        assert retriever.host == "test-host"
        assert retriever.port == 19532
        assert retriever.username == "testuser"
        assert retriever.password == "testpass"
        assert retriever.secure is True
        assert retriever.dimension == 512

    def test_initialization_defaults(self):
        """Test initialization with default values."""
        config = {"collection_name": "defaults_test"}
        
        retriever = MilvusRetriever(config)
        
        assert retriever.collection_name == "defaults_test"
        assert retriever.host == "localhost"
        assert retriever.port == 19530
        assert retriever.username == ""
        assert retriever.password == ""
        assert retriever.secure is False
        assert retriever.dimension == 1024
        assert retriever.collection is None

    def test_connection_name_generation(self):
        """Test connection name generation."""
        config1 = {"collection_name": "test1", "host": "host1", "port": 19530}
        config2 = {"collection_name": "test2", "host": "host2", "port": 19531}
        
        retriever1 = MilvusRetriever(config1)
        retriever2 = MilvusRetriever(config2)
        
        # Different configs should have different connection names
        assert retriever1.connection_name != retriever2.connection_name
        assert "askme_conn_" in retriever1.connection_name
        assert "askme_conn_" in retriever2.connection_name

    @pytest.mark.asyncio
    async def test_connect_success(self):
        """Test successful connection."""
        config = {"collection_name": "connect_test"}
        retriever = MilvusRetriever(config)
        
        with patch('askme.retriever.milvus_retriever.connections') as mock_connections:
            with patch('askme.retriever.milvus_retriever.utility') as mock_utility:
                with patch('askme.retriever.milvus_retriever.Collection') as mock_collection_class:
                    
                    mock_utility.has_collection.return_value = True
                    mock_collection = MagicMock()
                    mock_collection_class.return_value = mock_collection
                    
                    await retriever.connect()
                    
                    # Verify connection was attempted
                    mock_connections.connect.assert_called_once_with(
                        alias=retriever.connection_name,
                        host=retriever.host,
                        port=retriever.port,
                        user=retriever.username,
                        password=retriever.password,
                        secure=retriever.secure,
                    )
                    
                    # Verify collection was loaded
                    assert retriever.collection is not None
                    mock_collection.load.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_no_existing_collection(self):
        """Test connection when collection doesn't exist."""
        config = {"collection_name": "nonexistent"}
        retriever = MilvusRetriever(config)
        
        with patch('askme.retriever.milvus_retriever.connections'):
            with patch('askme.retriever.milvus_retriever.utility') as mock_utility:
                
                mock_utility.has_collection.return_value = False
                
                await retriever.connect()
                
                # Collection should remain None
                assert retriever.collection is None

    @pytest.mark.asyncio
    async def test_disconnect(self):
        """Test disconnection."""
        config = {"collection_name": "disconnect_test"}
        retriever = MilvusRetriever(config)
        
        with patch('askme.retriever.milvus_retriever.connections') as mock_connections:
            await retriever.disconnect()
            
            mock_connections.disconnect.assert_called_once_with(
                alias=retriever.connection_name
            )

    @pytest.mark.asyncio
    async def test_create_collection_basic(self):
        """Test basic collection creation."""
        config = {"collection_name": "create_test"}
        retriever = MilvusRetriever(config)
        
        with patch('askme.retriever.milvus_retriever.Collection') as mock_collection_class:
            with patch('askme.retriever.milvus_retriever.CollectionSchema'):
                with patch('askme.retriever.milvus_retriever.FieldSchema'):
                    
                    mock_collection = MagicMock()
                    mock_collection_class.return_value = mock_collection
                    
                    await retriever.create_collection(dimension=768, metric="cosine")
                    
                    # Verify collection was created and indexes set up
                    mock_collection_class.assert_called_once()
                    mock_collection.create_index.assert_called()
                    mock_collection.load.assert_called_once()

    @pytest.mark.asyncio
    async def test_insert_documents_basic(self):
        """Test basic document insertion."""
        config = {"collection_name": "insert_test"}
        retriever = MilvusRetriever(config)
        
        # Mock collection
        mock_collection = MagicMock()
        retriever.collection = mock_collection
        
        # Mock successful insertion
        mock_result = MagicMock()
        mock_result.primary_keys = ["doc1", "doc2"]
        mock_collection.insert.return_value = mock_result
        
        documents = [
            Document(
                id="doc1",
                content="First document",
                metadata={"source": "test1.txt"},
                embedding=[0.1] * 1024,
                sparse_embedding={1: 0.8, 5: 0.6}
            ),
            Document(
                id="doc2",
                content="Second document", 
                metadata={"source": "test2.txt"},
                embedding=[0.2] * 1024,
                sparse_embedding={2: 0.9, 3: 0.4}
            )
        ]
        
        result_ids = await retriever.insert_documents(documents)
        
        assert result_ids == ["doc1", "doc2"]
        mock_collection.insert.assert_called_once()
        mock_collection.flush.assert_called_once()

    @pytest.mark.asyncio
    async def test_insert_documents_no_collection(self):
        """Test insertion failure without collection."""
        config = {"collection_name": "no_collection"}
        retriever = MilvusRetriever(config)
        
        docs = [Document(
            id="test", 
            content="test",
            metadata={},
            embedding=[0.1] * 1024,
            sparse_embedding={1: 0.8}
        )]
        
        with pytest.raises(RuntimeError, match="Collection not initialized"):
            await retriever.insert_documents(docs)

    @pytest.mark.asyncio
    async def test_dense_search_basic(self):
        """Test basic dense search."""
        config = {"collection_name": "dense_search_test"}
        retriever = MilvusRetriever(config)
        
        # Mock collection
        mock_collection = MagicMock()
        retriever.collection = mock_collection
        
        # Mock search results
        mock_hit = MagicMock()
        mock_hit.id = "doc1"
        mock_hit.distance = 0.85
        mock_hit.entity = {
            "content": "Search result content",
            "metadata": '{"source": "result.txt"}',
            "id": "doc1"
        }
        
        mock_collection.search.return_value = [[mock_hit]]
        
        query_embedding = [0.1] * 1024
        results = await retriever.dense_search(query_embedding, topk=5)
        
        assert len(results) == 1
        assert results[0].document.id == "doc1"
        assert results[0].score == 0.85
        assert results[0].retrieval_method == "dense"
        assert results[0].document.content == "Search result content"

    @pytest.mark.asyncio
    async def test_dense_search_with_filters(self):
        """Test dense search with metadata filters."""
        config = {"collection_name": "filtered_search"}
        retriever = MilvusRetriever(config)
        
        mock_collection = MagicMock()
        retriever.collection = mock_collection
        mock_collection.search.return_value = [[]]
        
        filters = {"tags": ["important"], "source": "document.pdf"}
        await retriever.dense_search([0.1] * 1024, filters=filters)
        
        # Verify search was called with expression
        search_call = mock_collection.search.call_args
        assert "expr" in search_call[1]

    @pytest.mark.asyncio
    async def test_sparse_search_basic(self):
        """Test basic sparse search."""
        config = {"collection_name": "sparse_test"}
        retriever = MilvusRetriever(config)
        
        mock_collection = MagicMock()
        retriever.collection = mock_collection
        
        # Mock search results
        mock_hit = MagicMock()
        mock_hit.id = "doc2" 
        mock_hit.distance = 3.2
        mock_hit.entity = {
            "content": "Sparse search content",
            "metadata": '{"type": "sparse"}',
            "id": "doc2"
        }
        
        mock_collection.search.return_value = [[mock_hit]]
        
        query_terms = {1: 0.9, 5: 0.7, 10: 0.5}
        results = await retriever.sparse_search(query_terms, topk=10)
        
        assert len(results) == 1
        assert results[0].document.id == "doc2"
        assert results[0].score == 3.2
        assert results[0].retrieval_method == "sparse"

    @pytest.mark.asyncio
    async def test_build_filter_expression(self):
        """Test filter expression building."""
        config = {"collection_name": "filter_test"}
        retriever = MilvusRetriever(config)
        
        # Test empty filters
        expr = retriever._build_filter_expression({})
        assert expr == ""
        
        # Test single filter
        filters = {"source": "document.pdf"}
        expr = retriever._build_filter_expression(filters)
        assert "source" in expr
        assert "document.pdf" in expr
        
        # Test multiple filters
        filters = {
            "tags": ["important", "urgent"],
            "source": "test.txt",
            "created_after": "2023-01-01"
        }
        expr = retriever._build_filter_expression(filters)
        assert "tags" in expr
        assert "source" in expr
        assert "created_after" in expr

    @pytest.mark.asyncio
    async def test_get_collection_stats(self):
        """Test getting collection statistics."""
        config = {"collection_name": "stats_test"}
        retriever = MilvusRetriever(config)
        
        mock_collection = MagicMock()
        retriever.collection = mock_collection
        
        # Mock collection num_entities property
        mock_collection.num_entities = 1500
        
        with patch('askme.retriever.milvus_retriever.utility') as mock_utility:
            mock_utility.get_query_segment_info.return_value = [
                MagicMock(num_rows=800),
                MagicMock(num_rows=700)
            ]
            
            stats = await retriever.get_collection_stats()
            
            assert "total_entities" in stats
            assert stats["total_entities"] == 1500
            assert "collection_name" in stats
            assert stats["collection_name"] == "stats_test"

    @pytest.mark.asyncio
    async def test_get_document(self):
        """Test getting a single document by ID."""
        config = {"collection_name": "get_doc_test"}
        retriever = MilvusRetriever(config)
        
        mock_collection = MagicMock()
        retriever.collection = mock_collection
        
        # Mock query result
        mock_collection.query.return_value = [{
            "id": "test_doc",
            "content": "Retrieved document content",
            "metadata": '{"retrieved": true}',
            "dense_vector": [0.1] * 1024,
            "sparse_vector": {"indices": [1, 5], "values": [0.8, 0.6]}
        }]
        
        document = await retriever.get_document("test_doc")
        
        assert document is not None
        assert document.id == "test_doc"
        assert document.content == "Retrieved document content"
        assert len(document.embedding) == 1024

    @pytest.mark.asyncio
    async def test_get_document_not_found(self):
        """Test getting document that doesn't exist."""
        config = {"collection_name": "not_found_test"}
        retriever = MilvusRetriever(config)
        
        mock_collection = MagicMock()
        retriever.collection = mock_collection
        mock_collection.query.return_value = []
        
        document = await retriever.get_document("nonexistent")
        
        assert document is None

    @pytest.mark.asyncio
    async def test_delete_document(self):
        """Test document deletion."""
        config = {"collection_name": "delete_test"}
        retriever = MilvusRetriever(config)
        
        mock_collection = MagicMock()
        retriever.collection = mock_collection
        
        # Mock successful deletion
        mock_result = MagicMock()
        mock_result.delete_count = 1
        mock_collection.delete.return_value = mock_result
        
        success = await retriever.delete_document("doc_to_delete")
        
        assert success is True
        mock_collection.delete.assert_called_once()
        mock_collection.flush.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_document_not_found(self):
        """Test deleting non-existent document."""
        config = {"collection_name": "delete_missing"}
        retriever = MilvusRetriever(config)
        
        mock_collection = MagicMock()
        retriever.collection = mock_collection
        
        # Mock no deletion (document not found)
        mock_result = MagicMock()
        mock_result.delete_count = 0
        mock_collection.delete.return_value = mock_result
        
        success = await retriever.delete_document("missing_doc")
        
        assert success is False


class TestMilvusRetrieverHybridSearch:
    """Test hybrid search functionality."""

    @pytest.mark.asyncio
    async def test_hybrid_search_rrf(self):
        """Test hybrid search with RRF fusion."""
        config = {"collection_name": "hybrid_rrf"}
        retriever = MilvusRetriever(config)
        
        mock_collection = MagicMock()
        retriever.collection = mock_collection
        
        # Mock that RRF is available
        with patch('askme.retriever.milvus_retriever.HYBRID_SEARCH_AVAILABLE', True):
            with patch('askme.retriever.milvus_retriever.AnnSearchRequest') as mock_ann_req:
                with patch('askme.retriever.milvus_retriever.RRFRanker') as mock_rrf:
                    
                    # Mock search result
                    mock_hit = MagicMock()
                    mock_hit.id = "hybrid_doc"
                    mock_hit.distance = 0.9
                    mock_hit.entity = {
                        "content": "Hybrid result",
                        "metadata": '{"hybrid": true}',
                        "id": "hybrid_doc"
                    }
                    
                    mock_collection.hybrid_search.return_value = [[mock_hit]]
                    
                    query_embedding = [0.1] * 1024
                    query_terms = {1: 0.8, 5: 0.6}
                    params = HybridSearchParams(use_rrf=True, rrf_k=60, topk=5)
                    
                    results = await retriever.hybrid_search(
                        query_embedding=query_embedding,
                        query_terms=query_terms,
                        params=params
                    )
                    
                    assert len(results) == 1
                    assert results[0].document.id == "hybrid_doc"
                    assert results[0].retrieval_method == "hybrid"

    @pytest.mark.asyncio
    async def test_hybrid_search_alpha_fallback(self):
        """Test hybrid search with alpha fusion fallback."""
        config = {"collection_name": "hybrid_alpha"}
        retriever = MilvusRetriever(config)
        
        # Mock dense and sparse search methods
        with patch.object(retriever, 'dense_search') as mock_dense:
            with patch.object(retriever, 'sparse_search') as mock_sparse:
                
                # Setup mock results
                dense_results = [
                    RetrievalResult(
                        document=Document(id="doc1", content="Dense", metadata={}),
                        score=0.9,
                        rank=1,
                        retrieval_method="dense"
                    )
                ]
                sparse_results = [
                    RetrievalResult(
                        document=Document(id="doc2", content="Sparse", metadata={}),
                        score=2.5,
                        rank=1,
                        retrieval_method="sparse"
                    )
                ]
                
                mock_dense.return_value = dense_results
                mock_sparse.return_value = sparse_results
                
                query_embedding = [0.1] * 1024
                query_terms = {1: 0.8}
                params = HybridSearchParams(alpha=0.7, use_rrf=False, topk=10)
                
                results = await retriever.hybrid_search(
                    query_embedding=query_embedding,
                    query_terms=query_terms,
                    params=params
                )
                
                # Verify both searches were called
                mock_dense.assert_called_once()
                mock_sparse.assert_called_once()
                
                # Results should be fused
                assert len(results) <= 10


class TestMilvusRetrieverEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_search_empty_results(self):
        """Test search with no results."""
        config = {"collection_name": "empty_results"}
        retriever = MilvusRetriever(config)
        
        mock_collection = MagicMock()
        retriever.collection = mock_collection
        mock_collection.search.return_value = [[]]  # Empty results
        
        results = await retriever.dense_search([0.1] * 1024)
        assert results == []
        
        results = await retriever.sparse_search({1: 0.8})
        assert results == []

    @pytest.mark.asyncio
    async def test_operations_without_collection(self):
        """Test operations that require collection when none exists."""
        config = {"collection_name": "no_collection"}
        retriever = MilvusRetriever(config)
        
        # All these should fail gracefully
        with pytest.raises((RuntimeError, AttributeError)):
            await retriever.dense_search([0.1] * 1024)
            
        with pytest.raises((RuntimeError, AttributeError)):
            await retriever.sparse_search({1: 0.8})
            
        with pytest.raises((RuntimeError, AttributeError)):
            await retriever.get_document("test")

    def test_config_edge_cases(self):
        """Test configuration edge cases."""
        # Minimal config
        config = {}
        retriever = MilvusRetriever(config)
        assert retriever.collection_name == "askme_default"
        
        # Config with extra fields
        config = {
            "collection_name": "test",
            "extra_field": "should_be_ignored",
            "another_extra": 123
        }
        retriever = MilvusRetriever(config)
        assert retriever.collection_name == "test"
        # Extra fields should not cause errors

    @pytest.mark.asyncio
    async def test_insert_documents_validation(self):
        """Test document validation during insertion."""
        config = {"collection_name": "validation_test"}
        retriever = MilvusRetriever(config)
        
        mock_collection = MagicMock()
        retriever.collection = mock_collection
        
        # Document without embedding
        doc_no_embedding = Document(
            id="no_embed",
            content="No embedding",
            metadata={}
        )
        
        with pytest.raises(ValueError, match="missing dense embedding"):
            await retriever.insert_documents([doc_no_embedding])
        
        # Document without sparse embedding
        doc_no_sparse = Document(
            id="no_sparse",
            content="No sparse",
            metadata={},
            embedding=[0.1] * 1024
        )
        
        with pytest.raises(ValueError, match="missing sparse embedding"):
            await retriever.insert_documents([doc_no_sparse])


class TestMilvusRetrieverIntegration:
    """Integration-style tests."""

    @pytest.mark.asyncio
    async def test_full_workflow_simulation(self):
        """Test a complete workflow with all mocks."""
        config = {"collection_name": "integration_test", "dimension": 768}
        retriever = MilvusRetriever(config)
        
        with patch('askme.retriever.milvus_retriever.connections'):
            with patch('askme.retriever.milvus_retriever.utility') as mock_utility:
                with patch('askme.retriever.milvus_retriever.Collection') as mock_collection_class:
                    with patch('askme.retriever.milvus_retriever.CollectionSchema'):
                        with patch('askme.retriever.milvus_retriever.FieldSchema'):
                            
                            # Setup mocks
                            mock_utility.has_collection.return_value = False
                            mock_collection = MagicMock()
                            mock_collection_class.return_value = mock_collection
                            
                            # 1. Connect
                            await retriever.connect()
                            
                            # 2. Create collection
                            await retriever.create_collection(dimension=768)
                            assert retriever.collection is not None
                            
                            # 3. Insert documents
                            docs = [
                                Document(
                                    id="integration_doc",
                                    content="Integration test document",
                                    metadata={"test": "integration"},
                                    embedding=[0.1] * 768,
                                    sparse_embedding={1: 0.9, 2: 0.7}
                                )
                            ]
                            
                            mock_result = MagicMock()
                            mock_result.primary_keys = ["integration_doc"]
                            mock_collection.insert.return_value = mock_result
                            
                            result_ids = await retriever.insert_documents(docs)
                            assert result_ids == ["integration_doc"]
                            
                            # 4. Search
                            mock_hit = MagicMock()
                            mock_hit.id = "integration_doc"
                            mock_hit.distance = 0.95
                            mock_hit.entity = {
                                "content": "Integration test document",
                                "metadata": '{"test": "integration"}',
                                "id": "integration_doc"
                            }
                            mock_collection.search.return_value = [[mock_hit]]
                            
                            results = await retriever.dense_search([0.1] * 768, topk=1)
                            assert len(results) == 1
                            assert results[0].document.id == "integration_doc"
                            
                            # 5. Cleanup
                            await retriever.disconnect()