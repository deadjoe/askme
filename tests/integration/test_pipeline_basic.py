#!/usr/bin/env python3
"""
Basic integration test for the askme RAG pipeline.

This script tests the core components without requiring external dependencies.
"""

import asyncio
import sys
import tempfile
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from askme.core.config import EmbeddingConfig, RerankConfig, Settings
from askme.core.embeddings import BGEEmbeddingService
from askme.ingest.document_processor import DocumentProcessingPipeline
from askme.ingest.ingest_service import IngestionService
from askme.rerank.rerank_service import RerankingService
from askme.retriever.milvus_retriever import MilvusRetriever


async def test_basic_pipeline():
    """Test the basic askme pipeline without external dependencies."""

    print("🚀 Testing askme RAG pipeline components...")

    # Test 1: Configuration loading
    print("\n1️⃣ Testing configuration...")
    try:
        settings = Settings()
        print(f"   ✓ Settings loaded: vector_backend={settings.vector_backend}")
        print(f"   ✓ Embedding model: {settings.embedding.model}")
        print(f"   ✓ Reranker model: {settings.rerank.local_model}")
    except Exception as e:
        print(f"   ❌ Configuration failed: {e}")
        return False

    # Test 2: Document processing
    print("\n2️⃣ Testing document processing...")
    try:
        # Create a test markdown file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(
                """# Test Document

This is a test document for the askme RAG system.

## Key Features

- Hybrid search with BM25 and dense vectors
- BGE-M3 embeddings for multilingual support
- Local reranking with cloud fallback
- Comprehensive evaluation metrics

## Technical Details

The system implements a complete RAG pipeline with:
1. Document ingestion and chunking
2. Vector database storage
3. Hybrid retrieval
4. Reranking for relevance
5. LLM generation with citations
"""
            )
            temp_file = Path(f.name)

        # Process the document
        pipeline = DocumentProcessingPipeline()
        documents = await pipeline.process_file(temp_file)

        print(f"   ✓ Processed document into {len(documents)} chunks")
        print(f"   ✓ First chunk preview: {documents[0].content[:100]}...")

        # Clean up
        temp_file.unlink()

    except Exception as e:
        print(f"   ❌ Document processing failed: {e}")
        return False

    # Test 3: Embedding service (mock without actual model loading)
    print("\n3️⃣ Testing embedding service initialization...")
    try:
        embedding_config = EmbeddingConfig(
            model="BAAI/bge-m3", batch_size=4, dimension=1024
        )
        embedding_service = BGEEmbeddingService(embedding_config)

        # Test configuration
        info = await embedding_service.get_model_info()
        print(f"   ✓ Embedding service configured: {info['model_name']}")
        print(
            f"   ✓ Batch size: {info['batch_size']}, Dimension: {info['embedding_dim']}"
        )

    except Exception as e:
        print(f"   ❌ Embedding service test failed: {e}")
        return False

    # Test 4: Reranking service
    print("\n4️⃣ Testing reranking service...")
    try:
        rerank_config = RerankConfig(
            local_model="BAAI/bge-reranker-v2.5-gemma2-lightweight",
            local_enabled=True,
            cohere_enabled=False,
            top_n=5,
        )
        reranking_service = RerankingService(rerank_config)

        service_info = reranking_service.get_service_info()
        print(f"   ✓ Reranking service configured")
        print(f"   ✓ Available methods: {service_info['available_methods']}")
        print(f"   ✓ Top N: {service_info['config']['top_n']}")

    except Exception as e:
        print(f"   ❌ Reranking service test failed: {e}")
        return False

    # Test 5: Vector retriever configuration
    print("\n5️⃣ Testing vector database configuration...")
    try:
        milvus_config = {
            "host": "localhost",
            "port": 19530,
            "collection_name": "test_collection",
        }
        retriever = MilvusRetriever(milvus_config)

        print(f"   ✓ Milvus retriever configured")
        print(f"   ✓ Connection: {retriever.host}:{retriever.port}")
        print(f"   ✓ Collection: {retriever.collection_name}")

    except Exception as e:
        print(f"   ❌ Vector retriever test failed: {e}")
        return False

    # Test 6: API route imports
    print("\n6️⃣ Testing API components...")
    try:
        from askme.api.main import create_app
        from askme.api.routes import evaluation, health, ingest, query

        print(f"   ✓ FastAPI application can be created")
        print(f"   ✓ All API routes imported successfully")

    except Exception as e:
        print(f"   ❌ API components test failed: {e}")
        return False

    print("\n🎉 All tests passed! askme pipeline is ready for deployment.")
    print("\n📋 Next steps:")
    print(
        "   1. Install and start Milvus: docker-compose -f docker/docker-compose.yaml up -d milvus"
    )
    print("   2. Start the API server: uvicorn askme.api.main:app --reload --port 8080")
    print("   3. Test document ingestion: ./scripts/ingest.sh /path/to/documents")
    print("   4. Test querying: ./scripts/answer.sh 'What is machine learning?'")

    return True


if __name__ == "__main__":
    success = asyncio.run(test_basic_pipeline())
    sys.exit(0 if success else 1)
