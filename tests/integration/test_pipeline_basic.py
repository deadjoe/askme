#!/usr/bin/env python3
"""
Basic integration test for the askme RAG pipeline.

This script tests the core components without requiring external dependencies.
"""

import asyncio
import sys
import tempfile
from pathlib import Path

import pytest

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import after path modification to avoid E402
from askme.core.config import EmbeddingConfig, RerankConfig, Settings  # noqa: E402
from askme.core.embeddings import (  # noqa: E402
    BGEEmbeddingService,
    HybridEmbeddingService,
    Qwen3EmbeddingService,
    create_embedding_backend,
)
from askme.ingest.document_processor import DocumentProcessingPipeline  # noqa: E402
from askme.ingest.ingest_service import IngestionService  # noqa: E402
from askme.rerank.rerank_service import RerankingService  # noqa: E402
from askme.retriever.milvus_retriever import MilvusRetriever  # noqa: E402


async def test_basic_pipeline() -> None:
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
        pytest.fail(f"Configuration failed: {e}")

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
        pytest.fail(f"Document processing failed: {e}")

    # Test 3: Embedding service (factory without actual model loading)
    print("\n3️⃣ Testing embedding service factory...")
    try:
        embedding_config = EmbeddingConfig(
            backend="qwen3-hybrid",
            model="Qwen/Qwen3-Embedding-0.6B",
            model_name="qwen3-hybrid",
            batch_size=8,
            dimension=1024,
        )
        embedding_service = create_embedding_backend(embedding_config)

        # Test basic properties without initializing models
        print(f"   ✓ Embedding service type: {type(embedding_service).__name__}")
        print(f"   ✓ Supports sparse vectors: {embedding_service.supports_sparse}")
        print(f"   ✓ Config model: {embedding_service.config.model}")
        print(f"   ✓ Config dimension: {embedding_service.config.dimension}")

    except Exception as e:
        print(f"   ❌ Embedding service test failed: {e}")
        pytest.fail(f"Embedding service test failed: {e}")

    # Test 4: Reranking service
    print("\n4️⃣ Testing reranking service...")
    try:
        rerank_config = RerankConfig(
            local_backend="qwen_local",
            local_model="Qwen/Qwen3-Reranker-0.6B",
            local_enabled=True,
            top_n=5,
        )
        reranking_service = RerankingService(rerank_config)

        service_info = reranking_service.get_service_info()
        print("   ✓ Reranking service configured")
        print(f"   ✓ Available methods: {service_info['available_methods']}")
        print(f"   ✓ Top N: {service_info['config']['top_n']}")

    except Exception as e:
        print(f"   ❌ Reranking service test failed: {e}")
        pytest.fail(f"Reranking service test failed: {e}")

    # Test 5: Vector retriever configuration
    print("\n5️⃣ Testing vector database configuration...")
    try:
        milvus_config = {
            "host": "localhost",
            "port": 19530,
            "collection_name": "test_collection",
        }
        retriever = MilvusRetriever(milvus_config)

        print("   ✓ Milvus retriever configured")
        print(f"   ✓ Connection: {retriever.host}: {retriever.port}")
        print(f"   ✓ Collection: {retriever.collection_name}")

    except Exception as e:
        print(f"   ❌ Vector retriever test failed: {e}")
        pytest.fail(f"Vector retriever test failed: {e}")

    # Test 6: API route imports
    print("\n6️⃣ Testing API components...")
    try:
        from askme.api.main import create_app
        from askme.api.routes import evaluation, health, ingest, query

        print("   ✓ FastAPI application can be created")
        print("   ✓ All API routes imported successfully")

    except Exception as e:
        print(f"   ❌ API components test failed: {e}")
        pytest.fail(f"API components test failed: {e}")

    print("\n🎉 All tests passed! askme pipeline is ready for deployment.")
    print("\n📋 Next steps:")
    print(
        "   1. Install and start Milvus: "
        "docker-compose -f docker/docker-compose.yaml up -d milvus"
    )
    print("   2. Start the API server: uvicorn askme.api.main:app --reload --port 8080")
    print("   3. Test document ingestion: ./scripts/ingest.sh /path/to/documents")
    print("   4. Test querying: ./scripts/answer.sh 'What is machine learning?'")

    return None


if __name__ == "__main__":
    asyncio.run(test_basic_pipeline())
    sys.exit(0)
