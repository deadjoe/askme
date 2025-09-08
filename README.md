# askme

A production-ready hybrid RAG (Retrieval-Augmented Generation) system with intelligent reranking and comprehensive evaluation.

## Features

- **Hybrid Search**: Combines BM25/sparse and dense vector retrieval with configurable fusion
- **Intelligent Reranking**: Local BGE-reranker-v2.5 with Cohere Rerank 3.5 fallback
- **Query Enhancement**: HyDE and RAG-Fusion for improved recall
- **Multi-backend Support**: Milvus 2.5+, Weaviate, Qdrant
- **Comprehensive Evaluation**: TruLens RAG Triad + Ragas metrics
- **Production Ready**: Docker deployment, monitoring, extensive configuration

## Architecture

```
Documents → Ingest → Vector DB → Query → Retrieve (topK=50) → 
Rerank (topN=8) → LLM Generate → Answer with Citations → Evaluate
```

### Core Technologies

- **Embeddings**: BGE-M3 (multilingual, sparse+dense support)
- **Vector Database**: Milvus 2.5+ with Sparse-BM25 (primary), Weaviate/Qdrant support
- **Reranking**: BAAI/bge-reranker-v2.5-gemma2-lightweight (local), Cohere Rerank 3.5 (cloud)
- **Framework**: FastAPI with Python 3.10+
- **Evaluation**: TruLens + Ragas v0.2+

## Quick Start

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) package manager
- Docker and Docker Compose (for full deployment)

### Installation

```bash
# Clone repository
git clone <repository-url>
cd askme

# Install dependencies
uv sync

# Start vector database (Milvus)
docker-compose -f docker/docker-compose.yaml up -d milvus

# Start API server
uvicorn askme.api.main:app --reload --port 8080
```

### Basic Usage

```bash
# Ingest documents
./scripts/ingest.sh /path/to/documents --tags="project,docs"

# Ask questions
./scripts/answer.sh "What is machine learning?"

# Run evaluation
./scripts/evaluate.sh --suite=baseline
```

## Configuration

Core configuration in `configs/askme.yaml`:

```yaml
# Vector backend
vector_backend: milvus  # milvus | weaviate | qdrant

# Hybrid search parameters
hybrid:
  alpha: 0.5      # 0=sparse only, 1=dense only
  rrf_k: 60       # RRF fusion parameter
  topk: 50        # Initial candidates

# Models
embedding:
  model: "BAAI/bge-m3"
rerank:
  local_model: "BAAI/bge-reranker-v2.5-gemma2-lightweight"
  cohere_enabled: false  # Set ASKME_ENABLE_COHERE=1 to enable
```

## API Endpoints

### Core Operations
- `POST /ingest`: Document ingestion with metadata
- `POST /query`: Full question-answering pipeline
- `POST /query/retrieve`: Retrieval-only for testing
- `POST /eval/run`: Run evaluation suite
- `GET /health`: Health check

## Development

```bash
# Install development dependencies
uv sync --dev

# Run tests
pytest

# Format code
black askme/ && isort askme/

# Type checking
mypy askme/
```

---

Built for the RAG community