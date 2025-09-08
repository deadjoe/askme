# askme

A production-ready hybrid RAG (Retrieval-Augmented Generation) system with intelligent reranking and comprehensive evaluation.

## < Features

- **Hybrid Search**: Combines BM25/sparse and dense vector retrieval with configurable fusion
- **Intelligent Reranking**: Local BGE-reranker-v2.5 with Cohere Rerank 3.5 fallback
- **Query Enhancement**: HyDE and RAG-Fusion for improved recall
- **Multi-backend Support**: Milvus 2.5+, Weaviate, Qdrant
- **Comprehensive Evaluation**: TruLens RAG Triad + Ragas metrics
- **Production Ready**: Docker deployment, monitoring, extensive configuration

## <× Architecture

```
Documents ’ Ingest ’ Vector DB ’ Query ’ Retrieve (topK=50) ’ 
Rerank (topN=8) ’ LLM Generate ’ Answer with Citations ’ Evaluate
```

### Core Technologies

- **Embeddings**: BGE-M3 (multilingual, sparse+dense support)
- **Vector Database**: Milvus 2.5+ with Sparse-BM25 (primary), Weaviate/Qdrant support
- **Reranking**: BAAI/bge-reranker-v2.5-gemma2-lightweight (local), Cohere Rerank 3.5 (cloud)
- **Framework**: FastAPI with Python 3.10+
- **Evaluation**: TruLens + Ragas v0.2+

## =€ Quick Start

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

## =Ö Documentation

- **[Product Specification](docs/askme_Product_Spec.md)**: Detailed requirements and scope
- **[Design Document](docs/askme_Design_Doc.md)**: Technical architecture and implementation
- **[CLAUDE.md](CLAUDE.md)**: Developer context for Claude Code integration

## =à Configuration

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

## =' API Endpoints

### Core Operations
- `POST /ingest`: Document ingestion with metadata
- `POST /query`: Full question-answering pipeline
- `POST /query/retrieve`: Retrieval-only for testing
- `POST /eval/run`: Run evaluation suite
- `GET /health`: Health check

### Example Query
```bash
curl -X POST "http://localhost:8080/query/" \
  -H "Content-Type: application/json" \
  -d '{
    "q": "What is machine learning?",
    "topk": 50,
    "alpha": 0.5,
    "use_rrf": true,
    "reranker": "bge_local",
    "max_passages": 8
  }'
```

## =Ê Evaluation

Run comprehensive evaluation with multiple metrics:

```bash
# Baseline evaluation
./scripts/evaluate.sh --suite=baseline

# Custom evaluation
./scripts/evaluate.sh --metrics="faithfulness,context_precision" --sample-size=100

# Quick test
./scripts/evaluate.sh --suite=quick --format=table
```

### Available Metrics

**TruLens RAG Triad:**
- Context Relevance: How relevant retrieved context is to query
- Groundedness: How well answer is supported by context
- Answer Relevance: How relevant answer is to query

**Ragas Metrics:**
- Faithfulness: Factual consistency of answer with context
- Answer Relevancy: Relevance of answer to query
- Context Precision/Recall: Quality of retrieved context

## =3 Docker Deployment

```bash
# Full stack deployment
docker-compose -f docker/docker-compose.yaml up -d

# With alternative vector databases
docker-compose -f docker/docker-compose.yaml --profile weaviate up -d
docker-compose -f docker/docker-compose.yaml --profile qdrant up -d
```

## = Security & Privacy

- **Default Local**: Uses only local models and storage
- **Cloud Opt-in**: Cloud services (Cohere) require explicit enablement
- **Audit Logging**: All external API calls are tracked
- **Data Isolation**: Per-project vector namespaces

## >ê Development

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

## =È Performance Targets

- **Latency**: P95 < 1500ms (retrieval), < 1800ms (with reranking)
- **Scale**: ~50k documents per node
- **Quality**: TruLens Triad e 0.7, Ragas faithfulness e 0.7
- **Consistency**: e90% answer consistency with fixed seeds

## =ã Roadmap

- [ ] GraphRAG integration
- [ ] Multi-agent orchestration
- [ ] Advanced chunking strategies
- [ ] Real-time evaluation monitoring
- [ ] Auto-parameter tuning

## > Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit a pull request

All code, documentation, and commit messages should be in English for open-source compatibility.

## =Ä License

This project is open source. See LICENSE file for details.

## <˜ Support

- Check [CLAUDE.md](CLAUDE.md) for development context
- Review [troubleshooting guide](docs/troubleshooting.md)
- Submit issues via GitHub Issues

---

Built with d for the RAG community