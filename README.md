<img src="./logo.png" alt="askme banner image" width="30%">

# askme

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/FastAPI-ready-009688)](https://fastapi.tiangolo.com/)
[![Tests](https://img.shields.io/badge/tests-pytest-green)](https://pytest.org/)
[![Coverage](https://img.shields.io/badge/coverage-~88%25-success)](./htmlcov)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Production-ready hybrid RAG (Retrieval-Augmented Generation) system with intelligent reranking, multi-backend vector database support, and comprehensive evaluation framework.

## Features

- **Hybrid Search**: Combines BM25/sparse and dense vector retrieval with configurable fusion methods (Alpha, RRF, relative scoring)
- **Intelligent Reranking**: Local BGE-reranker-v2.5 with Cohere Rerank 3.5 fallback for optimal relevance scoring
- **Query Enhancement**: HyDE and RAG-Fusion techniques for improved recall and comprehensive coverage
- **Multi-Backend Support**: Weaviate (primary), Milvus 2.5+ (with sparse BM25), and Qdrant vector databases
- **Comprehensive Evaluation**: TruLens RAG Triad, Ragas v0.2+, offline local LLM judges, and embedding similarity metrics with A/B testing capabilities
- **Production Ready**: FastAPI service with health checks, Docker deployment, monitoring, and extensive configuration

## Architecture

```
Documents → Ingest → Vector DB (Hybrid Index) → Query → Retrieve (topK=50) →
Rerank (topN=8) → LLM Generate → Answer with Citations → Evaluate
```

### Core Technologies

- **Embeddings**: BGE-M3 multilingual model (dense + sparse support)
- **Vector Database**: Weaviate (primary), Milvus 2.5+/Qdrant alternatives with hybrid search support
- **Reranking**: BAAI/bge-reranker-v2-m3 cross-encoder (local), Cohere Rerank 3.5 (cloud fallback)
- **Framework**: FastAPI with Python 3.10+, uvicorn ASGI server
- **Evaluation**: TruLens + Ragas with local embedding metrics, configurable LLM judges, and automated quality thresholds
- **Generation**: OpenAI-compatible, local Ollama, or template-based approaches

## Quick Start

### Prerequisites

- Python 3.10 or higher
- [uv](https://github.com/astral-sh/uv) package manager (recommended) or pip
- Docker and Docker Compose (for vector database and full deployment)

### Installation

```bash
# Clone the repository
git clone https://github.com/deadjoe/askme.git
cd askme

# Install dependencies
uv sync --dev

# Start vector database (Weaviate)
docker compose -f docker/docker-compose.yaml --profile weaviate up -d weaviate

# Start API server (development mode)
ASKME_SKIP_HEAVY_INIT=1 uv run uvicorn askme.api.main:app --port 8080 --reload
```

### Basic Usage

```bash
# Ingest documents
./scripts/ingest.sh /path/to/documents --tags="project,documentation"

# Ask questions
./scripts/answer.sh "What is machine learning?"

# Retrieve documents only (for debugging/tuning)
./scripts/retrieve.sh "hybrid search techniques" --topk=25 --alpha=0.7

# Quick end-to-end smoke test (requires curl + jq)
./query_test.sh "Who created BM25?"

# Run evaluation
./scripts/evaluate.sh --suite=baseline
```

## Configuration

Configuration is managed through `configs/askme.yaml` with environment variable overrides. Key parameters:

```yaml
# Vector backend selection
vector_backend: weaviate  # weaviate | milvus | qdrant

# Hybrid search configuration
hybrid:
  mode: rrf             # rrf | alpha | relative_score | ranked
  alpha: 0.5           # 0=sparse only, 1=dense only, 0.5=balanced
  rrf_k: 60            # RRF fusion parameter
  topk: 50             # Initial retrieval candidates

# Embedding model
embedding:
  model: BAAI/bge-m3
  dimension: 1024
  normalize_embeddings: true

# Reranking
rerank:
  local_model: BAAI/bge-reranker-v2-m3
  local_enabled: true
  cohere_enabled: false  # Enable via ASKME_ENABLE_COHERE=1
  top_n: 8

# Generation
generation:
  provider: ollama       # simple | ollama | openai
  ollama_model: gpt-oss:20b
  ollama_endpoint: http://localhost:11434
  openai_model: gpt-4o-mini
```

### Environment Variables

- `ASKME_SKIP_HEAVY_INIT=1` - Skip heavy service initialization during development
- `ASKME_API_URL` / `ASKME_API_KEY` - Target API base URL and optional auth for CLI scripts
- `ASKME_ENABLE_COHERE=1` + `COHERE_API_KEY` - Enable Cohere reranking
- `ASKME_ENABLE_OLLAMA=1` - Enable local Ollama generation
- `ASKME_RAGAS_LLM_MODEL` - Override local LLM judge used for evaluations (default `gpt-oss:20b`)
- `ASKME_RAGAS_EMBED_MODEL` - Override embedding model for Ragas metrics (default `BAAI/bge-m3`)
- `ASKME_TRULENS_LLM_MODEL` - Override TruLens evaluation model (falls back to `ASKME_RAGAS_LLM_MODEL`)
- `OPENAI_BASE_URL` / `OPENAI_API_KEY` - OpenAI-compatible endpoints for evaluation

## API Reference

### Health Endpoints
- `GET /health/` - Basic health check
- `GET /health/ready` - Readiness check for orchestration
- `GET /health/live` - Liveness check for orchestration

### Document Ingestion
- `POST /ingest/` - Universal document ingestion (file/directory)
- `POST /ingest/file` - Single file ingestion
- `POST /ingest/directory` - Directory ingestion with recursion
- `GET /ingest/status/{task_id}` - Task status monitoring
- `GET /ingest/stats` - Global ingestion statistics

### Query & Retrieval
- `POST /query/` - Hybrid search + reranking + generation pipeline
- `POST /query/retrieve` - Retrieval-only endpoint for debugging
- `GET /query/similar/{doc_id}` - Similar document discovery
- `POST /query/explain` - Retrieval explanation (debugging)

### Evaluation
- `POST /eval/run` - Execute evaluation pipeline with TruLens + Ragas
- `GET /eval/runs/{run_id}` - Retrieve evaluation results
- `POST /eval/compare` - A/B test comparison between runs
- `GET /eval/runs` - List recent evaluation runs
- `DELETE /eval/runs/{run_id}` - Delete evaluation run
- `GET /eval/metrics` - Available evaluation metrics

#### Evaluation Toolkit

- `./scripts/evaluate.sh` provides a unified CLI for starting suites, overriding retrieval parameters (`--alpha`, `--topk`, `--topn`), and choosing output formats (`text`, `json`, `table`).
- Embedding similarity metrics run locally when `embedding_service` is available, adding groundedness, context precision/recall, and answer relevance without leaving the host.
- Local LLM judge metrics (faithfulness, answer relevancy, context precision/recall) default to Ollama via the OpenAI-compatible API; override with `ASKME_RAGAS_LLM_MODEL`, `OPENAI_BASE_URL`, or standard OpenAI keys as needed.
- TruLens metrics automatically fall back to LiteLLM/OpenAI providers and respect `ASKME_TRULENS_LLM_MODEL`, enabling fully offline evaluation when paired with Ollama.
- Use `ASKME_RAGAS_EMBED_MODEL` to swap in alternative embedding backends for evaluation-only workloads without touching production retrieval settings.

### Example Query Request

```bash
curl -X POST "http://localhost:8080/query/" \
  -H "Content-Type: application/json" \
  -d '{
    "q": "What is machine learning?",
    "topk": 50,
    "alpha": 0.5,
    "use_rrf": true,
    "reranker": "bge_local",
    "max_passages": 8,
    "include_debug": true
  }'
```

## Deployment

### Docker Deployment

```bash
# Full stack with Milvus (recommended)
docker compose -f docker/docker-compose.yaml up -d

# Alternative vector databases
docker compose -f docker/docker-compose.yaml --profile weaviate up -d
docker compose -f docker/docker-compose.yaml --profile qdrant up -d

# With monitoring
docker compose -f docker/docker-compose.yaml --profile monitoring up -d
```

### Production Considerations

- **Performance Targets**: P95 < 1500ms retrieval, < 1800ms with reranking
- **Scaling**: ~50k documents per node, horizontal scaling via load balancing
- **Security**: Local-only by default, cloud services require explicit opt-in
- **Monitoring**: Prometheus metrics and Grafana dashboards included

## Development

### Setup Development Environment

```bash
# Install development dependencies
uv sync --dev

# Install pre-commit hooks
uv run pre-commit install

# Run tests with coverage
uv run pytest --cov=askme --cov-report=term --cov-report=html

# Code formatting and type checking
uv run black askme tests && uv run isort askme tests
uv run mypy askme
```

### Testing

The project includes comprehensive test coverage with pytest:

```bash
# Run all tests
uv run pytest -ra

# Run specific test categories
uv run pytest -m "unit and not slow"
uv run pytest -m integration
uv run pytest -m "slow"

# Run with coverage reporting
uv run pytest --cov=askme --cov-report=html
```

**Test Markers:**
- `unit`: Unit tests for individual components
- `integration`: Cross-component integration tests
- `slow`: Time-intensive tests (model loading, evaluation)

### Code Quality

The project maintains high code quality standards:

- **Formatting**: Black and isort for consistent code style
- **Type Checking**: MyPy with strict configuration
- **Linting**: Flake8 with Black-compatible settings
- **Security**: Bandit security analysis
- **Pre-commit Hooks**: Automated quality checks on commit

## Evaluation

### Built-in Evaluation Suites

```bash
# Comprehensive baseline evaluation
./scripts/evaluate.sh --suite=baseline

# Quick evaluation for CI/CD
./scripts/evaluate.sh --suite=quick --sample-size=3

# Custom dataset evaluation
./scripts/evaluate.sh --dataset="/path/to/qa_dataset.jsonl" --metrics="faithfulness,context_precision"

# Parameter tuning evaluation
./scripts/evaluate.sh --suite=baseline --alpha=0.3 --topk=75 --topn=10
```

### Available Metrics

**TruLens RAG Triad:**
- **Context Relevance**: How relevant retrieved context is to the query
- **Groundedness**: How well the answer is supported by retrieved context
- **Answer Relevance**: How relevant the answer is to the original query

**Ragas Metrics:**
- **Faithfulness**: Factual consistency of answer with retrieved context
- **Answer Relevancy**: Semantic relevance of answer to query
- **Context Precision**: Precision of retrieved context chunks
- **Context Recall**: Recall of relevant context chunks

### Quality Thresholds

- **TruLens Triad**: ≥ 0.7 (configurable)
- **Ragas Faithfulness**: ≥ 0.7 (configurable)
- **Context Precision**: ≥ 0.6 (configurable)
- **Answer Consistency**: ≥ 90% with fixed seeds

## Contributing

We welcome contributions to the askme project! Please follow these guidelines:

### Development Workflow

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes following the code style guidelines
4. Add or update tests for your changes
5. Ensure all tests pass and code quality checks succeed
6. Update documentation as needed
7. Submit a pull request with a clear description

### Code Style Guidelines

- **Language**: All code, documentation, and commit messages must be in English
- **Formatting**: Use Black (line length 88) and isort for imports
- **Type Hints**: All functions must include proper type annotations
- **Documentation**: Follow Google-style docstrings
- **Testing**: Maintain or improve test coverage (currently ~88%)

### Quality Gates

Before submitting a PR, ensure:

```bash
# All tests pass
uv run pytest

# Code formatting
uv run black askme tests && uv run isort askme tests

# Type checking
uv run mypy askme

# Security check
uv run bandit -r askme

# Basic evaluation passes
./scripts/evaluate.sh --suite=quick
```

## Troubleshooting

### Common Issues

1. **Milvus Container Startup Issues**
   - If Milvus fails to start with port binding errors, try using Weaviate instead:
     ```bash
     docker run --name weaviate -p 8081:8080 -p 8082:50051 -d \
       cr.weaviate.io/semitechnologies/weaviate:1.24.1 \
       --host 0.0.0.0 --port 8080 --scheme http
     ```
   - Update `configs/askme.yaml` to use `vector_backend: weaviate`
   - Ensure both HTTP (8081) and gRPC (8082) ports are exposed for Weaviate

2. **Script Command Syntax**
   - Use `--param=value` format for script parameters:
     ```bash
     # Correct
     ./scripts/retrieve.sh "query" --topk=25 --alpha=0.7
     # Incorrect
     ./scripts/retrieve.sh "query" --topk 25 --alpha 0.7
     ```

3. **Slow Retrieval Performance**
   - Check hybrid search parameters (alpha, RRF vs alpha fusion)
   - Verify vector database connection and indexing
   - Monitor embedding service latency

4. **Poor Reranking Quality**
   - Ensure local BGE-reranker model is properly loaded
   - Check Cohere API key and fallback configuration
   - Verify reranking score thresholds

5. **Memory Issues**
   - Adjust batch sizes in `configs/askme.yaml`
   - Use `ASKME_SKIP_HEAVY_INIT=1` for development
   - Monitor model memory usage (BGE-M3 + reranker)

6. **Evaluation Failures**
   - Check TruLens and Ragas library versions
   - Verify evaluation dataset format (JSONL with required fields)
   - Ensure OpenAI-compatible API access for evaluation LLM

### Getting Help

- Submit issues via [GitHub Issues](https://github.com/deadjoe/askme/issues)
- Review [CLAUDE.md](CLAUDE.md) for development context
- Check the codebase for implementation details and examples

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [BAAI](https://github.com/FlagOpen/FlagEmbedding) for BGE-M3 embeddings and reranker models
- [Milvus](https://milvus.io/) for hybrid search capabilities with sparse BM25 support
- [TruLens](https://www.trulens.org/) and [Ragas](https://docs.ragas.io/) for evaluation frameworks
- [FastAPI](https://fastapi.tiangolo.com/) for the high-performance web framework

---
