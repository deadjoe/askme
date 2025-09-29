<img src="./logo.png" alt="askme banner image" width="30%">

# askme

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/FastAPI-ready-009688)](https://fastapi.tiangolo.com/)
[![Tests](https://img.shields.io/badge/tests-pytest-green)](https://pytest.org/)
[![Coverage](https://img.shields.io/badge/coverage-~88%25-success)](./htmlcov)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Production-ready hybrid RAG (Retrieval-Augmented Generation) system featuring dual-model embedding architecture (Qwen3-8B + BGE-M3), intelligent Qwen3 reranking, multi-backend vector database support, and comprehensive evaluation framework.

## Features

- **Dual-Model Hybrid Search**: Qwen3-Embedding-8B (4096D dense) + BGE-M3 (sparse lexical weights) with configurable fusion (Alpha, RRF, relative scoring)
- **Intelligent Reranking**: Local Qwen3-Reranker-8B with optional BGE fallback and Cohere Rerank 3.5 integration
- **Query Enhancement**: HyDE and RAG-Fusion techniques for improved recall and comprehensive coverage
- **Multi-Backend Support**: Weaviate (primary), Milvus 2.5+ (with sparse BM25), and Qdrant vector databases
- **Comprehensive Evaluation**: TruLens RAG Triad, Ragas v0.2+, offline local LLM judges, and embedding similarity metrics with A/B testing capabilities
- **Production Ready**: FastAPI service with health checks, Docker deployment, monitoring, and extensive configuration

## Architecture

```
Documents → Dual Embedding (Qwen3 Dense + BGE-M3 Sparse) → Vector DB (Hybrid Index) →
Query → Dual Encoding → Hybrid Retrieval (topK=50) → Qwen3 Rerank (topN=8) →
LLM Generate → Answer with Citations → Evaluate
```

### Core Technologies

- **Embeddings**: Hybrid dual-model architecture using Qwen3-Embedding-8B (4096D dense) + BGE-M3 (sparse lexical weights only)
- **Vector Database**: Weaviate (primary), Milvus 2.5+/Qdrant alternatives with hybrid search support
- **Reranking**: Qwen/Qwen3-Reranker-8B (local default), optional BGE reranker and Cohere Rerank 3.5 fallback
- **Framework**: FastAPI with Python 3.10+, uvicorn ASGI server
- **Evaluation**: TruLens + Ragas with configurable embedding backends, local LLM judges, and automated quality thresholds
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

# Start vector database (Weaviate by default, or Milvus for hybrid search)
docker compose -f docker/docker-compose.yaml --profile weaviate up -d weaviate
# OR for Milvus with hybrid search support:
# docker compose -f docker/docker-compose.yaml up -d milvus

# Start API server using startup script
./scripts/start-api.sh --skip-heavy-init --reload

# Alternative: Start API server directly
# ASKME_SKIP_HEAVY_INIT=1 uv run uvicorn askme.api.main:app --port 8080 --reload
```

### Basic Usage

```bash
# Ingest documents
./scripts/ingest.sh /path/to/documents --tags="project,documentation"

# Ask questions
./scripts/answer.sh "What is machine learning?"

# Retrieve documents only (for debugging/tuning)
./scripts/retrieve.sh "hybrid search techniques" --topk=50 --alpha=0.7

# Quick end-to-end smoke test (requires curl + jq)
./query_test.sh "Who created BM25?"

# Run evaluation
./scripts/evaluate.sh --suite=baseline
```

### Query Script Parameters

The `./scripts/answer.sh` script supports comprehensive parameter tuning for optimal results:

| Parameter | Default | Range/Options | Description | Impact |
|-----------|---------|---------------|-------------|---------|
| `--topk=N` | 50 | 1-100 | Number of initial retrieval candidates | Higher = better recall, slower response |
| `--alpha=X` | 0.5 | 0.0-1.0 | Hybrid search weight (0=sparse, 1=dense) | 0.0=keyword matching, 1.0=semantic similarity |
| `--rrf` / `--no-rrf` | `--rrf` | boolean | Use RRF vs alpha fusion | RRF=stable ranking, alpha=direct weighting |
| `--rrf-k=N` | 60 | 1-200 | RRF fusion smoothing parameter | Lower=aggressive reranking, higher=conservative |
| `--reranker=TYPE` | `qwen_local` | `qwen_local`, `bge_local` | Reranking model selection | Qwen3=optimized reranker, BGE=legacy fallback |
| `--max-passages=N` | 8 | 1-20 | Final passages for LLM generation | More=richer context, risk of attention dilution |
| `--hyde` | disabled | boolean | Enable HyDE query expansion | Better for abstract/conceptual queries |
| `--rag-fusion` | disabled | boolean | Multi-query generation and fusion | Better coverage for complex questions |
| `--debug` | disabled | boolean | Include retrieval debug information | Shows timing and score details |
| `--format=FORMAT` | `text` | `text`, `json`, `markdown` | Output format selection | Choose based on consumption needs |
| `--verbose` | disabled | boolean | Enable verbose logging | Detailed execution information |
| `--api-url=URL` | `localhost:8080` | URL | Target API base URL | Override for remote deployments |

### Query Examples

```bash
# Dense semantic search for conceptual queries (favor Qwen3 embeddings)
./scripts/answer.sh "Explain machine learning principles" --alpha=0.8 --topk=50 --max-passages=12

# Sparse keyword search for specific terms (favor BGE-M3 lexical weights)
./scripts/answer.sh "Vector store indexing parameters" --alpha=0.2 --topk=50

# Enhanced query with expansion techniques
./scripts/answer.sh "How does attention mechanism work?" --hyde --rag-fusion --format=markdown

# Debug mode with detailed retrieval information
./scripts/answer.sh "Vector database comparison" --debug --verbose --format=json
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

  # Search parameters (optimized for dual-model architecture)
  dense_weight: 1.0    # Qwen3 dense embeddings weight
  sparse_weight: 0.3   # BGE-M3 sparse weight (official research optimal)

# Dual-model embedding configuration
embedding:
  backend: qwen3-hybrid           # Hybrid: Qwen3 dense + BGE-M3 sparse
  model: Qwen/Qwen3-Embedding-8B  # Primary dense embedding model
  dimension: 4096                 # Qwen3 embedding dimension
  normalize_embeddings: true
  batch_size: 16                  # Optimized for Qwen3 dense processing
  pooling_method: last_token      # Qwen3 optimal pooling

  # BGE-M3 sparse configuration (lexical weights only)
  sparse:
    enabled: true
    backend: bge_m3
    model: BAAI/bge-m3
    batch_size: 4                 # BGE-M3 corpus optimal
    query_batch_size: 12          # BGE-M3 query optimal
    max_length: 8192              # BGE-M3 official optimal
    use_fp16: true

# Reranking
rerank:
  local_backend: qwen_local
  local_model: Qwen/Qwen3-Reranker-8B
  local_enabled: true
  top_n: 8

# Generation
generation:
  provider: ollama       # simple | ollama | openai
  ollama_model: gpt-oss:120b-cloud  # Updated for dual-model architecture
  ollama_endpoint: http://localhost:11434
  openai_model: gpt-4o-mini
```

- Use `backend: qwen3-hybrid` for the dual-model architecture (Qwen3 dense + BGE-M3 sparse)
- Use `backend: qwen3` for dense-only Qwen3 embeddings (no sparse vectors)
- Use `backend: bge_m3` for legacy BGE-M3-only mode (both dense and sparse from BGE-M3)

### Startup Scripts

For convenient development and deployment, use the provided startup and shutdown scripts:

```bash
# Start API server with custom configuration
./scripts/start-api.sh --port 8080 --ollama-model gpt-oss:120b-cloud --vector-backend milvus

# Quick development start (skip heavy initialization)
./scripts/start-api.sh --skip-heavy-init --reload

# Show configuration without starting
./scripts/start-api.sh --dry-run

# Stop API server
./scripts/stop-api.sh

# Force stop all related processes
./scripts/stop-api.sh --all --force
```

### Environment Variables

All configuration can be overridden using environment variables with the `ASKME_` prefix:

#### Core Configuration
| Variable | Default | Description |
|----------|---------|-------------|
| `ASKME_VECTOR_BACKEND` | `weaviate` | Vector database backend (weaviate/milvus/qdrant) |
| `ASKME_ENABLE_OLLAMA` | `false` | Enable local Ollama LLM generation |
| `ASKME_SKIP_HEAVY_INIT` | `false` | Skip heavy service initialization |
| `ASKME_LOG_LEVEL` | `INFO` | Log level (DEBUG/INFO/WARNING/ERROR) |

#### Database Configuration
| Variable | Default | Description |
|----------|---------|-------------|
| `ASKME_DATABASE__HOST` | `localhost` | Database host address |
| `ASKME_DATABASE__PORT` | `19530` | Database port |
| `ASKME_DATABASE__MILVUS__HOST` | `localhost` | Milvus host address |
| `ASKME_DATABASE__MILVUS__PORT` | `19530` | Milvus port |
| `ASKME_DATABASE__MILVUS__USERNAME` | `""` | Milvus username |
| `ASKME_DATABASE__MILVUS__PASSWORD` | `""` | Milvus password |
| `ASKME_DATABASE__MILVUS__SECURE` | `false` | Milvus secure connection |
| `ASKME_DATABASE__MILVUS__COLLECTION_NAME` | `askme_hybrid` | Milvus collection name |
| `ASKME_DATABASE__WEAVIATE__URL` | `http://localhost:8081` | Weaviate connection URL |
| `ASKME_DATABASE__WEAVIATE__API_KEY` | `""` | Weaviate API key |
| `ASKME_DATABASE__WEAVIATE__CLASS_NAME` | `AskmeDocument` | Weaviate class name |
| `ASKME_DATABASE__QDRANT__URL` | `http://localhost:6333` | Qdrant connection URL |
| `ASKME_DATABASE__QDRANT__API_KEY` | `""` | Qdrant API key |
| `ASKME_DATABASE__QDRANT__COLLECTION_NAME` | `askme` | Qdrant collection name |

#### Generation Configuration
| Variable | Default | Description |
|----------|---------|-------------|
| `ASKME_GENERATION__PROVIDER` | `simple` | LLM provider (simple/ollama/openai) |
| `ASKME_GENERATION__OLLAMA_MODEL` | `gpt-oss:120b-cloud` | Ollama model name |
| `ASKME_GENERATION__OLLAMA_ENDPOINT` | `http://localhost:11434` | Ollama endpoint URL |
| `ASKME_GENERATION__MODEL_NAME` | `gpt-4` | Default model name |
| `ASKME_GENERATION__MAX_TOKENS` | `2048` | Maximum generation tokens |
| `ASKME_GENERATION__TEMPERATURE` | `0.1` | Generation temperature |
| `ASKME_GENERATION__TOP_P` | `0.9` | Top-p sampling parameter |
| `ASKME_GENERATION__OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model name |
| `ASKME_GENERATION__OPENAI_BASE_URL` | `https://api.openai.com/v1` | OpenAI API endpoint |
| `ASKME_GENERATION__OPENAI_API_KEY_ENV` | `OPENAI_API_KEY` | OpenAI API key env var name |

#### Embedding Configuration
| Variable | Default | Description |
|----------|---------|-------------|
| `ASKME_EMBEDDING__BACKEND` | `qwen3-hybrid` | Embedding backend (`qwen3-hybrid`, `qwen3`, `bge_m3`) |
| `ASKME_EMBEDDING__MODEL` | `Qwen/Qwen3-Embedding-8B` | Dense embedding model (Qwen3) |
| `ASKME_EMBEDDING__DIMENSION` | `4096` | Dense embedding dimension |
| `ASKME_EMBEDDING__BATCH_SIZE` | `16` | Embedding batch size |
| `ASKME_EMBEDDING__MAX_LENGTH` | `8192` | Maximum input length |
| `ASKME_EMBEDDING__NORMALIZE_EMBEDDINGS` | `true` | Normalize embeddings |
| `ASKME_EMBEDDING__USE_FP16` | `true` | Use FP16 precision |

**Sparse Embedding Configuration (BGE-M3 for lexical weights):**
| Variable | Default | Description |
|----------|---------|-------------|
| `ASKME_EMBEDDING__SPARSE__ENABLED` | `true` | Enable sparse vectors |
| `ASKME_EMBEDDING__SPARSE__BACKEND` | `bge_m3` | Sparse embedding backend |
| `ASKME_EMBEDDING__SPARSE__MODEL` | `BAAI/bge-m3` | Sparse embedding model (BGE-M3) |
| `ASKME_EMBEDDING__SPARSE__BATCH_SIZE` | `4` | BGE-M3 corpus processing batch size |
| `ASKME_EMBEDDING__SPARSE__QUERY_BATCH_SIZE` | `12` | BGE-M3 query processing batch size |
| `ASKME_EMBEDDING__SPARSE__USE_FP16` | `true` | Use FP16 for sparse model |

#### Hybrid Search Configuration
| Variable | Default | Description |
|----------|---------|-------------|
| `ASKME_HYBRID__MODE` | `rrf` | Hybrid search mode (rrf/alpha/relative_score/ranked) |
| `ASKME_HYBRID__ALPHA` | `0.5` | Alpha fusion parameter (0=sparse, 1=dense) |
| `ASKME_HYBRID__RRF_K` | `60` | RRF fusion parameter |
| `ASKME_HYBRID__TOPK` | `50` | Initial retrieval candidates |
| `ASKME_HYBRID__DENSE_WEIGHT` | `1.0` | Dense search weight |
| `ASKME_HYBRID__SPARSE_WEIGHT` | `0.3` | BGE-M3 sparse search weight (official optimal) |

#### Reranking Configuration
| Variable | Default | Description |
|----------|---------|-------------|
| `ASKME_RERANK__LOCAL_BACKEND` | `qwen_local` | Local reranking backend |
| `ASKME_RERANK__LOCAL_MODEL` | `Qwen/Qwen3-Reranker-8B` | Local reranking model |
| `ASKME_RERANK__LOCAL_ENABLED` | `true` | Enable local reranking |
| `ASKME_RERANK__LOCAL_BATCH_SIZE` | `8` | Qwen3-Reranker-8B optimal batch size |
| `ASKME_RERANK__TOP_N` | `8` | Final reranked passages |

#### API Server Configuration
| Variable | Default | Description |
|----------|---------|-------------|
| `ASKME_API__HOST` | `0.0.0.0` | API server host |
| `ASKME_API__PORT` | `8080` | API server port |
| `ASKME_API__WORKERS` | `1` | Number of worker processes |
| `ASKME_API__RELOAD` | `false` | Enable hot reload |
| `ASKME_API__ACCESS_LOG` | `true` | Enable access logging |

#### External Services
| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_BASE_URL` | - | OpenAI-compatible API endpoint |
| `OPENAI_API_KEY` | - | OpenAI API key |
| `COHERE_API_KEY` | - | Cohere API key (required for Cohere reranking) |
| `ASKME_RAGAS_LLM_MODEL` | - | Override LLM model for Ragas evaluation |
| `ASKME_RAGAS_EMBED_MODEL` | `BAAI/bge-m3` | Override embedding model for Ragas |

#### Performance & Monitoring
| Variable | Default | Description |
|----------|---------|-------------|
| `ASKME_PERFORMANCE__BATCH__EMBEDDING_BATCH_SIZE` | `16` | Dense embedding batch size (Qwen3 optimal) |
| `ASKME_PERFORMANCE__BATCH__SPARSE_BATCH_SIZE` | `4` | Sparse embedding batch size (BGE-M3 corpus) |
| `ASKME_PERFORMANCE__BATCH__QUERY_BATCH_SIZE` | `12` | Query batch size (BGE-M3 query optimal) |
| `ASKME_PERFORMANCE__BATCH__RERANK_BATCH_SIZE` | `8` | Qwen3-Reranker-8B optimal batch size |
| `ASKME_PERFORMANCE__TIMEOUTS__RETRIEVAL_TIMEOUT` | `15` | Retrieval timeout (seconds) |
| `ASKME_PERFORMANCE__TIMEOUTS__RERANK_TIMEOUT` | `30` | Reranking timeout (seconds) |
| `ASKME_PERFORMANCE__TIMEOUTS__GENERATION_TIMEOUT` | `60` | Generation timeout (seconds) |

#### Development & Testing
| Variable | Default | Description |
|----------|---------|-------------|
| `ASKME_SKIP_HEAVY_INIT` | `false` | Skip heavy service initialization |
| `ASKME_OLLAMA_READ_TIMEOUT` | `120` | Ollama read timeout (seconds) |
| `ASKME_OLLAMA_THINKING` | `false` | Enable Ollama thinking mode |
| `TOKENIZERS_PARALLELISM` | - | Control tokenizers parallelism |

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
    "reranker": "qwen_local",
    "max_passages": 8,
    "include_debug": true
  }'
```

## Deployment

### Docker Deployment

```bash
# Full stack with Milvus (recommended for hybrid search)
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
./scripts/evaluate.sh --suite=baseline --alpha=0.5 --topk=50 --topn=8
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
     ./scripts/retrieve.sh "query" --topk=50 --alpha=0.7
     # Incorrect
     ./scripts/retrieve.sh "query" --topk 50 --alpha 0.7
     ```

3. **Slow Retrieval Performance**
   - Check hybrid search parameters (alpha, RRF vs alpha fusion)
   - Verify vector database connection and indexing
   - Monitor embedding service latency

4. **Poor Reranking Quality**
   - Ensure local Qwen3-Reranker-8B model is properly loaded
   - Verify batch size settings (optimal: 8 for Qwen3-Reranker)
   - Check BGE fallback configuration if needed
   - Verify reranking score thresholds

5. **Memory Issues**
   - Adjust batch sizes in `configs/askme.yaml` (Qwen3: 16, BGE-M3 corpus: 4, query: 12, reranker: 8)
   - Use `ASKME_SKIP_HEAVY_INIT=1` for development
   - Monitor dual-model memory usage (Qwen3-8B + BGE-M3 + Qwen3-Reranker)
   - Consider using CPU backend for BGE-M3 if MPS memory leaks occur

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

- [BAAI](https://github.com/FlagOpen/FlagEmbedding) for BGE-M3 sparse embeddings
- [Qwen Team](https://github.com/QwenLM) for Qwen3-Embedding-8B and Qwen3-Reranker-8B models
- [Milvus](https://milvus.io/) for hybrid search capabilities with sparse BM25 support
- [TruLens](https://www.trulens.org/) and [Ragas](https://docs.ragas.io/) for evaluation frameworks
- [FastAPI](https://fastapi.tiangolo.com/) for the high-performance web framework

---
