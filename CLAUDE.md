# askme — Claude Code Quickstart

## Project Overview
AskMe is a production-ready hybrid RAG (Retrieval-Augmented Generation) system that implements:
- **Hybrid Search**: BM25/sparse + dense vector retrieval with configurable fusion (alpha/RRF)
- **Intelligent Reranking**: Local BGE-reranker-v2.5 + Cohere Rerank 3.5 fallback
- **Query Enhancement**: HyDE and RAG-Fusion for improved recall
- **Comprehensive Evaluation**: TruLens RAG Triad + Ragas metrics
- **Multi-backend Support**: Milvus 2.5+ / Weaviate / Qdrant

## Architecture Flow
```
Documents → Ingest → Vector DB (Hybrid Index) → Query → Retrieve (topK=50) →
Rerank (topN=8) → LLM Generate → Answer with Citations → Evaluate
```

## Key Technologies
- **Embeddings**: BGE-M3 (multilingual, sparse+dense)
- **Vector DB**: Milvus 2.5+ with Sparse-BM25 support (primary)
- **Reranker**: BAAI/bge-reranker-v2.5-gemma2-lightweight (local)
- **Fallback**: Cohere Rerank 3.5 (cloud, requires ASKME_ENABLE_COHERE=1)
- **Evaluation**: TruLens + Ragas v0.2+
- **Framework**: FastAPI + Python 3.10+
- **Local LLM (optional)**: Ollama (enable via ASKME_ENABLE_OLLAMA=1)

## Core Commands

### Ingestion
```bash
# Basic ingestion
scripts/ingest.sh /path/to/documents

# With tags and metadata
scripts/ingest.sh /path/to/docs --tags="project,team" --overwrite=false
```

### Retrieval & Query
```bash
# Basic query with hybrid search
scripts/retrieve.sh "your question here" --alpha=0.5 --topk=50

# Enhanced query with HyDE and RAG-Fusion
scripts/answer.sh "complex question" --hyde --fusion --max-passages=8

# Reranking only
scripts/rerank.sh --model=bge_local --take=8
```

### Evaluation
```bash
# Run comprehensive evaluation suite
scripts/evaluate.sh --suite=baseline

# Custom evaluation with specific metrics
scripts/evaluate.sh --metrics="faithfulness,context_precision"

# Quick suite (small sample for CI)
scripts/evaluate.sh --suite=quick --sample-size=3
```

## Configuration Management
Primary config: `configs/askme.yaml`

Key parameters:
- `hybrid.alpha`: 0.5 (equal weight), >0.5 (dense bias), <0.5 (sparse bias)
- `hybrid.rrf_k`: 60 (RRF fusion parameter)
- `rerank.local_model`: BGE-reranker path
- `embedding.model`: BAAI/bge-m3

## Development Workflow

### Local Development
```bash
# Install dependencies
uv sync --dev

# Start API server
uvicorn askme.api.main:app --reload --port 8080

# Run tests
pytest tests/

# Format code
black askme/ && isort askme/
```

### Docker Deployment
```bash
# Start full stack (Milvus + API)
docker-compose -f docker/docker-compose.yaml up -d

# Health check
curl http://localhost:8080/health
```

## API Endpoints

### Core Endpoints
- `POST /ingest`: Document ingestion with metadata
- `POST /query`: Hybrid search with reranking
- `POST /eval/run`: Evaluation pipeline execution
- `GET /health`: System health check

### Query Parameters
```json
{
  "q": "search query",
  "topK": 50,
  "alpha": 0.5,
  "use_rrf": true,
  "rrf_k": 60,
  "use_hyde": false,
  "use_rag_fusion": false,
  "reranker": "bge_local",
  "max_passages": 8,
  "filters": {"tags": ["project"]}
}
```

## Performance Targets
- **Latency**: P95 < 1500ms (retrieval), < 1800ms (with reranking)
- **Scale**: ~50k documents per node
- **Quality**: TruLens Triad ≥ 0.7, Ragas faithfulness ≥ 0.7
- **Consistency**: ≥90% answer consistency with fixed seeds

## Security & Privacy
- **Default**: Local-only models and storage
- **Cloud Services**: Explicit opt-in via environment variables
- **Audit Logging**: All external API calls tracked
- **Data Isolation**: Per-project vector namespaces

## Testing Strategy
```bash
# Unit tests
pytest askme/tests/unit/

# Integration tests
pytest askme/tests/integration/

# End-to-end evaluation
pytest askme/tests/e2e/ --slow
```

## Troubleshooting

### Common Issues
1. **Slow retrieval**: Check hybrid parameters, consider RRF vs alpha
2. **Poor reranking**: Verify local model loading, check Cohere fallback
3. **Memory issues**: Adjust batch sizes in configs/askme.yaml
4. **Evaluation failures**: Check TruLens/Ragas versions and API keys

### Environment Variables
```bash
# Required
ASKME_VECTOR_BACKEND=milvus  # or weaviate, qdrant
ASKME_EMBEDDING_MODEL=BAAI/bge-m3

# Optional
ASKME_ENABLE_COHERE=1        # Enable Cohere reranking
ASKME_ENABLE_OLLAMA=1        # Enable local Ollama generator
ASKME_LOG_LEVEL=INFO
ASKME_BATCH_SIZE=32
```

## Claude Code Integration

### Plan Mode Usage
- Always use Plan mode for complex changes
- Review hybrid parameter impacts before applying
- Validate evaluation metrics after modifications

### Allowed Tools
- **Edit/Write**: Code modifications
- **Bash**: git operations, python scripts, docker commands
- **Read/Grep**: Codebase exploration
- **MCP**: External tool integration

### Custom Commands
Located in `.claude/commands/`:
- `askme-retrieval.md`: Hybrid search testing
- `askme-eval.md`: Quick evaluation runs
- `askme-config.md`: Parameter tuning assistance

## Quality Gates
Before committing:
1. All tests pass: `pytest`
2. Code formatting: `black . && isort .`
3. Type checking: `mypy askme/`
4. Basic evaluation: `scripts/evaluate.sh --quick`

## Contributing
- Commit messages: Professional, concise, English
- Don't include Cluade as co-author in new commit and keep checking the previous commited and pushed commits to prevent Claude as the Co-author
- Documentation: Technical accuracy, open-source ready
- Code comments: English, focused on "why" not "what"
- APIs: RESTful, well-documented, backward compatible

## References
- Product Spec: `docs/askme_Product_Spec.md`
- Design Doc: `docs/askme_Design_Doc.md`
- Milvus 2.5 Hybrid: https://milvus.io/docs/multi-vector-search.md
- BGE Models: https://huggingface.co/BAAI/bge-m3
- TruLens: https://www.trulens.org/getting_started/core_concepts/rag_triad/
- Ragas: https://docs.ragas.io/en/stable/concepts/metrics/
