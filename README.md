<img src="./logo.png" alt="askme banner image" width="30%">

# askme

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/FastAPI-ready-009688)](https://fastapi.tiangolo.com/)
[![Tests](https://img.shields.io/badge/tests-pytest-green)](https://pytest.org/)
[![Coverage](https://img.shields.io/badge/coverage-~88%25-success)](./htmlcov)

Production‑ready Hybrid RAG (Retrieval‑Augmented Generation) system with hybrid search, local reranking, optional cloud fallback, and built‑in evaluation.

## Features

- Hybrid search (alpha/rrf) combining BM25/sparse and dense vectors
- Local BGE reranker (FlagEmbedding) with optional Cohere fallback
- Query enhancement: HyDE + RAG‑Fusion (deterministic, no external LLM)
- Vector backends: Milvus 2.5+ (primary, with sparse BM25), Weaviate, Qdrant
- Evaluation: TruLens triad + Ragas 0.2 metrics（可选，自动降级）
- FastAPI service with health/readiness endpoints，脚本与 Docker 编排

## Architecture

```
Documents → Ingest → Vector DB → Query → Retrieve (topK=50)
→ Rerank (topN=8) → Generate → Answer + Citations → Evaluate
```

### Core Technologies

- Embeddings: BGE‑M3（dense+sparse）
- Vector DB: Milvus 2.5+（内建 BM25 + hybrid），Weaviate/Qdrant
- Reranking: BAAI/bge‑reranker‑v2.5‑gemma2‑lightweight（本地），Cohere Rerank 3.5（可选）
- Framework: FastAPI（Python 3.10+）
- Evaluation: TruLens + Ragas 0.2

## Quick Start

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) package manager
- Docker / Docker Compose（用于向量库与一体化部署）

### Installation

```bash
# 克隆与安装
git clone <repository-url>
cd askme
uv sync --dev

# 启动向量库（Milvus）
docker compose -f docker/docker-compose.yaml up -d milvus

# 启动 API（开发模式，可跳过重量初始化）
ASKME_SKIP_HEAVY_INIT=1 uv run uvicorn askme.api.main:app --port 8080 --reload
```

### Basic Usage

```bash
# 文档入库（自动判断 file/dir/url）
./scripts/ingest.sh /path/to/documents --tags="project,docs"

# 直接问答
./scripts/answer.sh "What is machine learning?"

# 仅检索（调参/调试）
./scripts/retrieve.sh "hybrid search" --topk 25 --alpha 0.7

# 评测
./scripts/evaluate.sh --suite=baseline
```

## Configuration

配置来源：`configs/askme.yaml`（若存在）→ 环境变量 → 默认值（Pydantic Settings）。核心项：

```yaml
# 向量后端（默认 milvus）
vector_backend: milvus  # milvus | weaviate | qdrant

hybrid:                 # 混合检索
  mode: rrf             # rrf | alpha | relative_score | ranked
  use_rrf: true         # 兼容标志
  alpha: 0.5
  rrf_k: 60
  topk: 50

embedding:
  model: BAAI/bge-m3

rerank:
  local_model: BAAI/bge-reranker-v2.5-gemma2-lightweight
  local_enabled: true
  cohere_enabled: false  # 可通过环境变量开启

generation:
  provider: simple       # simple | ollama | openai
  ollama_endpoint: http://localhost:11434
  openai_base_url: https://api.openai.com/v1
  openai_model: gpt-4o-mini
```

常用环境变量：
- `ASKME_SKIP_HEAVY_INIT=1` 开发时跳过重量初始化
- `ASKME_ENABLE_COHERE=1` + `COHERE_API_KEY` 启用 Cohere rerank
- `ASKME_ENABLE_OLLAMA=1` 或 `generation.provider=ollama` 启用本地 Ollama
- `OPENAI_BASE_URL` / `OPENAI_API_KEY`（评测与 OpenAI 兼容端口）

## API Endpoints

基路径按 router 注册：`/health`, `/ingest`, `/query`, `/eval`

### Health
- `GET /health/` 基础健康检查
- `GET /health/ready` 就绪检查
- `GET /health/live` 存活检查

### Ingest
- `POST /ingest/` 统一入库（file/dir，URL 未实现将返回 501）
- `POST /ingest/file`、`POST /ingest/directory` 精细化入库
- `GET /ingest/status/{task_id}` 任务状态
- `GET /ingest/stats` 全局统计

### Query
- `POST /query/` 混合检索 + 重排 + 生成（`include_debug` 可返回调试指标）
- `POST /query/retrieve` 仅检索（便于调参与调试）
- `GET /query/similar/{doc_id}` 相似文档（占位返回样例）
- `POST /query/explain` 检索解释（占位返回样例）

### Evaluation
- `POST /eval/run` 运行评测（TruLens + Ragas，自动回退）
- `GET /eval/runs/{run_id}` 查看评测结果
- `POST /eval/compare` 评测对比
- `GET /eval/runs` 列出最近评测
- `DELETE /eval/runs/{run_id}` 删除评测
- `GET /eval/metrics` 可用指标

## Development

```bash
# 安装开发依赖
uv sync --dev

# 运行测试（严格 markers 配置、asyncio=strict 已在 pyproject.toml 内）
uv run pytest -ra

# 覆盖率
uv run pytest --cov=askme --cov-report=term --cov-report=html

# 代码风格与类型
uv run black askme tests && uv run isort askme tests
uv run mypy askme
```

Pytest markers：`unit`、`integration`、`slow`。示例：`uv run pytest -m 'unit and not slow'`。

---

Built for the RAG community
