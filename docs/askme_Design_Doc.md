# Design Doc — askme

> Version: 2025-09 • Authors: you + Claude Code • Status: Draft (implementation-ready)

## A) 架构与数据流

### A.1 总体组件（文本图）
```
[Client/CLI/Claude Code]
          |
       (REST)
          v
+---------------------+       +-----------------------+
|  API Gateway        |  -->  |  Retriever (Hybrid)   |--->[Vector DB]
|  (FastAPI)          |       |  - BGE-M3 embed/query |    - Milvus 2.5 (Sparse-BM25/Hybrid)
+---------------------+       |  - BM25/sparse path   |    - 或 Weaviate / Qdrant
          |                   |  - alpha / fusion     |
          v                   +-----------------------+
+---------------------+                 |
|  Reranker           | <---------------+
|  - BGE v2.5 local   |       (topK)
|  - Cohere 3.5 cloud |
+---------------------+
          |
          v
+---------------------+       +-----------------------+
|  Enhancer           | <---- |  HyDE / RAG-Fusion    |
|  - prompt+fanout    |       |  - 多子查询 / 伪文档   |
+---------------------+       +-----------------------+
          |
          v
+---------------------+
|  Generator (LLM)    |--> 引用/高亮/元数据
+---------------------+
          |
          v
+---------------------+
|  Evals (TruLens &   |
|  Ragas pipelines)   |
+---------------------+
```

### A.2 关键算法与参数
- **Hybrid 权重 `alpha`**（Weaviate 示例）：`alpha ∈ [0,1]`；`0.5` 等权，>0.5 偏向稠密，<0.5 偏向关键词（BM25F）。  
- **融合策略**：在不同后端使用其**原生融合**（如 Weaviate 的 `relativeScoreFusion`/`rankedFusion`；Qdrant 的 server-side hybrid；Milvus 的多向量混合）。对需要“查询多样化”的场景，可采用 **RRF**（常用 `k≈60`）。  
- **BGE-M3 嵌入**：同一模型支持**稠密/稀疏/多向量**表示，覆盖多语言与多粒度。  
- **Rerank**：将 `topK=50~100` 候选送入 **BGE-Reranker-v2.5-gemma2-lightweight**（本地、低延迟、多语）；如需云兜底用 **Cohere Rerank 3.5**。取 `topN=5~12` 供生成。  
- **HyDE**：由 LLM 生成“假设文档”→ 编码 → 再检索，显著提升零/少样本召回。  
- **RAG-Fusion**：生成多子查询并行检索 → 融合（RRF/RelativeScoreFusion）以提升稳健性。

> 参考：Milvus 2.5（Sparse-BM25/FunctionType.BM25、Hybrid）、Weaviate Hybrid（alpha/融合）、Qdrant 1.10（server-side hybrid）、BGE-M3/BGE-Reranker、Cohere Rerank 3.5、HyDE、RAG-Fusion、RRF、TruLens/Ragas。详见文末链接。

## B) 接口设计（REST v0）
### B.1 `POST /ingest`
- **Body**
  ```json
  { "source": "file|dir|url", "path": "/abs/or/uri", "tags": ["team","project"], "overwrite": false }
  ```
- **Resp**
  ```json
  { "taskId": "uuid", "status": "queued" }
  ```

### B.2 `POST /query`
- **Body**
  ```json
  {
    "q": "string",
    "topK": 50,
    "alpha": 0.5,
    "use_rrf": true,
    "rrf_k": 60,
    "use_hyde": false,
    "use_rag_fusion": false,
    "reranker": "bge_local",
    "max_passages": 8,
    "filters": { "tags": ["project"] }
  }
  ```
- **Resp**
  ```json
  {
    "answer": "text with citations",
    "citations": [ { "doc_id": "id", "title": "t", "start": 123, "end": 256, "score": 0.87 } ],
    "retrieval_debug": {
      "bm25_hits": 40, "dense_hits": 45, "fusion": "rrf", "alpha": 0.5,
      "rerank_model": "bge-reranker-v2.5-gemma2-lightweight", "latency_ms": 740
    }
  }
  ```

## C) 配置与部署
### C.1 `configs/askme.yaml`（示例）
```yaml
vector_backend: milvus  # or weaviate|qdrant
hybrid:
  mode: rrf        # or alpha|relative_score|ranked
  alpha: 0.5
  rrf_k: 60
embedding:
  model: BAAI/bge-m3
  dim: 1024
rerank:
  local_model: BAAI/bge-reranker-v2.5-gemma2-lightweight
  cohere_fallback: false
enhancer:
  hyde: true
  rag_fusion: true
eval:
  trulens: {enabled: true}
  ragas: {enabled: true}
```

### C.2 Docker Compose（最小可跑）
```yaml
services:
  milvus:
    image: milvusdb/milvus:2.5.0
    ports: ["19530:19530","9091:9091"]
  askme-api:
    build: ./docker
    env_file: .env
    depends_on: [milvus]
    ports: ["8080:8080"]
```

## D) 评测方案
- **TruLens RAG Triad**：评测 **Context Relevance / Groundedness / Answer Relevance** 三项，作为端到端健康度核心信号。  
- **Ragas v0.2+**：使用 `faithfulness`、`answer_relevancy`、`context_precision/recall` 等指标；建立离线基线集 + 线上抽样回归，纳入 CI。

## E) 安全与隐私
- 默认**不出网**；云重排（Cohere 3.5）需显式开启 `ASKME_ENABLE_COHERE=1`。  
- 审计：记录所有外部调用（模型名/版本、入参大小、响应延迟与错误）。

## F) 风险与缓解
- **Hybrid 参数误设**：提供可视化与灰度配置（alpha / rrf_k），并配合 A/B。  
- **语料异构**：保持分库与多策略（BM25 权重上调覆盖术语密集场景）。  
- **重排延迟/成本**：优先本地轻量 reranker；量大时批量化；必要时 Cohere 兜底并限流。

## G) 面向 Claude Code 的集成

### G.1 `CLAUDE.md`（存仓库根，供自动上下文）
```md
# askme — Claude Code Quickstart

## Goals
- Build a hybrid RAG pipeline: Milvus 2.5 + BGE-M3 + RRF/alpha + BGE-Reranker v2.5 + HyDE/RAG-Fusion + TruLens/Ragas.

## Key scripts
- scripts/ingest.sh <path_or_glob>
- scripts/retrieve.sh "<query>" --alpha=0.5 --rrf --topk=50
- scripts/rerank.sh --model=bge_local --take=8
- scripts/answer.sh "<query>" --hyde --fusion --n=8
- scripts/evaluate.sh --suite=baseline

## Conventions
- Default to local reranker; enable Cohere via env ASKME_ENABLE_COHERE=1.
- Tune hybrid in configs/askme.yaml (alpha/rrf_k).
- Return citations+highlights with every answer.

## Claude Code tips
- Use **Plan Mode**; show the plan before changes.
- Use **Headless** for CI: `claude -p "<prompt>" --output-format stream-json`.
- Allowed tools: Edit, Bash(git:* / python*), MCP servers.
```

### G.2 自定义 Slash 命令（`.claude/commands/askme-retrieval.md`）
```md
Run hybrid retrieval for: $ARGUMENTS

Steps:
1) Read configs/askme.yaml; get alpha/rrf_k.
2) Run scripts/retrieve.sh "$ARGUMENTS" --alpha=$(alpha) --rrf --topk=50.
3) Print JSON with doc ids, scores, and which channel (BM25 vs dense) contributed.
```

### G.3 脚本示例（Claude 可直接执行）
```bash
# scripts/retrieve.sh
#!/usr/bin/env bash
set -euo pipefail
QUERY="${1:-}"; shift || true
python -m api.cli.retrieve --q "$QUERY" "$@"
```

## H) 参考（核查来源）
- **Milvus 2.5（Sparse-BM25/FunctionType.BM25、Hybrid）**：
  - https://milvus.io/blog/introduce-milvus-2-5-full-text-search-powerful-metadata-filtering-and-more.md
  - https://milvus.io/blog/get-started-with-hybrid-semantic-full-text-search-with-milvus-2-5.md
  - https://milvus.io/docs/multi-vector-search.md
- **Weaviate Hybrid/alpha/融合**：
  - https://docs.weaviate.io/weaviate/search/hybrid
  - https://weaviate.io/blog/hybrid-search-fusion-algorithms
- **Qdrant（server-side hybrid/Query API、重排教程）**：
  - https://qdrant.tech/articles/hybrid-search/
  - https://qdrant.tech/documentation/advanced-tutorials/reranking-hybrid-search/
- **BGE-M3 模型卡**：https://huggingface.co/BAAI/bge-m3
- **BGE-Reranker v2.5（gemma2-lightweight）/v2 族**：
  - https://huggingface.co/BAAI/bge-reranker-v2.5-gemma2-lightweight
  - https://huggingface.co/BAAI/bge-reranker-v2-gemma
- **Cohere Rerank 3.5**：
  - https://docs.cohere.com/changelog/rerank-v3.5
  - https://aws.amazon.com/blogs/machine-learning/cohere-rerank-3-5-is-now-available-in-amazon-bedrock-through-rerank-api/
  - https://docs.oracle.com/en-us/iaas/Content/generative-ai/cohere-rerank-3-5.htm
- **HyDE**：https://arxiv.org/abs/2212.10496 （PDF：https://arxiv.org/pdf/2212.10496）
- **RAG-Fusion / RRF**：
  - https://arxiv.org/abs/2402.03367
  - https://learn.microsoft.com/en-us/azure/search/hybrid-search-ranking
- **评测**：
  - TruLens（RAG Triad/Quickstart）：https://www.trulens.org/getting_started/core_concepts/rag_triad/ ，https://www.trulens.org/getting_started/quickstarts/quickstart/
  - Ragas（metrics）：https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/ ，https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_precision/ ，https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/
- **Claude Code（Plan/Headless/Slash/MCP/SDK）**：
  - 最佳实践（Headless/Plan）：https://www.anthropic.com/engineering/claude-code-best-practices
  - 概览/CLI/工作流/设置：
    https://docs.anthropic.com/en/docs/claude-code/overview
    https://docs.anthropic.com/en/docs/claude-code/cli-reference
    https://docs.anthropic.com/en/docs/claude-code/common-workflows
    https://docs.anthropic.com/en/docs/claude-code/settings
  - MCP（官方 docs 与站点/Quickstart）：
    https://docs.anthropic.com/en/docs/mcp
    https://modelcontextprotocol.io/
    https://modelcontextprotocol.io/quickstart/server
