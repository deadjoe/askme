# Product Spec — askme

> Version: 2025-09 • Scope: “Hybrid + Rerank + Generate + Evaluate” 最小闭环（个人/小组级）

## 0) 摘要
- **目标**：落地一个稳健的 RAG 最小闭环：**Hybrid 检索（BM25/稀疏 + 稠密）→ 轻量级重排 → 生成 → 评测**，强调 **准、稳、可测**。
- **非目标**：本阶段不纳入 GraphRAG / 多代理编排 / 复杂工作流。优先单体或轻量部署，避免重依赖外部搜索集群。

## 1) 关键用户故事
1. **索引**：一键导入 PDF/Markdown/HTML/纯文本/代码片段；可视化看到索引状态与失败重试。
2. **问答**：对任意查询执行 **Hybrid** 召回、**Rerank** 二次重排，返回含**高亮片段与可溯源引用**的答案。
3. **评测**：在仪表盘查看 **TruLens RAG Triad**（Context Relevance / Groundedness / Answer Relevance）与 **Ragas** 指标，并对比不同参数与模型的 A/B。  
4. **可配置**：可切换 **本地 BGE Reranker** 与 **Cohere Rerank 3.5** 云兜底；可调整 Hybrid `alpha` 或启用 **RRF/RelativeScoreFusion**；按需开启 **HyDE / RAG-Fusion**。

## 2) 范围与需求
### 2.1 功能性需求
- **索引层**  
  - 向量库任选其一：**Milvus 2.5+**（引入 Sparse-BM25/多向量混合与 `FunctionType.BM25` 生成稀疏向量）、**Weaviate ≥1.17**（Hybrid + 可配置融合：`alpha`/Relative-Score-Fusion/Ranked-Fusion）、**Qdrant ≥1.10**（新 Query API 的 **server-side hybrid**）。  
  - 嵌入模型：**BGE-M3**（多语、多功能、多粒度；支持稀疏/稠密/多向量）。  
  - 支持元数据（路径、段落/标题、作者、时间、标签）与增量/并发索引。
- **检索层**  
  - **Hybrid**：稠密 + BM25/稀疏。默认 `alpha = 0.5`；或使用 **排名/相对分数/（在支持处）RRF** 融合。  
  - **召回参数**：`topK = 50`（可调），支持过滤与多字段检索。
- **重排层**  
  - 本地优先：**BAAI/bge-reranker-v2.5-gemma2-lightweight**；云兜底：**Cohere Rerank 3.5**。  
  - 典型流程：将 `topK=50~100` 的候选送入重排，取 `topN=5~12` 用于生成。
- **检索增强**  
  - **HyDE**：由 LLM 生成“假设文档”后再检索，提高零/少样本与长尾召回。  
  - **RAG-Fusion**：多子查询 → 并行检索 → 融合（RRF/RelativeScoreFusion），提升稳健性与覆盖度。
- **生成层**  
  - 对接你常用的 LLM（本地/私有云/商用 API），返回**结构化答案**（含引用与高亮）。
- **评测层**  
  - **TruLens**：RAG Triad（Context Relevance / Groundedness / Answer Relevance）。  
  - **Ragas v0.2+**：`faithfulness`、`answer_relevancy`、`context_precision/recall` 等，支持离线与线上抽样回归。

### 2.2 非功能性需求
- **隐私与默认策略**：默认仅使用本地模型与本地存储；云重排须显式开启（环境变量）。
- **性能**：在 ~50k 文档规模、单节点下，**P95 延迟** 纯检索 < 1500 ms；带本地重排 < 1800 ms（以参考实现为准，可按硬件调整）。
- **可观测性**：记录检索/重排/生成耗时、召回贡献（BM25 vs 稠密）、命中分布、失败重试与警报。

## 3) 验收标准（Definition of Done）
- **一致性**：固定随机种子和参数时，答案一致性 ≥ **90%**（多次运行对比）。
- **质量阈值**：TruLens Triad 三项均值 ≥ **0.7**；Ragas **faithfulness ≥ 0.7**、**context_precision ≥ 0.6**（阈值可随语料调整）。
- **融合增益**：RRF/Relative-Score-Fusion + Rerank 的 **Top-N 覆盖率** 相对单路检索提升 ≥ **15%**。
- **增强收益**：对“无命中/弱命中”查询，开启 **HyDE / RAG-Fusion** 后可使可用召回条数增长 ≥ **20%**。

## 4) 版本与依赖（建议 pin）
- **Milvus 2.5+**（Sparse-BM25、`FunctionType.BM25`、多向量混合）；或 **Weaviate ≥1.17**（Hybrid + 可配置融合）；或 **Qdrant ≥1.10**（server-side hybrid）。  
- **Embeddings**：**BAAI/bge-m3**。  
- **Reranker**：**BAAI/bge-reranker-v2.5-gemma2-lightweight**（本地） / **Cohere Rerank 3.5**（云）。  
- **Eval**：**TruLens** + **Ragas v0.2+**。  
- **Claude Code**：CLI（Plan/Headless）、自定义 Slash Commands、MCP 连接。

## 5) 面向 Claude Code 的项目结构
```
askme/
  api/                 # FastAPI/Flask REST（/ingest, /query, /eval）
  ingest/              # 文档解析与索引
  retriever/           # Hybrid 适配层（Milvus/Weaviate/Qdrant）
  rerank/              # BGE 本地重排 & Cohere 兜底
  enhancer/            # HyDE & RAG-Fusion
  evals/               # TruLens & Ragas 脚本 + 基线数据
  configs/
    askme.yaml         # 模型/阈值/alpha/RRF.k 等
  docker/
    docker-compose.yaml
  scripts/
    ingest.sh  retrieve.sh  rerank.sh  answer.sh  evaluate.sh
  .claude/
    commands/          # 自定义 slash 命令（面向 Claude Code）
  .mcp.json            # MCP 服务器/工具接入
  CLAUDE.md            # **给 Claude Code 的上下文说明与指令**
  README.md
```

### `CLAUDE.md`（建议内容概要）
- 项目目标与关键路径：**ingest → retrieve → rerank → answer → evaluate**。  
- 参数/脚本：`scripts/*.sh`；如何切换 Cohere、如何启用 HyDE/RAG-Fusion、如何调整 `alpha`/`rrf_k`。  
- 运行说明：Plan 模式/Headless 模式、允许的工具白名单、MCP 服务器接入。

## 6) API（草案）
- `POST /ingest`：入参=文件或目录 URI；出参=taskId。  
- `POST /query`：入参=`q`, `topK`(50), `alpha`(0.5), `use_rrf`(bool), `use_hyde`(bool), `use_rag_fusion`(bool), `reranker`(`bge_local|cohere`), `max_passages`(8)。出参=答案、引用、打分、耗时。  
- `POST /eval/run`：运行 TruLens + Ragas，出参=各项指标与报告链接。

## 7) 部署（最小可跑，示意）
```yaml
# docker/docker-compose.yaml
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

## 8) 风险与缓解
- **索引耗时**：增量/批处理 + 低优先级队列。  
- **多语言质量差异**：默认 BGE-M3；必要时按语言/域分库。  
- **重排延迟**：批量重排、本地轻量模型优先；必要时 Cohere 兜底与限流。

## 9) 参考（核查与进一步阅读）
- **Milvus 2.5（Sparse-BM25/FunctionType.BM25、混合搜索）**：  
  - Intro 与博客（Sparse-BM25 与多向量混合）：https://milvus.io/blog/introduce-milvus-2-5-full-text-search-powerful-metadata-filtering-and-more.md  
  - 入门：Hybrid/全文搜索与 `FunctionType.BM25`：https://milvus.io/blog/get-started-with-hybrid-semantic-full-text-search-with-milvus-2-5.md  
  - 文档：Multi-Vector Hybrid（内置 BM25 生成稀疏向量示例）：https://milvus.io/docs/multi-vector-search.md
- **Weaviate Hybrid**（alpha/融合算法/Hybrid 概念）：  
  - 文档（Hybrid/alpha）：https://docs.weaviate.io/weaviate/search/hybrid  
  - 博客（Hybrid 详解/融合算法演进）：https://weaviate.io/blog/hybrid-search-fusion-algorithms
- **Qdrant**（server-side hybrid/Query API、重排教程）：  
  - 1.10+ Query API 与 Hybrid：https://qdrant.tech/articles/hybrid-search/  
  - 重排与 BM25 讲解：https://qdrant.tech/documentation/advanced-tutorials/reranking-hybrid-search/
- **BGE-M3 模型卡**：https://huggingface.co/BAAI/bge-m3
- **BGE-Reranker v2.5（gemma2-lightweight）**：https://huggingface.co/BAAI/bge-reranker-v2.5-gemma2-lightweight
- **Cohere Rerank 3.5**（发布/云集成）：  
  - Changelog：https://docs.cohere.com/changelog/rerank-v3.5  
  - Bedrock 公告：https://aws.amazon.com/blogs/machine-learning/cohere-rerank-3-5-is-now-available-in-amazon-bedrock-through-rerank-api/  
  - OCI 文档：https://docs.oracle.com/en-us/iaas/Content/generative-ai/cohere-rerank-3-5.htm
- **HyDE（论文）**：https://arxiv.org/abs/2212.10496 （PDF：https://arxiv.org/pdf/2212.10496）
- **RAG-Fusion**（多子查询 + RRF/相对分数融合，综述/实践）：  
  - arXiv 案例研究（2024）：https://arxiv.org/abs/2402.03367  
  - 博客/教程集合：https://weaviate.io/blog/hybrid-search-explained ，https://www.assembled.com/blog/better-rag-results-with-reciprocal-rank-fusion-and-hybrid-search
- **RRF（Reciprocal Rank Fusion）**（算法/参数 `k≈60` 实践）：  
  - Azure AI Search 文档（含 `k` 说明）：https://learn.microsoft.com/en-us/azure/search/hybrid-search-ranking
- **评测**：  
  - **TruLens RAG Triad**（官方文档/Quickstart）：https://www.trulens.org/getting_started/core_concepts/rag_triad/ ，https://www.trulens.org/getting_started/quickstarts/quickstart/  
  - **Ragas**（metrics 与 v0.2+ 文档）：https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/ ，https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_precision/ ，https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/
- **Claude Code（Plan/Headless/Slash/MCP）**：  
  - 最佳实践（Headless/Plan）：https://www.anthropic.com/engineering/claude-code-best-practices  
  - 概览/CLI 参考/常见工作流/设置：  
    https://docs.anthropic.com/en/docs/claude-code/overview  
    https://docs.anthropic.com/en/docs/claude-code/cli-reference  
    https://docs.anthropic.com/en/docs/claude-code/common-workflows  
    https://docs.anthropic.com/en/docs/claude-code/settings  
  - MCP（官方 docs 与站点）：  
    https://docs.anthropic.com/en/docs/mcp  
    https://modelcontextprotocol.io/
