# askme — 核心RAG迭代计划（仅聚焦主干能力）

目标：在 2–3 个短迭代内，把“摄取 → 混合检索 → 重排 → 生成 → 评测”主干能力打磨到“可严苛评测、质量稳定、效果显著”的水平；非核心与非功能项（安全、运维、UI 等）暂缓。

**质量与效果度量（退出准则）**
- 端到端评测：Ragas（faithfulness ≥ 0.70、context_precision ≥ 0.60）在 `data/eval/baseline_qa.jsonl` 上稳定复现；不可用时启发式得分需持续上升并趋稳。
- 实际检索质量：TopK=50 → 重排 TopN=8 的上下文命中与覆盖可通过样例复核（含 Weaviate/Milvus 两路径抽样）。
- 回归保障：核心路径单测/集成测全绿；主模块 mypy/flake8 严格通过；关键函数圈复杂度下降。

**范围与聚焦**
- 检索：Weaviate 原生 hybrid(alpha) 与 Milvus 2.5（BM25 + dense + RRF/alpha）两套主路径打磨；保证“结果稳定 + 可解释 + 可调参”。
- 重排：本地 BGE reranker 为主，内容长度与批处理参数合理控制；必要时保留 Cohere 兜底但不作为主路径依赖。
- 生成：保持 Ollama/OpenAI 兼容；离线模板不变但优化引用表达以利评测可读性（不破坏现有测试约定）。
- 评测：以 Ragas 为主，TruLens 做可选；确保 /eval/run 在无外网/无密钥时也稳定可跑并给出对比趋势。

---

## 迭代路线图（只列核心）

第 0 轮（今天，准备与校准）
- 清点评测基线：确认 `data/eval/baseline_qa.jsonl` 可用（已存在）。
- 修正 Weaviate collection 向量索引配置，确保近邻检索与 hybrid 性能稳定。
- 优化本地重排：对超长段落做轻量截断（按配置近似字符上限），提升速度与稳定性。
- 跑通 quick 回归：本地无外网/无密钥情况下，/eval/run（baseline，sample_size=3）可稳定返回，记录初始分。

第 1 轮（检索与重排的“效果提升”）
- Weaviate 路径：核对 alpha=1/0 极值替代 BM25/dense-only 的近似质量；补充调参建议与默认值校准。
- 重排参数调优：`topK/TopN/batch_size` 梯度实验（小样本），锁定默认组合；为长文本截断阈值提供配置说明与合理默认。
- 模板生成表达：保留现有“Question/Passages/Sources”格式，补充 1 行“简要结论”摘要（不破坏单测）。
- 评测跑分：记录与对比第 0/1 轮指标，确认是否达到或逼近阈值；若未达标，复盘召回/重排/分块参数。

第 2 轮（稳固与清洁）
- mypy/flake8 只清理主干模块（retriever/ rerank/ embeddings/ generation/ api.routes），不做非核心改造。
- 降圈复杂度：将 `query_documents` 与 `/eval/run` 拆出子步骤函数（不改行为）。
- 最终回归 + 文档：更新默认参数与“最佳实践”，固化可重复跑分方法。

---

## 开发循环（每轮保持节奏）
- 设计与更新：本文件同步迭代目标与度量 →
- 实施改动：小步提交（检索/重排/生成 的最小变更）→
- 测试验证：pytest + /eval/run（quick baseline）→
- 记录评测：保存 run_id 与关键参数 →
- 文档更新：结论与下一步 →
- 推送与复盘：进入下一轮。

---

## 第 0 轮执行清单（落地项）
- [x] Weaviate collection 创建时补充 HNSW 向量索引配置（与 COSINE 距离对齐）。
- [x] 本地 BGE reranker 对长段落做轻量截断（按 `local_max_length` 的字符近似），避免评测与生产中的过长输入抖动。
- [x] 新增融合单测：验证 RRF 与 alpha 融合（tests/unit/test_search_fusion.py）。
- [x] Quick 评测：`POST /eval/run { suite: baseline, sample_size: 3 }`（在未启用向量检索与嵌入的条件下走启发式评测），记录 run 与分数，作为第 1 轮对照基线。

### 基线评测记录（第0轮）
- run_id: `af80eb55-d280-4346-98fe-07d01a01083f`
- 条件：`ASKME_SKIP_HEAVY_INIT=1`（未连接向量库/未加载嵌入模型，评测走启发式回退）
- 指标（overall_metrics）：
  - faithfulness: 0.00（阈值 0.70）
  - context_precision: 0.00（阈值 0.60）
  - answer_relevancy: 0.4213
  - 说明：该分数仅用于对照，真实提升需启用检索/重排/生成主路径后再评测。

---

## 第一轮真实基线（2025-09-10）

- 真实环境：Weaviate（8081/8082，本地）、BGE‑M3（本地）、Ollama gpt-oss:20b（本地）。
- 语料：`docs/`（3 文件 ≈ 21 chunks）。
- 参数网格（小样本，N=3）：`alpha ∈ {0, 0.5, 1}`；`topK ∈ {30,50,80}`；`topN ∈ {5,8,10}`。
- 主要观测（/query include_debug + 本地 BGE 语义相关度）：
  - semantic_relevancy_avg ≈ 0.7967；avg_citations ≈ 2；avg_latency_ms ≈ 1250（CPU + 本地LLM）。
  - 差异不显著（小样本/小语料），倾向选择低时延组合。
- 推荐默认（当前阶段，低时延优先）：
  - `alpha=0.0`，`topK=30`，`topN=5`。
- Ragas/TruLens 现状（本地化路径）：
  - 已显式绑定 Ollama(OpenAI 兼容) + HF BGE‑M3；少数指标分支仍可能尝试远端模型（例如 gpt-4o-mini），导致 404。
  - 兜底：即使子任务失败，评测流程不阻断；短期以“BGE 语义相关度 + /query include_debug + 人工 spot‑check”为主判据。

### 可复现实验步骤（本地 uv + docker + ollama）
- 准备：
  - `uv sync --dev`
  - `docker compose -f docker/docker-compose.yaml --profile weaviate up -d weaviate`
  - `ollama list` 确认 `bge-m3`、`gpt-oss:20b` 已就绪
- 启动 API：
  - 环境：`ASKME_ENABLE_OLLAMA=1 OPENAI_BASE_URL=http://localhost:11434/v1 OPENAI_API_KEY=ollama-local`
  - `uv run uvicorn askme.api.main:app --host 0.0.0.0 --port 8080`
- 导入语料：`./scripts/ingest.sh docs --tags="project,docs"`
- 快速检索验证：`./scripts/retrieve.sh "混合检索是什么" --alpha=0.0 --topk=30 --debug`
- 评测（保守指标 + 覆盖我们验证通过的本地路径）：
  - `./scripts/evaluate.sh --suite=baseline --metrics="answer_relevancy,context_precision" --sample-size=6 --alpha=0.0 --topk=30 --topn=5 --format=text`

---

## 第二轮计划（本地真实环境，聚焦可分辨提升）

- 语料扩充：新增 10–30 条领域问答/片段到 `data/eval/baseline_qa.jsonl` 或 `data/eval/custom_qa.jsonl`（已提交一个基础版 baseline_qa.jsonl）。
- 评测配置：
  - 先仅启用稳定通过的 ragas 指标（`answer_relevancy`, `context_precision`）。
  - faithfulness/recall 在 Ollama 端到端兼容稳定后再纳入阈值考核。
- 参数复跑与选型：
  - 网格细化到 `alpha∈{0.0,0.25,0.5}`、`topK∈{30,50,80}`、`topN∈{5,8,10}`，记录 run_id→参数→分数表。
  - 输出两套推荐：质量优先（更高 topK/topN）与时延优先（当前组合）。
- 工程化与清洁：
  - 仅针对核心模块进行 mypy/flake8/复杂度收敛（不改变行为）。

---

## 第三方评审对照核查（要点）

- 结论一致：以本地真实环境（Weaviate + BGE‑M3 + Ollama）做第一轮基线；参数对照和指标记录方式一致。
- 补充动作：将“配置覆盖”下沉到脚本层（现已在 `scripts/evaluate.sh` 增加 `--alpha/--topk/--topn/--overrides-json`），便于批量网格。
- 风险承接：ragas 对 OpenAI 兼容路径的内部 fallback 风险已在框架内兜底且在 DEV_PROGRESS.md 显式记录。

---

## Backlog（非核心 / 后续处理）

- 安全与运维：鉴权、速率限制细化、监控与追踪（Traces）完善。
- UI/可视化：检索/重排贡献度、延迟分解、指标对比视图。
- 多后端一致性：Milvus/Qdrant 的同构特性测试与对齐。
- 评测版本固定：对 ragas/trulens 版本进一步 pin 与适配，彻底消除 fallback 风险。
