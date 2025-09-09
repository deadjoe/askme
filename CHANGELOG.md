## Changelog

### 2025-09-09 — Weaviate v4 hybrid + debug visibility, reranker fallback, API fixes

- Weaviate retriever
  - Add `doc_id` property and include it in return properties for stable provenance.
  - Delete logic prefers deleting by `doc_id` (query → uuid) with fallback to direct uuid.
  - Collection stats include `num_entities` via `aggregate.over_all(total_count=True)` when available.
  - Docker compose exposes gRPC `8082` to match client v4 requirements.
- Query API
  - Real-path error staging with `include_debug=true` returns `error` indicating which stage failed: `embedding` / `retrieval` / `rerank` / `generation`.
  - Retrieval-only endpoint syntax/indentation fixed; dense/BM25 overlap and latency breakdown preserved.
- Reranking
  - Local BGE reranker falls back to `BAAI/bge-reranker-base` when the configured model requires `trust_remote_code`.
- Embeddings / Generation
  - Optional imports for `FlagEmbedding` and `OpenAI` to ease tests and offline setups.
- Evaluation
  - `GET /eval/runs/{run_id}` returns a friendly placeholder (status `not_found`) instead of 404 when the run is missing.
- CLI
  - `scripts/answer.sh --debug` prints `retrieval_debug.error` when present.

Notes:
- Tests: 38 passed, 1 skipped (Weaviate contract).
- Storage: `data/eval_runs/` is now ignored by Git (except for `.gitkeep`).

