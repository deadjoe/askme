# askme — Development Progress Log (ongoing)

This document captures the current state, decisions, and next steps so we can seamlessly resume work later. It is written for future you and the AI coding assistant to re-sync context quickly.

## Scope Recap
- Goal: Production-lean RAG loop — Ingest → Hybrid Retrieve → Rerank → Generate → Evaluate.
- Backends: Milvus 2.5 (BM25 + hybrid) or Weaviate (native hybrid). One is sufficient.
- Generators: simple (offline), Ollama (local), OpenAI-compatible (for 3rd-party compatible endpoints).
- Evaluation: TruLens (Triad) + Ragas (0.2+) with graceful fallback when providers are unavailable.

## What’s Implemented
- API foundation (FastAPI) with real service lifecycle
  - Lifespan now `initialize()`s heavy components so real path runs without mocks.
  - Global exception handler and CORS setup.
- Retrieval backends
  - Milvus retriever (BM25 + hybrid) — existing.
  - Weaviate retriever (NEW): native `hybrid(query, vector, alpha)` with filters; selected via `vector_backend`.
  - Query model carries `original_query` for backends that need raw text.
- Ingestion
  - `/ingest` routes into `IngestionService` for files/dirs; real task status at `/ig est/status/{task_id}` and stats at `/ingest/stats`.
  - Document processors: PDF/Markdown/HTML/Text; chunking strategies.
- Reranking
  - BGE local reranker + optional Cohere fallback; service initialized on startup.
- Generation
  - `SimpleTemplateGenerator` (offline, deterministic) and `LocalOllamaGenerator` (local HTTP) — already wired.
  - OpenAI-compatible generator (NEW): supports `base_url` + `api_key`, for third-party OpenAI-compatible endpoints.
- Evaluation
  - Endpoints updated to run real evaluation where possible.
  - Ragas runner (NEW) for `faithfulness/answer_relevancy/context_precision/context_recall`; fallback to heuristic if unavailable.
  - TruLens runner (NEW) for Triad (context_relevance/groundedness/answer_relevance); optional and skipped if provider not configured.
  - Pipeline runner (NEW): runs in-process RAG once (embed → retrieve → rerank → generate) to produce end-to-end samples.
  - Storage (NEW): eval runs persisted under `data/eval_runs/<run_id>.json`; `/eval/runs*` now use real storage.
- Scripts & Tests
  - `scripts/rerank.sh` (NEW)
  - Unit tests: generation template + OpenAI generator (monkeypatched client).
  - Integration: Weaviate contract test (conditional via `WEAVIATE_URL`).
- Config
  - `configs/askme.yaml`: `generation.provider: simple|ollama|openai`, OpenAI-compatible fields, Weaviate URL.

## How to Run (local)
- Start API
  - `uvicorn askme.api.main:app --reload --port 8080`
- Choose vector backend
  - Milvus: `configs/askme.yaml → vector_backend: milvus` (start compose services `etcd`, `minio`, then `milvus`).
  - Weaviate (recommended to start):
    - Minimal docker run: `docker run -d --name askme-weaviate -p 8081:8080 -e QUERY_DEFAULTS_LIMIT=25 -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true -e PERSISTENCE_DATA_PATH=/var/lib/weaviate cr.weaviate.io/semitechnologies/weaviate:1.24.8`
    - Configure `database.weaviate.url: "http://localhost:8081"` and set `vector_backend: weaviate`.
- Generation options
  - Simple (default): no extra setup.
  - Ollama local: `export ASKME_ENABLE_OLLAMA=1` (or set `generation.provider: ollama`).
  - OpenAI-compatible: set `generation.provider: openai`, optionally use `OPENAI_BASE_URL`, `OPENAI_API_KEY`.
- End-to-end evaluation (quick)
  - `scripts/evaluate.sh --suite=quick --sample-size=3`
  - List runs: `GET /eval/runs`
  - Get run: `GET /eval/runs/{run_id}`
- Weaviate contract test (optional)
  - `export WEAVIATE_URL="http://localhost:8081"`
  - `pytest tests/integration/test_weaviate_contract.py -q`

## Decisions & Rationale
- Weaviate added first as alt-backend
  - Simple native hybrid API with `alpha` makes it fast to tune; good ecosystem/docs.
- OpenAI-compatible API is the “baseline” LLM provider
  - Many third-party services provide this; we support `base_url` + `api_key` so local proxies (e.g., LiteLLM→Ollama) work.
- Evaluation always returns something
  - Order: TruLens (if configured) → Ragas (if installed) → heuristic fallback.
  - Results are persisted for later comparison.

## Current Status & Known Gaps
- End-to-end works locally with configured backend/model; mocks remain only as safe fallback.
- Gaps to address next:
  1) TruLens end-to-end: wrap the pipeline with TruLens App and record triad per sample; use OpenAI-compatible provider (can be local proxy → Ollama).
  2) CI quick suite: small sample run added to CI as a quality gate.
  3) Weaviate tests: expand contract tests, and add toggled CI run if port/profile is available.
  4) Input source processors: add JSON/DOCX and URL fetcher (default no network unless enabled).
  5) Security gates: API key/ratelimit middleware; production CORS tightening.
  6) Observability (later): Prometheus metrics; real BM25 vs Dense contribution breakdown.
  7) Lint/Type: third wave cleanup to get local hooks green without `--no-verify`.

## Immediate Next Steps (small, verifiable)
1) Lint/Type cleanup round 3
   - Scope: embeddings/rerank/routes minor leftovers, no behavior changes.
   - CI: install/setup bandit/safety or temporarily disable, then re-enable later.
2) TruLens end-to-end
   - Provider: OpenAI-compatible (`OPENAI_BASE_URL`, `OPENAI_API_KEY`).
   - Wire TruLens App around pipeline; write run records; expose in `/eval/run` & `/eval/runs`.
   - Update docs with quick start for local proxy (LiteLLM→Ollama) and recommended params.
3) Weaviate contract → CI
   - Add job that runs only if `WEAVIATE_URL` set, or start profile in CI runner.

## Quick Reference (env)
- Backend
  - `vector_backend: milvus|weaviate`
  - `database.weaviate.url: http://localhost:8081`
- Generation
  - `ASKME_ENABLE_OLLAMA=1` or `generation.provider: ollama`
  - `generation.provider: openai`, `OPENAI_BASE_URL`, `OPENAI_API_KEY`
- Evaluation
  - TruLens on/off: `evaluation.trulens.enabled: true|false`
  - Ragas on/off: `evaluation.ragas.enabled: true|false`

## Files Touched (summary)
- API/services
  - `askme/api/main.py` — lifecycle init; generator selection.
  - `askme/api/routes/{ingest,query,evaluation}.py` — real ingestion + status; E2E evaluation; storage integration.
- Retrieval
  - `askme/retriever/{milvus_retriever.py,weaviate_retriever.py,base.py}` — hybrid+params.
- Generation
  - `askme/generation/generator.py` — Simple/Ollama/OpenAI-compatible.
- Evaluation
  - `askme/evals/{ragas_runner.py,trulens_runner.py,pipeline_runner.py,storage.py,evaluator.py}` — runners, E2E, persistence.
- Config
  - `configs/askme.yaml`, `askme/core/config.py` — generation+weaviate config fields。
- Scripts & Tests
  - `scripts/rerank.sh`（new）；
  - `tests/unit/test_generation.py`, `tests/unit/test_openai_generator.py`；
  - `tests/integration/test_weaviate_contract.py`。

## Tips to Resume
- Pick one backend (Weaviate recommended for quick start). Confirm health:
  - `curl http://localhost:8081/v1/.well-known/ready`
- Start API, then try quick ingest → query → evaluate path.
- For higher-fidelity evaluation, start an OpenAI-compatible proxy (LiteLLM) pointing to Ollama and set `OPENAI_BASE_URL`+`OPENAI_API_KEY` before running `/eval/run`.

---
This log is maintained to carry context across sessions. Add new notes here before ending a work cycle.
