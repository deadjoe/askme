"""
Query and retrieval endpoints.
"""

import time
from datetime import datetime
from typing import Any, Dict, List, Optional, cast

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from askme.core.config import Settings, get_settings
from askme.enhancer.query_enhancer import generate_fusion_queries, generate_hyde_passage
from askme.retriever.base import HybridSearchParams, SearchFusion


class Citation(BaseModel):
    """Document citation model."""

    doc_id: str = Field(..., description="Document identifier")
    title: str = Field(..., description="Document title")
    content: str = Field(..., description="Relevant content snippet")
    start: int = Field(..., description="Start position in document")
    end: int = Field(..., description="End position in document")
    score: float = Field(..., ge=0, le=1, description="Relevance score")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Document metadata"
    )


class RetrievalDebugInfo(BaseModel):
    """Debug information for retrieval process."""

    bm25_hits: int = Field(..., description="Number of BM25 matches")
    dense_hits: int = Field(..., description="Number of dense vector matches")
    fusion_method: str = Field(..., description="Fusion method used")
    alpha: Optional[float] = Field(
        default=None, description="Alpha parameter for fusion"
    )
    rrf_k: Optional[int] = Field(default=None, description="RRF k parameter")
    rerank_model: str = Field(..., description="Reranker model used")
    rerank_scores: Optional[List[float]] = Field(
        default=None, description="Reranking scores"
    )
    latency_ms: int = Field(..., description="Total retrieval latency in milliseconds")
    embedding_latency_ms: int = Field(
        ..., description="Embedding latency in milliseconds"
    )
    search_latency_ms: int = Field(..., description="Search latency in milliseconds")
    rerank_latency_ms: int = Field(..., description="Reranking latency in milliseconds")
    overlap_hits: Optional[int] = Field(
        default=None, description="Overlap between dense-only and BM25-only topK"
    )
    # Optional error when falling back to mock path
    error: Optional[str] = Field(default=None, description="Fallback error detail")


class QueryRequest(BaseModel):
    """Query request model."""

    q: str = Field(..., description="Search query", min_length=1)
    topk: int = Field(
        default=50,
        ge=1,
        le=100,
        description="Number of candidates to retrieve",
        alias="topK",
    )
    alpha: float = Field(
        default=0.5, ge=0, le=1, description="Hybrid search alpha parameter"
    )
    use_rrf: bool = Field(default=True, description="Use reciprocal rank fusion")
    rrf_k: int = Field(default=60, ge=1, le=200, description="RRF k parameter")
    use_hyde: bool = Field(default=False, description="Use HyDE query expansion")
    use_rag_fusion: bool = Field(
        default=False, description="Use RAG-Fusion multi-query"
    )
    reranker: str = Field(
        default="bge_local", description="Reranker model: bge_local, cohere"
    )
    max_passages: int = Field(
        default=8, ge=1, le=20, description="Maximum passages for generation"
    )
    filters: Optional[Dict[str, Any]] = Field(
        default=None, description="Metadata filters"
    )
    include_debug: bool = Field(default=False, description="Include debug information")


class QueryResponse(BaseModel):
    """Query response model."""

    answer: str = Field(..., description="Generated answer with citations")
    citations: List[Citation] = Field(..., description="Source citations")
    query_id: str = Field(..., description="Unique query identifier")
    timestamp: datetime = Field(..., description="Query timestamp")
    retrieval_debug: Optional[RetrievalDebugInfo] = Field(
        default=None, description="Debug info"
    )


class RetrievalRequest(BaseModel):
    """Retrieval-only request model."""

    q: str = Field(..., description="Search query", min_length=1)
    topk: int = Field(
        default=50,
        ge=1,
        le=100,
        description="Number of candidates to retrieve",
        alias="topK",
    )
    use_hyde: bool = Field(default=False, description="Use HyDE query expansion")
    use_rag_fusion: bool = Field(
        default=False, description="Use RAG-Fusion multi-query"
    )
    alpha: float = Field(
        default=0.5, ge=0, le=1, description="Hybrid search alpha parameter"
    )
    use_rrf: bool = Field(default=True, description="Use reciprocal rank fusion")
    rrf_k: int = Field(default=60, ge=1, le=200, description="RRF k parameter")
    filters: Optional[Dict[str, Any]] = Field(
        default=None, description="Metadata filters"
    )


class RetrievalResponse(BaseModel):
    """Retrieval-only response model."""

    documents: List[Citation] = Field(..., description="Retrieved documents")
    query_id: str = Field(..., description="Unique query identifier")
    timestamp: datetime = Field(..., description="Query timestamp")
    retrieval_debug: RetrievalDebugInfo = Field(..., description="Debug information")


router = APIRouter()


@router.post("/", response_model=QueryResponse)
async def query_documents(
    req: QueryRequest,
    request: Request,
    settings: Settings = Depends(get_settings),
) -> QueryResponse:
    """
    Query documents using hybrid search with optional enhancements.

    Process:
    1. Optional query enhancement (HyDE, RAG-Fusion)
    2. Hybrid retrieval (dense + sparse)
    3. Reranking
    4. LLM generation with citations
    """

    import uuid

    # Validate reranker selection
    if req.reranker == "cohere" and not settings.rerank.cohere_enabled:
        raise HTTPException(
            status_code=400,
            detail="Cohere reranker not enabled. Set ASKME_ENABLE_COHERE=1",
        )

    # Try real pipeline using app services if available, else fall back to mock
    try:
        app = request.app
        embedding_service = getattr(app.state, "embedding_service", None)
        retriever = getattr(app.state, "retriever", None)
        reranking_service = getattr(app.state, "reranking_service", None)

        if not embedding_service or not retriever or not reranking_service:
            raise RuntimeError("Services not initialized")

        t0 = time.monotonic()
        stage = "init"
        # 1) (Optional) enhancement
        query_list: List[str] = [req.q]
        if req.use_hyde:
            hyde_text = generate_hyde_passage(req.q)
            query_list.append(hyde_text)
        if req.use_rag_fusion:
            fusion = generate_fusion_queries(req.q, num_queries=3)
            # Include generated variants; original already present
            query_list.extend([q for q in fusion if q not in query_list])

        # 2) Encode + retrieve
        t1 = time.monotonic()
        params = HybridSearchParams(
            alpha=req.alpha,
            use_rrf=req.use_rrf,
            rrf_k=req.rrf_k,
            topk=req.topk,
            filters=req.filters,
            original_query=req.q,
        )
        result_lists = []
        try:
            stage = "embedding"
            for q in query_list:
                q_emb = await embedding_service.encode_query(q)
                stage = "retrieval"
                res = await retriever.hybrid_search(
                    q_emb["dense_embedding"], q_emb["sparse_embedding"], params
                )
                result_lists.append(res)
        except Exception as _e:
            # Raise with stage info for outer handler
            raise RuntimeError(f"{stage} failed: {_e}")
        # Fuse multiple result lists (if any) with RRF; otherwise use single
        results = (
            SearchFusion.reciprocal_rank_fusion(result_lists, k=req.rrf_k)
            if len(result_lists) > 1
            else result_lists[0]
        )
        t2 = time.monotonic()
        # 4) Rerank
        from askme.rerank.rerank_service import RerankingService

        rerank_svc = cast(RerankingService, reranking_service)
        try:
            stage = "rerank"
            reranked = await rerank_svc.rerank(
                req.q, results, top_n=req.max_passages, prefer_local=True
            )
        except Exception as _e:
            raise RuntimeError(f"{stage} failed: {_e}")
        t3 = time.monotonic()

        # Build citations
        citations: List[Citation] = []
        for r in reranked:
            citations.append(
                Citation(
                    doc_id=r.document.id,
                    title=r.document.metadata.get("title", r.document.id),
                    content=r.document.content[:200],
                    start=0,
                    end=min(200, len(r.document.content)),
                    score=max(0.0, min(1.0, float(r.rerank_score))),
                    metadata=r.document.metadata,
                )
            )

        # 5) Generate answer via configured generator
        from askme.generation.generator import Passage  # local import to avoid cycles

        generator = getattr(app.state, "generator", None)
        passages = [
            Passage(doc_id=c.doc_id, title=c.title, content=c.content, score=c.score)
            for c in citations
        ]
        if generator is None:
            # Fallback minimal template
            titles = ", ".join([c.title for c in citations]) or "context"
            answer = (
                f"Based on retrieved context, here is an answer about '{req.q}'. "
                f"Sources: {titles}."
            )
        else:
            try:
                stage = "generation"
                answer = await generator.generate(req.q, passages)
            except Exception as _e:
                raise RuntimeError(f"{stage} failed: {_e}")

        debug_info = None
        if req.include_debug:
            # 真实统计 dense-only 与 BM25-only 的 topK 命中（以原始 query 近似）
            bm25_hits = 0
            dense_hits = 0
            overlap_hits = 0
            try:
                q0_emb = await embedding_service.encode_query(req.q)
                dense_only = await retriever.dense_search(
                    q0_emb["dense_embedding"], topk=req.topk, filters=req.filters
                )
                sparse_only = await retriever.sparse_search(
                    q0_emb["sparse_embedding"], topk=req.topk, filters=req.filters
                )
                if not sparse_only:
                    # Weaviate 走 alpha 极值近似 BM25-only / dense-only
                    w_params_dense = HybridSearchParams(
                        alpha=1.0,
                        use_rrf=False,
                        rrf_k=req.rrf_k,
                        topk=req.topk,
                        filters=req.filters,
                        original_query=req.q,
                    )
                    w_params_sparse = HybridSearchParams(
                        alpha=0.0,
                        use_rrf=False,
                        rrf_k=req.rrf_k,
                        topk=req.topk,
                        filters=req.filters,
                        original_query=req.q,
                    )
                    dense_only = await retriever.hybrid_search(
                        q0_emb["dense_embedding"],
                        q0_emb["sparse_embedding"],
                        w_params_dense,
                    )
                    sparse_only = await retriever.hybrid_search(
                        q0_emb["dense_embedding"],
                        q0_emb["sparse_embedding"],
                        w_params_sparse,
                    )
                dense_ids = {r.document.id for r in dense_only}
                sparse_ids = {r.document.id for r in sparse_only}
                bm25_hits = len(sparse_ids)
                dense_hits = len(dense_ids)
                overlap_hits = len(dense_ids & sparse_ids)
            except Exception:
                pass

            debug_info = RetrievalDebugInfo(
                bm25_hits=bm25_hits or len(results),
                dense_hits=dense_hits or len(results),
                fusion_method="rrf" if req.use_rrf else "alpha",
                alpha=req.alpha,
                rrf_k=req.rrf_k if req.use_rrf else None,
                rerank_model=req.reranker,
                rerank_scores=[float(r.rerank_score) for r in reranked],
                latency_ms=int((t3 - t0) * 1000),
                embedding_latency_ms=int((t1 - t0) * 1000),
                search_latency_ms=int((t2 - t1) * 1000),
                rerank_latency_ms=int((t3 - t2) * 1000),
                overlap_hits=overlap_hits or None,
            )

        return QueryResponse(
            answer=answer,
            citations=citations,
            query_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow(),
            retrieval_debug=debug_info,
        )
    except Exception as e:
        # Fall through to mock response below; capture error detail for debug
        fallback_error = str(e)

    # Mock response for now
    mock_citations = [
        Citation(
            doc_id="doc_001",
            title="Sample Document 1",
            content="This is a sample content snippet that matches the query...",
            start=100,
            end=200,
            score=0.92,
            metadata={"author": "John Doe", "date": "2025-01-01"},
        ),
        Citation(
            doc_id="doc_002",
            title="Sample Document 2",
            content="Another relevant content piece with information about...",
            start=50,
            end=150,
            score=0.87,
            metadata={"author": "Jane Smith", "date": "2025-01-02"},
        ),
    ]

    debug_info = None
    if req.include_debug:
        debug_info = RetrievalDebugInfo(
            bm25_hits=25,
            dense_hits=30,
            fusion_method="rrf" if req.use_rrf else "alpha",
            alpha=req.alpha,
            rrf_k=req.rrf_k if req.use_rrf else None,
            rerank_model=req.reranker,
            rerank_scores=[0.92, 0.87, 0.79, 0.71],
            latency_ms=1250,
            embedding_latency_ms=150,
            search_latency_ms=800,
            rerank_latency_ms=300,
            error=locals().get("fallback_error"),
        )

    # 若真实流程失败，仍尽量调用已配置的 generator 生成答案，便于本地快速联调（如 Ollama ）
    try:
        if request is not None:
            app = request.app
            generator = getattr(app.state, "generator", None)
            if generator is not None:
                from askme.generation.generator import Passage

                passages = [
                    Passage(
                        doc_id=c.doc_id, title=c.title, content=c.content, score=c.score
                    )
                    for c in mock_citations
                ]
                answer = await generator.generate(req.q, passages)
            else:
                raise RuntimeError("generator unavailable")
        else:
            raise RuntimeError("no request context")
    except Exception:
        answer = (
            f"Based on the provided context, {req.q}... "
            f"[Doc 001: Sample Document 1] [Doc 002: Sample Document 2]"
        )

    return QueryResponse(
        answer=answer,
        citations=mock_citations,
        query_id=str(uuid.uuid4()),
        timestamp=datetime.utcnow(),
        retrieval_debug=debug_info,
    )


@router.post("/retrieve", response_model=RetrievalResponse)
async def retrieve_documents(
    req: RetrievalRequest,
    request: Request,
    settings: Settings = Depends(get_settings),
) -> RetrievalResponse:
    """
    Retrieve documents without LLM generation.

    Useful for:
    - Testing retrieval quality
    - Building custom generation pipelines
    - Debugging search parameters
    """

    import uuid

    # Try real retrieval if services exist
    try:
        app = request.app
        embedding_service = getattr(app.state, "embedding_service", None)
        retriever = getattr(app.state, "retriever", None)
        if not embedding_service or not retriever:
            raise RuntimeError("Services not initialized")

        t0 = time.monotonic()
        params = HybridSearchParams(
            alpha=req.alpha,
            use_rrf=req.use_rrf,
            rrf_k=req.rrf_k,
            topk=req.topk,
            filters=req.filters,
            original_query=req.q,
        )
        result_lists = []
        query_list: List[str] = [req.q]
        if req.use_hyde:
            query_list.append(generate_hyde_passage(req.q))
        if req.use_rag_fusion:
            query_list.extend(
                [q for q in generate_fusion_queries(req.q, 3) if q not in query_list]
            )
        for q in query_list:
            q_emb = await embedding_service.encode_query(q)
            res = await retriever.hybrid_search(
                q_emb["dense_embedding"], q_emb["sparse_embedding"], params
            )
            result_lists.append(res)
        results = (
            SearchFusion.reciprocal_rank_fusion(result_lists, k=req.rrf_k)
            if len(result_lists) > 1
            else result_lists[0]
        )
        t1 = time.monotonic()

        documents: List[Citation] = []
        for r in results[: req.topk]:
            documents.append(
                Citation(
                    doc_id=r.document.id,
                    title=r.document.metadata.get("title", r.document.id),
                    content=r.document.content[:200],
                    start=0,
                    end=min(200, len(r.document.content)),
                    score=max(0.0, min(1.0, float(r.score))),
                    metadata=r.document.metadata,
                )
            )

        bm25_hits = len(results)
        dense_hits = len(results)
        overlap_hits = None
        try:
            q0_emb = await embedding_service.encode_query(req.q)
            dense_only = await retriever.dense_search(
                q0_emb["dense_embedding"], topk=req.topk, filters=req.filters
            )
            sparse_only = await retriever.sparse_search(
                q0_emb["sparse_embedding"], topk=req.topk, filters=req.filters
            )
            if not sparse_only:
                w_params_dense = HybridSearchParams(
                    alpha=1.0,
                    use_rrf=False,
                    rrf_k=req.rrf_k,
                    topk=req.topk,
                    filters=req.filters,
                    original_query=req.q,
                )
                w_params_sparse = HybridSearchParams(
                    alpha=0.0,
                    use_rrf=False,
                    rrf_k=req.rrf_k,
                    topk=req.topk,
                    filters=req.filters,
                    original_query=req.q,
                )
                dense_only = await retriever.hybrid_search(
                    q0_emb["dense_embedding"],
                    q0_emb["sparse_embedding"],
                    w_params_dense,
                )
                sparse_only = await retriever.hybrid_search(
                    q0_emb["dense_embedding"],
                    q0_emb["sparse_embedding"],
                    w_params_sparse,
                )
            dense_ids = {r.document.id for r in dense_only}
            sparse_ids = {r.document.id for r in sparse_only}
            bm25_hits = len(sparse_ids)
            dense_hits = len(dense_ids)
            overlap_hits = len(dense_ids & sparse_ids)
        except Exception:
            pass

        debug_info = RetrievalDebugInfo(
            bm25_hits=bm25_hits,
            dense_hits=dense_hits,
            fusion_method="rrf" if req.use_rrf else "alpha",
            alpha=req.alpha,
            rrf_k=req.rrf_k if req.use_rrf else None,
            rerank_model="bge_local",
            latency_ms=int((t1 - t0) * 1000),
            embedding_latency_ms=0,
            search_latency_ms=int((t1 - t0) * 1000),
            rerank_latency_ms=0,
            overlap_hits=overlap_hits,
        )

        return RetrievalResponse(
            documents=documents,
            query_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow(),
            retrieval_debug=debug_info,
        )
    except Exception:
        pass

    mock_documents = [
        Citation(
            doc_id="doc_001",
            title="Retrieved Document 1",
            content="Content snippet from document 1...",
            start=0,
            end=100,
            score=0.95,
            metadata={"source": "manual", "tags": ["important"]},
        )
    ]

    debug_info = RetrievalDebugInfo(
        bm25_hits=20,
        dense_hits=35,
        fusion_method="rrf" if req.use_rrf else "alpha",
        alpha=req.alpha,
        # Always record provided rrf_k for transparency, even if RRF disabled
        rrf_k=req.rrf_k,
        rerank_model="bge_local",
        latency_ms=900,
        embedding_latency_ms=120,
        search_latency_ms=780,
        rerank_latency_ms=0,  # No reranking in retrieval-only
    )

    return RetrievalResponse(
        documents=mock_documents,
        query_id=str(uuid.uuid4()),
        timestamp=datetime.utcnow(),
        retrieval_debug=debug_info,
    )


@router.get("/similar/{doc_id}")
async def find_similar_documents(
    doc_id: str, limit: int = 10, settings: Settings = Depends(get_settings)
) -> List[Citation]:
    """
    Find documents similar to a given document.
    """

    # TODO: Implement similar document search

    return [
        Citation(
            doc_id="similar_001",
            title="Similar Document",
            content="Content similar to the target document...",
            start=0,
            end=50,
            score=0.89,
        )
    ]


@router.post("/explain")
async def explain_retrieval(
    request: RetrievalRequest, settings: Settings = Depends(get_settings)
) -> Dict[str, Any]:
    """
    Explain why specific documents were retrieved for a query.

    Useful for debugging and understanding retrieval behavior.
    """

    # TODO: Implement retrieval explanation

    return {
        "query": request.q,
        "explanation": "Mock explanation of retrieval process",
        "matching_terms": ["term1", "term2"],
        "semantic_similarity": 0.85,
        "ranking_factors": {"bm25_score": 0.7, "dense_score": 0.9, "final_score": 0.82},
    }
