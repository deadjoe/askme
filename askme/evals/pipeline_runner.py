"""
Minimal in-process pipeline runner to execute the RAG flow once
using application services (embedding -> hybrid retrieve -> rerank -> generate).

Used by evaluation to produce answer/context dynamically for end-to-end scoring.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from askme.core.config import Settings
from askme.enhancer.query_enhancer import generate_fusion_queries, generate_hyde_passage
from askme.generation.generator import Passage
from askme.retriever.base import HybridSearchParams, SearchFusion


@dataclass
class PipelineResult:
    question: str
    answer: str
    contexts: List[str]
    citations: List[Dict[str, Any]]


async def run_pipeline_once(
    app: Any, question: str, settings: Settings
) -> PipelineResult:
    """Run a single query through the in-process pipeline.

    Falls back to a minimal template answer if any service is missing.
    """
    embedding_service = getattr(app.state, "embedding_service", None)
    retriever = getattr(app.state, "retriever", None)
    reranking_service = getattr(app.state, "reranking_service", None)
    generator = getattr(app.state, "generator", None)

    if not (embedding_service and retriever and reranking_service and generator):
        # Minimal fallback
        return PipelineResult(
            question=question,
            answer=f"No services available to answer: {question}",
            contexts=[],
            citations=[],
        )

    # Build query list with optional enhancements (respect config defaults)
    query_list: List[str] = [question]
    if settings.enhancer.hyde.enabled:
        query_list.append(generate_hyde_passage(question))
    if settings.enhancer.rag_fusion.enabled:
        query_list.extend(
            [
                q
                for q in generate_fusion_queries(
                    question, settings.enhancer.rag_fusion.num_queries
                )
                if q not in query_list
            ]
        )

    params = HybridSearchParams(
        alpha=settings.hybrid.alpha,
        use_rrf=(settings.hybrid.mode == "rrf" or settings.hybrid.use_rrf),
        rrf_k=settings.hybrid.rrf_k,
        topk=settings.hybrid.topk,
        filters=None,
        original_query=question,
    )

    result_lists = []
    for q in query_list:
        try:
            # Add timeout protection for embedding service
            q_emb = await asyncio.wait_for(
                embedding_service.encode_query(q), timeout=30.0
            )
            # Add timeout protection for retrieval
            res = await asyncio.wait_for(
                retriever.hybrid_search(
                    q_emb["dense_embedding"], q_emb["sparse_embedding"], params
                ),
                timeout=60.0,
            )
            result_lists.append(res)
        except asyncio.TimeoutError:
            print(f"DEBUG: Timeout during query processing for: {q}")
            # Return empty result for this query to prevent complete failure
            result_lists.append([])

    results = (
        SearchFusion.reciprocal_rank_fusion(result_lists, k=params.rrf_k)
        if len(result_lists) > 1
        else result_lists[0]
    )

    # Add timeout protection for reranking
    try:
        reranked = await asyncio.wait_for(
            reranking_service.rerank(
                question, results, top_n=settings.rerank.top_n, prefer_local=True
            ),
            timeout=30.0,
        )
    except asyncio.TimeoutError:
        print(f"DEBUG: Timeout during reranking for question: {question}")
        # Use original results without reranking as fallback
        reranked = results[: settings.rerank.top_n]

    # Build passages and citations
    citations: List[Dict[str, Any]] = []
    passages: List[Passage] = []
    contexts: List[str] = []

    for r in reranked:
        title = r.document.metadata.get("title", r.document.id)
        snippet = r.document.content
        citations.append(
            {
                "doc_id": r.document.id,
                "title": title,
                "score": float(r.rerank_score),
                "metadata": r.document.metadata,
            }
        )
        contexts.append(snippet)
        passages.append(
            Passage(
                doc_id=r.document.id,
                title=title,
                content=snippet,
                score=float(r.rerank_score),
            )
        )

    # Add timeout protection for LLM generation (most critical blocking point)
    try:
        answer = await asyncio.wait_for(
            generator.generate(question, passages), timeout=120.0
        )
    except asyncio.TimeoutError:
        print(f"DEBUG: Timeout during LLM generation for question: {question}")
        # Provide fallback answer to prevent complete pipeline failure
        answer = f"[TIMEOUT] Unable to generate answer for: {question}"
    except Exception as e:
        print(f"DEBUG: Generation failed with error: {e}")
        answer = f"[ERROR] Generation failed for: {question}"

    return PipelineResult(
        question=question,
        answer=answer,
        contexts=contexts,
        citations=citations,
    )
