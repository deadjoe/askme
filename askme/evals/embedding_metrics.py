"""Utility for computing RAG evaluation metrics using local embeddings."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
    runtime_checkable,
)

import numpy as np

from askme.evals.evaluator import EvalItem


@runtime_checkable
class EmbeddingService(Protocol):
    async def get_dense_embedding(self, text: str) -> Sequence[float]:
        ...


@dataclass
class EmbeddingScores:
    groundedness: float
    answer_relevance: float
    context_relevance: float
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    if not a.size or not b.size:
        return 0.0
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0.0:
        return 0.0
    return float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))


async def _embed_text(service: EmbeddingService, text: str) -> np.ndarray:
    vec = await service.get_dense_embedding(text)
    return np.asarray(vec, dtype=np.float32)


async def _embed_many(
    service: EmbeddingService, texts: Sequence[str]
) -> List[np.ndarray]:
    results: List[np.ndarray] = []
    for txt in texts:
        results.append(await _embed_text(service, txt))
    return results


async def _score_single(
    service: EmbeddingService,
    item: EvalItem,
    precision_cutoff: float,
    recall_cutoff: float,
) -> EmbeddingScores:
    question_vec = await _embed_text(service, item.query)
    answer_vec = await _embed_text(service, item.answer)
    contexts = item.contexts or []

    if contexts:
        context_vecs = await _embed_many(service, contexts)
    else:
        context_vecs = []

    answer_context_sims = [max(0.0, _cosine(answer_vec, c)) for c in context_vecs]
    question_context_sims = [max(0.0, _cosine(question_vec, c)) for c in context_vecs]

    groundedness = max(answer_context_sims) if answer_context_sims else 0.0
    context_relevance = (
        float(np.mean(question_context_sims)) if question_context_sims else 0.0
    )
    answer_relevance = max(0.0, _cosine(answer_vec, question_vec))

    precision_hits = (
        sum(1 for s in answer_context_sims if s >= precision_cutoff)
        if answer_context_sims
        else 0
    )
    recall_hits = (
        sum(1 for s in question_context_sims if s >= recall_cutoff)
        if question_context_sims
        else 0
    )

    context_precision = (
        precision_hits / max(1, len(context_vecs)) if context_vecs else 0.0
    )
    context_recall = recall_hits / max(1, len(context_vecs)) if context_vecs else 0.0

    return EmbeddingScores(
        groundedness=groundedness,
        answer_relevance=answer_relevance,
        context_relevance=context_relevance,
        faithfulness=groundedness,
        answer_relevancy=answer_relevance,
        context_precision=context_precision,
        context_recall=context_recall,
    )


async def compute_embedding_metrics(
    service: Optional[EmbeddingService],
    samples: Sequence[EvalItem],
    thresholds: Any,
    requested_metrics: Optional[Set[str]] = None,
) -> List[Dict[str, Any]]:
    if service is None:
        return []

    precision_cutoff = getattr(thresholds, "ragas_precision_min", 0.6) or 0.6
    recall_cutoff = getattr(thresholds, "ragas_precision_min", 0.6) or 0.6

    metrics_accumulators: Dict[str, List[float]] = {
        "groundedness": [],
        "answer_relevance": [],
        "context_relevance": [],
        "faithfulness": [],
        "answer_relevancy": [],
        "context_precision": [],
        "context_recall": [],
    }

    for sample in samples:
        scores = await _score_single(
            service,
            sample,
            precision_cutoff=precision_cutoff,
            recall_cutoff=recall_cutoff,
        )
        for key, value in scores.__dict__.items():
            metrics_accumulators[key].append(value)

    def _mean(name: str) -> float:
        vals = metrics_accumulators[name]
        if not vals:
            return 0.0
        return float(np.mean(vals))

    metric_map: Dict[str, Tuple[float, Optional[float]]] = {
        "groundedness": (
            _mean("groundedness"),
            getattr(thresholds, "trulens_min", None),
        ),
        "answer_relevance": (
            _mean("answer_relevance"),
            getattr(thresholds, "trulens_min", None),
        ),
        "context_relevance": (
            _mean("context_relevance"),
            getattr(thresholds, "trulens_min", None),
        ),
        "faithfulness": (
            _mean("faithfulness"),
            getattr(thresholds, "ragas_faithfulness_min", None),
        ),
        "answer_relevancy": (_mean("answer_relevancy"), None),
        "context_precision": (
            _mean("context_precision"),
            getattr(thresholds, "ragas_precision_min", None),
        ),
        "context_recall": (
            _mean("context_recall"),
            getattr(thresholds, "ragas_precision_min", None),
        ),
    }

    results: List[Dict[str, Any]] = []
    for name, (value, threshold) in metric_map.items():
        if requested_metrics is not None and name not in requested_metrics:
            continue
        passed = True
        if threshold is not None:
            passed = value >= threshold
        details = {
            "samples": len(samples),
            "method": "embedding_similarity",
            "precision_cutoff": precision_cutoff,
        }
        if name == "context_recall":
            details["recall_cutoff"] = recall_cutoff

        results.append(
            {
                "name": name,
                "value": float(max(0.0, min(1.0, value))),
                "threshold": threshold,
                "passed": passed,
                "details": details,
            }
        )

    return results
