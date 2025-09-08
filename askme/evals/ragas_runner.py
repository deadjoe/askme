"""
Ragas evaluation runner. Attempts to evaluate standard metrics using ragas v0.2+.

Requirements:
- ragas>=0.2.0
- datasets
- A compatible LLM provider for metrics that require one (e.g., OpenAI), or
  restrict to embedding-based metrics if no LLM is configured.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def run_ragas(
    samples: List[Dict[str, Any]],
    metrics: Optional[List[str]] = None,
) -> Dict[str, float]:
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except Exception as e:
        raise RuntimeError(
            "Ragas not available. Install ragas>=0.2.0 and datasets."
        ) from e

    # Default metrics
    metric_objs = []
    metrics = metrics or [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ]
    for m in metrics:
        if m == "faithfulness":
            metric_objs.append(faithfulness)
        elif m in ("answer_relevancy", "answer_relevance"):
            metric_objs.append(answer_relevancy)
        elif m == "context_precision":
            metric_objs.append(context_precision)
        elif m == "context_recall":
            metric_objs.append(context_recall)

    # Ragas expects a dataset with these columns
    q = [s["question"] for s in samples]
    a = [s.get("answer", "") for s in samples]
    ctx = [s.get("contexts", []) for s in samples]
    gt = [s.get("ground_truths", []) for s in samples]
    data = Dataset.from_dict(
        {
            "question": q,
            "answer": a,
            "contexts": ctx,
            "ground_truths": gt,
        }
    )

    result = evaluate(data, metrics=metric_objs)
    # result is a pandas dataframe-like with column names as metrics
    scores: Dict[str, float] = {}
    for m in metrics:
        col = (
            m
            if m in result.columns
            else ("answer_relevancy" if m == "answer_relevance" else m)
        )
        if col in result.columns:
            try:
                scores[m] = float(result[col].mean())
            except Exception:
                pass
    return scores
