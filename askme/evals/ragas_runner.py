"""
Ragas evaluation runner. Attempts to evaluate standard metrics using ragas v0.2+.

Requirements:
- ragas>=0.2.0
- datasets
- A compatible LLM provider for metrics that require one (e.g., OpenAI), or
  restrict to embedding-based metrics if no LLM is configured.
"""

from __future__ import annotations

import os
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

        # Try to configure local providers explicitly (Ollama/OpenAI-compatible + HF BGE-M3)
        llm = None
        emb = None
        try:
            # Prefer HuggingFace embeddings for BGE-M3 (local, no network)
            from ragas.embeddings import HuggingFaceEmbeddings

            model_name = os.getenv("ASKME_RAGAS_EMBED_MODEL", "BAAI/bge-m3")
            emb = HuggingFaceEmbeddings(model_name=model_name)
        except Exception:
            emb = None
        try:
            # Use OpenAI-compatible client pointed to Ollama for LLM metrics
            from openai import OpenAI
            from ragas.llms import OpenAI as RagasOpenAI

            base_url = os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
            api_key = os.getenv("OPENAI_API_KEY", "ollama-local")
            model = os.getenv("ASKME_RAGAS_LLM_MODEL", "gpt-oss:20b")
            client = OpenAI(base_url=base_url, api_key=api_key)
            llm = RagasOpenAI(client=client, model=model)
        except Exception:
            llm = None
    except Exception as e:
        raise RuntimeError(
            "Ragas not available. Install ragas>=0.2.0 and datasets."
        ) from e

    # Default metrics
    from typing import Any as AnyMetric

    metric_objs: List[AnyMetric] = []
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
    # Ragas>=0.2 有些指标（如 context_precision）期望列名 `reference`
    # 这里用首个 ground_truth 作为 reference（若存在）
    ref = [(g[0] if isinstance(g, list) and g else "") for g in gt]
    data = Dataset.from_dict(
        {
            "question": q,
            "answer": a,
            "contexts": ctx,
            "ground_truths": gt,
            "reference": ref,
        }
    )

    # Prefer passing explicit providers when available
    try:
        if llm is not None or emb is not None:
            result = evaluate(data, metrics=metric_objs, llm=llm, embeddings=emb)
        else:
            result = evaluate(data, metrics=metric_objs)
    except Exception:
        # Last-resort fallback: try without providers
        result = evaluate(data, metrics=metric_objs)
    # result is a pandas dataframe-like with column names as metrics
    scores: Dict[str, float] = {}
    for m in metrics:
        col = (
            m
            if hasattr(result, "columns") and m in result.columns
            else ("answer_relevancy" if m == "answer_relevance" else m)
        )
        if hasattr(result, "columns") and col in result.columns:
            try:
                col_data = result[col]
                if hasattr(col_data, "mean"):
                    scores[m] = float(col_data.mean())
                elif isinstance(col_data, list):
                    scores[m] = (
                        float(sum(col_data) / len(col_data)) if col_data else 0.0
                    )
                else:
                    scores[m] = float(col_data)
            except Exception:
                scores[m] = 0.0
        else:
            scores[m] = 0.0
    return scores
