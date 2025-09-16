"""
Ragas evaluation runner. Attempts to evaluate standard metrics using ragas v0.2+.

Requirements:
- ragas>=0.2.0
- datasets
- A compatible LLM provider for metrics that require one (e.g., OpenAI), or
  restrict to embedding-based metrics if no LLM is configured.
"""

from __future__ import annotations

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence


def _as_float_list(value: Any) -> List[float]:
    """Best-effort conversion of a metric payload to a list of floats."""

    if value is None:
        return []

    if isinstance(value, (int, float)):
        return [float(value)]

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        out: List[float] = []
        for v in value:
            if isinstance(v, (int, float)):
                out.append(float(v))
        if out:
            return out

    # pandas Series / numpy array like objects expose mean/tolist/values
    if hasattr(value, "tolist"):
        try:
            arr = value.tolist()
            if isinstance(arr, Iterable) and not isinstance(
                arr, (str, bytes, bytearray)
            ):
                return [float(v) for v in arr if isinstance(v, (int, float))]
        except Exception:
            pass

    if hasattr(value, "mean") and callable(getattr(value, "mean")):
        try:
            mean_val = value.mean()
            if isinstance(mean_val, (int, float)):
                return [float(mean_val)]
        except Exception:
            pass

    return []


def _extract_from_mapping(
    mapping: Dict[str, Any], aliases: Sequence[str]
) -> List[float]:
    for alias in aliases:
        if alias in mapping:
            converted = _as_float_list(mapping[alias])
            if converted:
                return converted
    return []


def _collect_metric_values(result: Any, aliases: Sequence[str]) -> List[float]:
    """Collect metric values from ragas evaluate output supporting multiple layouts."""

    # Direct dict payload
    if isinstance(result, dict):
        values = _extract_from_mapping(result, aliases)
        if values:
            return values

    # Result objects may expose helpful helpers
    for attr in ("scores", "metrics", "results"):
        payload = getattr(result, attr, None)
        if isinstance(payload, dict):
            values = _extract_from_mapping(payload, aliases)
            if values:
                return values

    # DataFrame-like objects
    columns = getattr(result, "columns", None)
    if isinstance(columns, Iterable):
        try:
            column_names = list(columns)
        except Exception:
            column_names = []
        for alias in aliases:
            if alias in column_names:
                try:
                    col_obj = result[alias]
                    values = _as_float_list(col_obj)
                    if values:
                        return values
                except Exception:
                    continue

    # Converters
    for method in ("to_pandas", "to_dataframe", "to_dict"):
        fn = getattr(result, method, None)
        if callable(fn):
            try:
                converted = fn()
            except Exception:
                continue

            if isinstance(converted, dict):
                values = _extract_from_mapping(converted, aliases)
                if values:
                    return values

            columns = getattr(converted, "columns", None)
            if isinstance(columns, Iterable):
                try:
                    column_names = list(columns)
                except Exception:
                    column_names = []
                for alias in aliases:
                    if alias in column_names:
                        try:
                            col_obj = converted[alias]
                            values = _as_float_list(col_obj)
                            if values:
                                return values
                        except Exception:
                            continue

    # Attribute direct access last
    for alias in aliases:
        if hasattr(result, alias):
            values = _as_float_list(getattr(result, alias))
            if values:
                return values

    return []


def run_ragas(
    samples: List[Dict[str, Any]],
    metrics: Optional[List[str]] = None,
) -> Dict[str, float]:
    restore_policy: Optional[asyncio.AbstractEventLoopPolicy] = None
    try:
        try:
            policy = asyncio.get_event_loop_policy()
            if type(policy).__module__.startswith("uvloop"):
                restore_policy = policy
                asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
        except Exception:
            restore_policy = None

        try:
            try:
                import nest_asyncio as nest_asyncio_module

                _orig_apply = getattr(nest_asyncio_module, "apply", None)

                if _orig_apply is not None:
                    original_apply: Callable[..., Any] = _orig_apply

                    def _safe_apply(*args: Any, **kwargs: Any) -> None:
                        try:
                            original_apply(*args, **kwargs)
                        except ValueError as exc:
                            if "Can't patch loop" not in str(exc):
                                raise

                    setattr(nest_asyncio_module, "apply", _safe_apply)
            except Exception:
                pass

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

        # Configure local providers (Ollama, HF BGE-M3) when available
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
            os.environ.setdefault("OPENAI_API_BASE", base_url)
            os.environ.setdefault("OPENAI_API_KEY", api_key)
            model = os.getenv("ASKME_RAGAS_LLM_MODEL", "gpt-oss:20b")
            client = OpenAI(base_url=base_url, api_key=api_key)
            llm = RagasOpenAI(client=client, model=model)
        except Exception:
            llm = None

        # Default metrics
        from typing import Any as AnyMetric

        metric_objs: List[AnyMetric] = []
        selected_metrics = metrics or [
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
        ]
        for m in selected_metrics:
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
        def _run_evaluation() -> Any:
            if llm is not None or emb is not None:
                return evaluate(data, metrics=metric_objs, llm=llm, embeddings=emb)
            return evaluate(data, metrics=metric_objs)

        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                result = pool.submit(_run_evaluation).result()
        except Exception:
            result = _run_evaluation()
        scores: Dict[str, float] = {}
        for metric_name in selected_metrics:
            aliases: List[str] = [metric_name]
            if metric_name == "answer_relevance":
                aliases.append("answer_relevancy")
            elif metric_name == "answer_relevancy":
                aliases.append("answer_relevance")

            values = _collect_metric_values(result, aliases)
            if values:
                scores[metric_name] = sum(values) / len(values)
            else:
                scores[metric_name] = 0.0
        return scores
    finally:
        if restore_policy is not None:
            try:
                asyncio.set_event_loop_policy(restore_policy)
            except Exception:
                pass
