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
import math
import os
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Any, Dict, Iterable, List, Optional, Sequence


def _as_float_list(value: Any) -> List[float]:
    """Best-effort conversion of a metric payload to a list of floats."""

    if value is None:
        return []

    if isinstance(value, (int, float)):
        v = float(value)
        return [v] if math.isfinite(v) else []

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
                return [
                    float(v)
                    for v in arr
                    if isinstance(v, (int, float)) and math.isfinite(float(v))
                ]
        except Exception:
            pass

    if hasattr(value, "mean") and callable(getattr(value, "mean")):
        try:
            mean_val = value.mean()
            if isinstance(mean_val, (int, float)):
                return [float(mean_val)] if math.isfinite(float(mean_val)) else []
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
        if isinstance(payload, list):
            collected: List[float] = []
            for item in payload:
                collected.extend(_collect_metric_values(item, aliases))
            if collected:
                return collected

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
    # Set environment variables to reduce warnings
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # Import uvloop compatibility utilities
    from askme.evals.uvloop_compat import (
        patch_nest_asyncio_import,
        run_in_thread_with_new_loop,
    )

    try:
        # Pre-patch nest_asyncio to handle uvloop
        patch_nest_asyncio_import()

        # Import ragas components in uvloop-compatible way
        def import_ragas() -> Dict[str, Any]:
            print("DEBUG: Attempting to import datasets...")
            from datasets import Dataset

            print("DEBUG: datasets import successful")

            print("DEBUG: Attempting to import ragas...")
            import ragas

            version = getattr(ragas, "__version__", "unknown")
            print(f"DEBUG: ragas version: {version}")
            from ragas import evaluate

            print("DEBUG: ragas.evaluate import successful")

            print("DEBUG: Attempting to import ragas metrics...")
            from ragas.metrics import (
                answer_relevancy,
                context_precision,
                context_recall,
                faithfulness,
            )

            print("DEBUG: All ragas metrics imported successfully")

            return {
                "Dataset": Dataset,
                "evaluate": evaluate,
                "metrics": {
                    "answer_relevancy": answer_relevancy,
                    "context_precision": context_precision,
                    "context_recall": context_recall,
                    "faithfulness": faithfulness,
                },
            }

        try:
            # Try direct import first (works in most environments)
            ragas_modules = import_ragas()
            print("DEBUG: Direct ragas import successful")
        except Exception as e:
            print(f"DEBUG: Direct import failed: {e}, trying thread isolation")
            # If direct import fails due to uvloop, use thread isolation
            ragas_modules = run_in_thread_with_new_loop(import_ragas)
            print("DEBUG: Thread-isolated ragas import successful")

        # Extract imported modules
        Dataset = ragas_modules["Dataset"]
        evaluate = ragas_modules["evaluate"]
        answer_relevancy = ragas_modules["metrics"]["answer_relevancy"]
        context_precision = ragas_modules["metrics"]["context_precision"]
        context_recall = ragas_modules["metrics"]["context_recall"]
        faithfulness = ragas_modules["metrics"]["faithfulness"]

    except Exception as e:
        print(f"DEBUG: All ragas import strategies failed: {e}")
        import traceback

        print(f"DEBUG: Final traceback: {traceback.format_exc()}")
        raise RuntimeError(f"Ragas initialization failed: {e}") from e

    # Configure local provider: prioritize local embedding + Ollama, fallback as needed
    llm = None
    emb = None
    try:
        from ragas.embeddings import HuggingFaceEmbeddings

        model_name = os.getenv("ASKME_RAGAS_EMBED_MODEL", "Qwen/Qwen3-Embedding-8B")
        emb = HuggingFaceEmbeddings(
            model=model_name,
            use_api=False,
            device="cpu",
            normalize_embeddings=True,
            batch_size=32,
        )
        print(f"DEBUG: Ragas embeddings configured with model={model_name}")
    except Exception as e:
        print(f"DEBUG: Failed to configure Ragas embeddings: {e}")
        emb = None

    llm_settings = {
        "model": os.getenv("ASKME_RAGAS_LLM_MODEL", "gpt-oss:20b"),
        "base_url": os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1"),
        "api_key": os.getenv("OPENAI_API_KEY", "ollama-local"),
    }

    try:
        from askme.evals.ragas_llm import build_ollama_llm

        llm = build_ollama_llm(llm_settings)
        print(
            "DEBUG: Ragas LLM configured with model=%s, base_url=%s"
            % (llm_settings["model"], llm_settings["base_url"])
        )
    except Exception as e:
        print(f"DEBUG: Failed to configure Ragas LLM: {e}")
        import traceback

        print(f"DEBUG: Full traceback: {traceback.format_exc()}")
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
    # Ragas>=0.2 some metrics (like context_precision) expect column name `reference`
    # Use first ground_truth as reference (if exists)
    ref = [(g[0] if isinstance(g, list) and g else "") for g in gt]

    # Prefer passing explicit providers when available
    def _evaluate_dataset(ds: Any) -> Any:
        print(
            "DEBUG: Running evaluation with LLM=%s, EMB=%s"
            % (llm is not None, emb is not None)
        )
        if llm is not None and emb is not None:
            print("DEBUG: Using custom LLM and embeddings")
            return evaluate(ds, metrics=metric_objs, llm=llm, embeddings=emb)
        if llm is not None:
            print("DEBUG: Using custom LLM only")
            return evaluate(ds, metrics=metric_objs, llm=llm)
        if emb is not None:
            print("DEBUG: Using custom embeddings only")
            return evaluate(ds, metrics=metric_objs, embeddings=emb)
        print("DEBUG: Using default providers")
        return evaluate(ds, metrics=metric_objs)

    per_metric_values: Dict[str, List[float]] = {m: [] for m in selected_metrics}

    for idx in range(len(samples)):
        single_ds = Dataset.from_dict(
            {
                "question": [q[idx]],
                "answer": [a[idx]],
                "contexts": [ctx[idx]],
                "ground_truths": [gt[idx]],
                "reference": [ref[idx]],
            }
        )

        try:
            result = _evaluate_dataset(single_ds)
        except Exception as e:
            print(f"DEBUG: Ragas per-sample evaluation failed at index {idx}: {e}")
            continue

        for metric_name in selected_metrics:
            aliases: List[str] = [metric_name]
            if metric_name == "answer_relevance":
                aliases.append("answer_relevancy")
            elif metric_name == "answer_relevancy":
                aliases.append("answer_relevance")

            values = _collect_metric_values(result, aliases)
            if values:
                per_metric_values[metric_name].extend(values)

    scores: Dict[str, float] = {}
    for metric_name in selected_metrics:
        values = per_metric_values.get(metric_name, [])
        if values:
            scores[metric_name] = sum(values) / len(values)
        else:
            scores[metric_name] = 0.0

    return scores
