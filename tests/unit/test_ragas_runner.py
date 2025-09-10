"""
Unit tests to raise coverage for askme.evals.ragas_runner.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from askme.evals.ragas_runner import run_ragas


@pytest.mark.parametrize("modname", ["datasets", "ragas"])
def test_ragas_import_error_raises_runtime_error(monkeypatch, modname: str) -> None:
    import builtins

    real_import = builtins.__import__

    def _imp(name: str, *args: Any, **kwargs: Any):  # type: ignore[no-redef]
        if name.startswith(modname):
            raise ImportError(f"No module named {modname}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _imp)
    with pytest.raises(RuntimeError, match="Ragas not available"):
        run_ragas([])


def test_ragas_success_with_fallback_and_alias(monkeypatch) -> None:
    # Build stub modules for datasets + ragas
    class _Dataset:
        @staticmethod
        def from_dict(d: Dict[str, Any]) -> "_DS":  # type: ignore[override]
            return _DS(d)

    class _DS:
        def __init__(self, data: Dict[str, Any]):
            self.data = data

    eval_calls = {"count": 0}

    class _Res:
        def __init__(self) -> None:
            # Provide columns and __getitem__ accessor
            self.columns = [
                "faithfulness",
                "answer_relevancy",
                "context_precision",
                "context_recall",
            ]

        def __getitem__(self, key: str):
            class _Col:
                def mean(self_nonlocal):  # type: ignore[no-redef]
                    return 0.8

            return _Col()

    def _evaluate(dataset: Any, metrics: List[Any], llm=None, embeddings=None):  # type: ignore[no-redef]
        # First call with provider present should raise to trigger fallback
        eval_calls["count"] += 1
        if eval_calls["count"] == 1 and (llm is not None or embeddings is not None):
            raise RuntimeError("provider error")
        return _Res()

    class _HFEmb:
        def __init__(self, *args: Any, **kwargs: Any) -> None:  # type: ignore[no-redef]
            pass

    class _OpenAI:
        def __init__(self, *args: Any, **kwargs: Any) -> None:  # type: ignore[no-redef]
            pass

    # Wire stub modules
    import sys

    sys.modules["datasets"] = SimpleNamespace(Dataset=_Dataset)  # type: ignore[attr-defined]
    sys.modules["ragas"] = SimpleNamespace(evaluate=_evaluate)
    sys.modules["ragas.metrics"] = SimpleNamespace(
        answer_relevancy=object(),
        context_precision=object(),
        context_recall=object(),
        faithfulness=object(),
    )
    sys.modules["ragas.embeddings"] = SimpleNamespace(HuggingFaceEmbeddings=_HFEmb)
    sys.modules["ragas.llms"] = SimpleNamespace(OpenAI=_OpenAI)

    samples = [{"question": "q", "answer": "a", "contexts": ["c"], "ground_truths": ["g"]}]
    # include alias 'answer_relevance' to test mapping
    scores = run_ragas(samples, metrics=["faithfulness", "answer_relevance", "context_precision", "context_recall"])
    assert set(scores.keys()) == {"faithfulness", "answer_relevance", "context_precision", "context_recall"}
    # All metric means resolved via .mean() -> 0.8
    assert all(abs(v - 0.8) < 1e-6 for v in scores.values())

