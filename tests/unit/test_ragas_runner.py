"""Tests for askme.evals.ragas_runner."""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any, Dict, Generator, List

import pytest

from askme.evals.ragas_runner import run_ragas


class _FakeSeries(list):
    def mean(self) -> float:
        return sum(self) / len(self)


class _DataFrameLike:
    def __init__(self, columns: Dict[str, List[float]]) -> None:
        self.columns = list(columns.keys())
        self._cols = columns

    def __getitem__(self, key: str) -> Any:
        return _FakeSeries(self._cols[key])


class _DictLikeResult:
    def __init__(self, payload: Dict[str, Any]) -> None:
        self._payload = payload

    def to_dict(self) -> Dict[str, Any]:
        return self._payload


def _install_fake_ragas(monkeypatch: pytest.MonkeyPatch, evaluate_result: Any) -> None:
    datasets_module = ModuleType("datasets")

    class _Dataset:
        @classmethod
        def from_dict(cls, data: Dict[str, Any]) -> Dict[str, Any]:
            return data

    setattr(datasets_module, "Dataset", _Dataset)
    monkeypatch.setitem(sys.modules, "datasets", datasets_module)

    ragas_module = ModuleType("ragas")
    ragas_module.__path__ = []  # mark as package

    def _evaluate(*_args: Any, **_kwargs: Any) -> Any:
        return evaluate_result

    setattr(ragas_module, "evaluate", _evaluate)
    monkeypatch.setitem(sys.modules, "ragas", ragas_module)

    metrics_module = ModuleType("ragas.metrics")
    setattr(metrics_module, "answer_relevancy", object())
    setattr(metrics_module, "context_precision", object())
    setattr(metrics_module, "context_recall", object())
    setattr(metrics_module, "faithfulness", object())
    monkeypatch.setitem(sys.modules, "ragas.metrics", metrics_module)
    setattr(ragas_module, "metrics", metrics_module)

    embeddings_module = ModuleType("ragas.embeddings")

    class _HFEmb:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

    setattr(embeddings_module, "HuggingFaceEmbeddings", _HFEmb)
    monkeypatch.setitem(sys.modules, "ragas.embeddings", embeddings_module)
    setattr(ragas_module, "embeddings", embeddings_module)

    llms_module = ModuleType("ragas.llms")

    class _RagasOpenAI:
        def __init__(self, client: Any, model: str) -> None:
            self.client = client
            self.model = model

    setattr(llms_module, "OpenAI", _RagasOpenAI)
    monkeypatch.setitem(sys.modules, "ragas.llms", llms_module)
    setattr(ragas_module, "llms", llms_module)

    openai_module = ModuleType("openai")

    class _OpenAIClient:
        def __init__(self, base_url: str, api_key: str) -> None:
            self.base_url = base_url
            self.api_key = api_key

    setattr(openai_module, "OpenAI", _OpenAIClient)
    monkeypatch.setitem(sys.modules, "openai", openai_module)


@pytest.fixture(autouse=True)
def _cleanup_modules() -> Generator[None, None, None]:
    originals = sys.modules.copy()
    yield
    for name in list(sys.modules.keys()):
        if name not in originals:
            sys.modules.pop(name, None)


def test_run_ragas_handles_dataframe_like(monkeypatch: pytest.MonkeyPatch) -> None:
    df_result = _DataFrameLike(
        {
            "faithfulness": [0.8, 0.9],
            "answer_relevancy": [0.6, 0.7],
        }
    )
    _install_fake_ragas(monkeypatch, df_result)

    samples = [
        {"question": "q1", "answer": "a1", "contexts": ["c1"], "ground_truths": []},
        {"question": "q2", "answer": "a2", "contexts": ["c2"], "ground_truths": []},
    ]

    scores = run_ragas(samples, metrics=["faithfulness", "answer_relevancy"])

    assert pytest.approx(scores["faithfulness"], 0.0001) == 0.85
    assert pytest.approx(scores["answer_relevancy"], 0.0001) == 0.65


def test_run_ragas_handles_dict_payload_and_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dict_result = _DictLikeResult(
        {
            "answer_relevancy": 0.82,
            "context_recall": [0.4, 0.6],
        }
    )
    _install_fake_ragas(monkeypatch, dict_result)

    samples = [
        {"question": "q", "answer": "a", "contexts": ["c"], "ground_truths": ["ref"]},
    ]

    scores = run_ragas(samples, metrics=["answer_relevance", "context_recall"])

    assert pytest.approx(scores["answer_relevance"], 0.0001) == 0.82
    assert pytest.approx(scores["context_recall"], 0.0001) == 0.5
