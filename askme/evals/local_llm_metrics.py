"""Local LLM-based evaluators to mirror Ragas metrics using Ollama."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, cast

import openai


def _default_client() -> openai.OpenAI:
    base_url = getenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
    api_key = getenv("OPENAI_API_KEY", "ollama-local")
    return openai.OpenAI(base_url=base_url, api_key=api_key)


def getenv(name: str, default: str) -> str:
    from os import getenv as _getenv

    return _getenv(name, default)


@dataclass
class LocalJudge:
    model: str = getenv("ASKME_RAGAS_LLM_MODEL", "gpt-oss:20b")
    max_retries: int = 2
    temperature: float = 0.0

    def __post_init__(self) -> None:
        self._client = _default_client()

    def _chat(self, system: str, user: str) -> str:
        last_exc: Optional[Exception] = None
        for _ in range(self.max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=self.temperature,
                )
                return resp.choices[0].message.content or ""
            except Exception as exc:  # pragma: no cover
                last_exc = exc
        raise RuntimeError(f"LocalJudge failed after retries: {last_exc}")

    def _safe_json(self, text: str) -> Dict[str, Any]:
        try:
            return cast(Dict[str, Any], json.loads(text))
        except json.JSONDecodeError:
            cleaned = text.strip()
            cleaned = cleaned[cleaned.find("{") : cleaned.rfind("}") + 1]
            return cast(Dict[str, Any], json.loads(cleaned))

    def score_faithfulness(self, contexts: List[str], answer: str) -> Dict[str, Any]:
        joined = "\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
        system = "You are a meticulous fact-checker."
        json_schema = json.dumps(
            {
                "score": "float between 0 and 1",
                "justification": "short reason",
            }
        )
        user = f"""
Evaluate how well the answer is supported by the provided context.

Context passages
{joined}

Answer
{answer}

Respond in JSON format: {json_schema}."""
        raw = self._chat(system, user)
        payload = self._safe_json(raw)
        score = float(payload.get("score", 0.0))
        score = max(0.0, min(1.0, score))
        return {"score": score, "raw": payload}

    def score_answer_relevancy(self, question: str, answer: str) -> Dict[str, Any]:
        system = "You judge how directly an answer addresses a question."
        json_schema = json.dumps(
            {
                "score": "float between 0 and 1",
                "justification": "short sentence",
            }
        )
        user = f"""
Question {question}
Answer {answer}

Return JSON payload {json_schema}."""
        raw = self._chat(system, user)
        payload = self._safe_json(raw)
        score = float(payload.get("score", 0.0))
        score = max(0.0, min(1.0, score))
        return {"score": score, "raw": payload}

    def score_context_precision(
        self, contexts: List[str], answer: str
    ) -> Dict[str, Any]:
        joined = "\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
        system = "You evaluate which context passages are actually used in an answer."
        json_schema = json.dumps(
            {
                "used_ids": "list of passage indices starting at 1",
                "justification": "short text",
            }
        )
        user = f"""
Context passages
{joined}

Answer
{answer}

Which passages provided relevant information for the answer? Respond with JSON
schema {json_schema}"""
        raw = self._chat(system, user)
        payload = self._safe_json(raw)
        used = payload.get("used_ids", [])
        if isinstance(used, int):
            used_ids = [used]
        else:
            used_ids = [int(x) for x in used if isinstance(x, (int, float, str))]
        used_ids = [i for i in used_ids if 1 <= i <= len(contexts)]
        precision = len(set(used_ids)) / max(1, len(contexts))
        return {"score": precision, "raw": payload}

    def score_context_recall(self, contexts: List[str], answer: str) -> Dict[str, Any]:
        joined = "\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
        system = (
            "You check whether the answer covers the important points from context."
        )
        json_schema = json.dumps(
            {
                "score": "float between 0 and 1",
                "justification": "short reason",
            }
        )
        user = f"""
Context passages
{joined}

Answer
{answer}

Does the answer cover the key facts the context provides? Respond with JSON
schema {json_schema}"""
        raw = self._chat(system, user)
        payload = self._safe_json(raw)
        score = float(payload.get("score", 0.0))
        score = max(0.0, min(1.0, score))
        return {"score": score, "raw": payload}


def evaluate_samples(
    samples: List[Dict[str, Any]], metrics: Optional[List[str]] = None
) -> Dict[str, float]:
    metrics = metrics or [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ]
    judge = LocalJudge()
    accum: Dict[str, List[float]] = {m: [] for m in metrics}

    for sample in samples:
        question = sample.get("question", "")
        answer = sample.get("answer", "")
        contexts = list(sample.get("contexts", []))

        if "faithfulness" in metrics:
            res = judge.score_faithfulness(contexts, answer)
            accum["faithfulness"].append(res["score"])
        if "answer_relevancy" in metrics:
            res = judge.score_answer_relevancy(question, answer)
            accum["answer_relevancy"].append(res["score"])
        if "context_precision" in metrics:
            res = judge.score_context_precision(contexts, answer)
            accum["context_precision"].append(res["score"])
        if "context_recall" in metrics:
            res = judge.score_context_recall(contexts, answer)
            accum["context_recall"].append(res["score"])

    averaged: Dict[str, float] = {}
    for name, values in accum.items():
        averaged[name] = sum(values) / len(values) if values else 0.0
    return averaged
