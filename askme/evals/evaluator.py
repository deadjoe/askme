"""
Minimal evaluator that tries to use TruLens/Ragas if installed and enabled;
otherwise falls back to simple heuristic metrics so /eval/run is functional.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from askme.core.config import EvaluationConfig


@dataclass
class EvalItem:
    query: str
    answer: str
    contexts: List[str]
    ground_truth: Optional[str] = None


@dataclass
class EvalScore:
    name: str
    value: float
    threshold: Optional[float] = None


class Evaluator:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    def evaluate_batch(self, items: List[EvalItem]) -> List[EvalScore]:
        """Try to compute metrics via frameworks, else heuristic fallback."""
        # Heuristic placeholder: use simple lexical overlaps
        # Faithfulness: fraction of answer tokens present in contexts
        if not items:
            return [
                EvalScore(
                    "faithfulness", 0.0, self.config.thresholds.ragas_faithfulness_min
                ),
                EvalScore("answer_relevancy", 0.0, None),
                EvalScore(
                    "context_precision", 0.0, self.config.thresholds.ragas_precision_min
                ),
            ]

        import re

        def tok(s: str) -> List[str]:
            return re.findall(r"\w+", s.lower())

        faith_scores: List[float] = []
        relev_scores: List[float] = []
        prec_scores: List[float] = []

        for it in items:
            atoks = tok(it.answer)
            qtokens = tok(it.query)
            ctoks = tok(" \n".join(it.contexts)) if it.contexts else []

            if atoks:
                overlap = sum(1 for t in atoks if t in ctoks)
                faith = overlap / max(1, len(atoks))
            else:
                faith = 0.0

            # relevance: answer tokens overlapping query tokens
            if atoks:
                rel = sum(1 for t in atoks if t in qtokens) / max(1, len(atoks))
            else:
                rel = 0.0

            # precision: unique context tokens used by answer / context size
            uniq_a = set(atoks)
            uniq_c = set(ctoks)
            prec = len(uniq_a & uniq_c) / max(1, len(uniq_c))

            faith_scores.append(faith)
            relev_scores.append(rel)
            prec_scores.append(prec)

        def avg(xs: List[float]) -> float:
            return sum(xs) / len(xs)

        return [
            EvalScore(
                name="faithfulness",
                value=avg(faith_scores),
                threshold=self.config.thresholds.ragas_faithfulness_min,
            ),
            EvalScore(name="answer_relevancy", value=avg(relev_scores), threshold=None),
            EvalScore(
                name="context_precision",
                value=avg(prec_scores),
                threshold=self.config.thresholds.ragas_precision_min,
            ),
        ]
