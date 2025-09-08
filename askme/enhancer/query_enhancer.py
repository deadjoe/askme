"""
Query enhancement utilities: HyDE and RAG-Fusion (lightweight, deterministic).

These implementations are heuristic and local-only to keep the system offline
by default. They are designed to be fast and side-effect free, suitable for
unit tests and local development environments without external LLMs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class HydeConfig:
    enabled: bool = False
    max_tokens: int = 256


@dataclass
class RagFusionConfig:
    enabled: bool = False
    num_queries: int = 3


def generate_hyde_passage(query: str, max_tokens: int = 256) -> str:
    """Generate a simple hypothetical passage for HyDE without external calls.

    This is a deterministic, template-based expansion that mirrors the intent of
    HyDE: providing a likely relevant passage to improve retrieval recall.
    """
    base = (
        "This passage discusses the topic: {q}. It provides background, key concepts, "
        "and practical details to support answering questions about it in a concise way."
    )
    text = base.format(q=query)
    return text[: max_tokens * 4]  # rough truncation


def generate_fusion_queries(query: str, num_queries: int = 3) -> List[str]:
    """Generate a small set of related queries deterministically.

    Heuristics: add variants emphasizing definition, benefits, and steps.
    The original query is always included first.
    """
    variants = [
        query,
        f"Definition and overview of: {query}",
        f"Key benefits and use cases of {query}",
        f"Main steps, techniques, or components of {query}",
    ]
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for v in variants:
        if v not in seen:
            seen.add(v)
            unique.append(v)
    return unique[: max(1, num_queries)]
