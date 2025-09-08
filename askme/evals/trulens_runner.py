"""
TruLens RAG Triad runner.

Attempts to compute context_relevance, groundedness, and answer_relevance.
This module is optional and will raise a clear RuntimeError if TruLens or
compatible providers are not available/configured.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import os


def run_trulens(
    samples: List[Dict[str, Any]],
    provider: Optional[str] = None,
) -> Dict[str, float]:
    try:
        from trulens_eval import Tru
        from trulens_eval.feedback import Feedback, Groundedness
        from trulens_eval.feedback.provider import OpenAI
    except Exception as e:
        raise RuntimeError(
            "TruLens not available or provider missing. Install trulens-eval "
            "and configure a provider."
        ) from e

    # NOTE: This is a minimal placeholder. In a real setup, you would wrap your
    # RAG pipeline with TruLens App and record traces. Here we leverage
    # text-level feedback providers to approximate triad scores on the samples.
    try:
        # Provider selection: currently only OpenAI placeholder if API is present.
        # Users running locally without OpenAI will likely hit the exception
        # and should skip TruLens in favor of Ragas.
        base_url = os.getenv("OPENAI_BASE_URL", "")
        if base_url:
            os.environ["OPENAI_API_BASE"] = base_url
        provider_client = OpenAI()

        grounded = Groundedness(groundedness_provider=provider_client)
        f_ans_rel = Feedback(provider_client.relevance).on(
            Feedback.ANSWER, Feedback.QUESTION
        )
        f_ctx_rel = Feedback(provider_client.relevance).on(
            Feedback.CONTEXT, Feedback.QUESTION
        )

        # Aggregate
        scores = {
            "groundedness": 0.0,
            "answer_relevance": 0.0,
            "context_relevance": 0.0,
        }
        n = max(1, len(samples))
        for s in samples:
            q = s.get("question", "")
            a = s.get("answer", "")
            contexts = s.get("contexts", [])
            ctx_text = "\n".join(contexts)

            # Compute individual scores
            try:
                g = grounded.groundedness_measure(a, contexts)
                scores["groundedness"] += float(g)
            except Exception:
                pass
            try:
                ar = f_ans_rel(a, q)
                scores["answer_relevance"] += float(ar)
            except Exception:
                pass
            try:
                cr = f_ctx_rel(ctx_text, q)
                scores["context_relevance"] += float(cr)
            except Exception:
                pass

        for k in list(scores.keys()):
            scores[k] = scores[k] / n

        return scores
    except Exception as e:
        raise RuntimeError(
            "TruLens evaluation failed: "
            f"{e}. Ensure provider credentials are configured."
        ) from e
