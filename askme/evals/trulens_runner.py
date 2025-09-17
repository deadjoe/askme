"""
TruLens RAG Triad runner.

Attempts to compute context_relevance, groundedness, and answer_relevance.
This module is optional and will raise a clear RuntimeError if TruLens or
compatible providers are not available/configured.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Any, Dict, List, Optional


def _patch_openai_client(provider_client: Any) -> None:
    """Ensure OpenAI provider exposes completions API compatible with TruLens."""

    try:
        client = getattr(provider_client, "client", None)
        if client is None:
            return

        if not hasattr(client, "completions"):
            chat = getattr(client, "chat", None)

            if chat is None:
                return

            class _CompletionsProxy:
                def __init__(self, chat_obj: Any):
                    self._chat = chat_obj

                def create(self, **kwargs: Any) -> Any:
                    if "messages" in kwargs:
                        return self._chat.completions.create(**kwargs)
                    return self._chat.completions.create(messages=[], **kwargs)

            client.completions = _CompletionsProxy(chat)
    except Exception as exc:  # pragma: no cover - best effort
        print(f"DEBUG: Failed to patch OpenAI client for TruLens: {exc}")


def run_trulens(
    samples: List[Dict[str, Any]],
    provider: Optional[str] = None,
) -> Dict[str, float]:
    # Import uvloop compatibility utilities
    try:
        from askme.evals.uvloop_compat import (
            patch_nest_asyncio_import,
            run_in_thread_with_new_loop,
        )
    except ImportError as exc:  # pragma: no cover - triggered in tests
        raise RuntimeError(
            "TruLens not available or provider missing. Install trulens-eval "
            "and configure a provider."
        ) from exc

    try:
        # Pre-patch nest_asyncio to handle uvloop
        patch_nest_asyncio_import()

        # Import TruLens components in uvloop-compatible way
        def import_trulens() -> Dict[str, Any]:
            print("DEBUG: Attempting to import trulens_eval...")
            from trulens_eval import Tru  # noqa: F401 - imported for side effects

            print("DEBUG: Attempting to import trulens_eval.feedback...")
            from trulens_eval.feedback import Feedback, Groundedness

            print("DEBUG: trulens_eval.feedback.Feedback import successful")

            provider_class = None
            provider_kind = "openai"
            try:
                from trulens_eval.feedback.provider.litellm import LiteLLM

                provider_class = LiteLLM
                provider_kind = "litellm"
                print("DEBUG: trulens_eval.feedback.provider.litellm import successful")
            except Exception:
                from trulens_eval.feedback.provider import OpenAI

                provider_class = OpenAI
                provider_kind = "openai"
                print("DEBUG: LiteLLM provider unavailable; using OpenAI provider")

            return {
                "Feedback": Feedback,
                "Groundedness": Groundedness,
                "provider": provider_class,
                "provider_kind": provider_kind,
            }

        try:
            # Try direct import first (works in most environments)
            trulens_modules = import_trulens()
            print("DEBUG: Direct TruLens import successful")
        except Exception as e:
            print(f"DEBUG: Direct TruLens import failed: {e}, trying thread isolation")
            # If direct import fails due to uvloop, use thread isolation
            trulens_modules = run_in_thread_with_new_loop(import_trulens)
            print("DEBUG: Thread-isolated TruLens import successful")

        # Extract imported modules
        Feedback = trulens_modules["Feedback"]
        Groundedness = trulens_modules["Groundedness"]
        provider_cls = trulens_modules["provider"]
        provider_kind = trulens_modules["provider_kind"]

    except Exception as e:
        print(f"DEBUG: All TruLens import strategies failed: {e}")
        import traceback

        print(f"DEBUG: TruLens final traceback: {traceback.format_exc()}")
        raise RuntimeError(
            "TruLens not available or provider missing. Install trulens-eval "
            "and configure a provider."
        ) from e

    # NOTE: This is a minimal placeholder. In a real setup, you would wrap your
    # RAG pipeline with TruLens App and record traces. Here we leverage
    # text-level feedback providers to approximate triad scores on the samples.
    try:
        # Provider selection: use OpenAI-compatible endpoint (Ollama) if configured.
        base_url = os.getenv("OLLAMA_BASE_URL")
        if not base_url:
            base_url = os.getenv("OPENAI_BASE_URL", "http://localhost:11434")

        api_key = os.getenv("OLLAMA_API_KEY")
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY", "ollama-local")

        provider_model = os.getenv("ASKME_RAGAS_LLM_MODEL", "ollama/gpt-oss:20b")
        model_engine = os.getenv(
            "ASKME_TRULENS_LLM_MODEL",
            provider_model,
        )

        if provider_kind == "litellm":
            provider_client = provider_cls(
                model_engine=model_engine,
                completion_kwargs={"api_base": base_url, "api_key": api_key},
            )
        else:
            # Fallback to OpenAI provider - ensure environment variables align
            os.environ["OPENAI_API_BASE"] = base_url.rstrip("/")
            os.environ.setdefault("OPENAI_API_KEY", api_key)
            provider_client = provider_cls()
            _patch_openai_client(provider_client)

        # Create feedback functions using modern TruLens API
        grounded = Groundedness(groundedness_provider=provider_client)
        f_ans_rel = Feedback(provider_client.relevance)
        f_ctx_rel = Feedback(provider_client.context_relevance)

        # Helper function to add timeout protection to LLM calls
        def _call_with_timeout(func: Any, *args: Any, timeout: float = 60.0) -> Any:
            try:
                with ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(func, *args)
                    return future.result(timeout=timeout)
            except FutureTimeoutError:
                print(f"DEBUG: TruLens LLM call timed out after {timeout} seconds")
                return 0.0
            except Exception as e:
                print(f"DEBUG: TruLens LLM call failed: {e}")
                return 0.0

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
            if isinstance(contexts, str):
                context_list = [contexts]
            else:
                context_list = list(contexts)
            ctx_text = "\n".join(context_list)

            # Compute individual scores with timeout protection
            try:
                g = _call_with_timeout(
                    grounded.groundedness_measure,
                    a,
                    context_list,
                    timeout=60.0,
                )
                scores["groundedness"] += float(g)
            except Exception:
                pass
            try:
                ar = _call_with_timeout(f_ans_rel, a, q, timeout=60.0)
                scores["answer_relevance"] += float(ar)
            except Exception:
                pass
            try:
                cr = _call_with_timeout(f_ctx_rel, ctx_text, q, timeout=60.0)
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
