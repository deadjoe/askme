"""
Document reranking service supporting Qwen3 local reranker with optional BGE fallback.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, cast

from askme.core.config import RerankConfig
from askme.retriever.base import Document, RetrievalResult

# Optional dependency: lazy import to avoid blocking other features when not installed
_np: Any
try:
    import numpy as _np_module
except Exception:  # pragma: no cover
    _np = None
else:
    _np = _np_module

_torch: Any
try:
    import torch as _torch_module
except Exception:  # pragma: no cover
    _torch = None
else:
    _torch = _torch_module


def _detect_torch_device() -> str:
    """Detect the best available torch device."""

    device = "cpu"

    if _torch is None:
        return device

    # Prefer CUDA when available
    if hasattr(_torch, "cuda"):
        cuda_mod = getattr(_torch, "cuda")
        is_available = getattr(cuda_mod, "is_available", None)
        if callable(is_available) and is_available():
            return "cuda"

    # Fall back to Apple Silicon MPS when present
    if hasattr(_torch, "backends") and hasattr(_torch.backends, "mps"):
        mps_mod = getattr(_torch.backends, "mps")
        is_available = getattr(mps_mod, "is_available", None)
        if callable(is_available) and is_available():
            return "mps"

    return device


logger = logging.getLogger(__name__)


@dataclass
class RerankResult:
    """Result from reranking operation."""

    document: Document
    original_score: float
    rerank_score: float
    original_rank: int
    new_rank: int
    reranker_used: str
    debug_info: Optional[Dict[str, Any]] = None


class BaseReranker(ABC):
    """Abstract base class for rerankers."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize underlying model resources."""

    @abstractmethod
    async def rerank(
        self, query: str, documents: List[RetrievalResult], top_n: int = 8
    ) -> List[RerankResult]:
        """Rerank documents based on query relevance."""
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the reranker model."""
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """Release underlying resources."""
        pass


class BGEReranker(BaseReranker):
    """BGE reranker using FlagEmbedding local models."""

    def __init__(self, config: RerankConfig):
        self.config = config
        self.model: Any | None = None
        # Device detection matching embeddings.py logic
        self.device = _detect_torch_device()
        self._is_initialized = False

    async def initialize(self) -> None:
        """Initialize the BGE reranker model."""
        if self._is_initialized:
            return

        try:
            logger.info(f"Loading BGE reranker: {self.config.local_model}")

            # Lazy import model to reduce import errors when dependencies missing
            from FlagEmbedding import FlagReranker

            # Load BGE reranker model; fallback on trust_remote_code issues
            try:
                self.model = FlagReranker(
                    self.config.local_model,
                    use_fp16=(self.device != "cpu"),
                    device=self.device,
                    trust_remote_code=True,
                )
            except Exception as e:
                msg = str(e)
                if (
                    "trust_remote_code" in msg
                    or "allow custom code" in msg
                    or "Unrecognized configuration class" in msg
                ):
                    fallback = "BAAI/bge-reranker-base"
                    logger.warning(
                        "Local reranker requires trust_remote_code, falling back to %s",
                        fallback,
                    )
                    self.model = FlagReranker(
                        fallback,
                        use_fp16=(self.device != "cpu"),
                        device=self.device,
                        trust_remote_code=True,
                    )
                else:
                    raise

            self._is_initialized = True
            logger.info(f"BGE reranker loaded on device: {self.device}")

        except Exception as e:
            logger.error(f"Failed to initialize BGE reranker: {e}")
            raise

    async def rerank(
        self, query: str, documents: List[RetrievalResult], top_n: int = 8
    ) -> List[RerankResult]:
        """Rerank documents using BGE reranker."""
        if not self._is_initialized:
            await self.initialize()

        if not documents:
            return []

        try:
            # Prepare query-document pairs for reranking
            pairs: List[List[str]] = []
            original_docs: List[RetrievalResult] = []

            for doc_result in documents:
                # Intelligent token-based truncation using BGE tokenizer
                content = self._truncate_by_tokens(
                    doc_result.document.content, query, self.config.local_max_length
                )
                pairs.append([query, content])
                original_docs.append(doc_result)

            # Get reranking scores in batches
            batch_size = self.config.local_batch_size
            all_scores: List[float] = []

            for i in range(0, len(pairs), batch_size):
                batch_pairs = pairs[i : i + batch_size]

                logger.debug(
                    "Reranking batch %s: %s pairs",
                    (i // batch_size) + 1,
                    len(batch_pairs),
                )

                # Get scores for this batch
                if self.model is None:
                    raise RuntimeError("BGE reranker model is not initialized")
                batch_scores = self.model.compute_score(batch_pairs)

                # Handle single score vs list of scores
                if isinstance(batch_scores, (int, float)):
                    batch_scores = [batch_scores]
                elif _np is not None and isinstance(batch_scores, _np.ndarray):
                    batch_scores = batch_scores.tolist()

                all_scores.extend(batch_scores)

            # Create rerank results
            rerank_results: List[RerankResult] = []

            for i, (original_result, rerank_score) in enumerate(
                zip(original_docs, all_scores)
            ):
                rerank_results.append(
                    RerankResult(
                        document=original_result.document,
                        original_score=original_result.score,
                        rerank_score=float(rerank_score),
                        original_rank=original_result.rank,
                        new_rank=i + 1,  # Will be updated after sorting
                        reranker_used="bge_local",
                        debug_info={
                            "model": self.config.local_model,
                            "device": self.device,
                            "original_method": original_result.retrieval_method,
                        },
                    )
                )

            # Sort by rerank score (descending) with stable tie-breaking
            rerank_results.sort(key=lambda x: (-x.rerank_score, x.original_rank))

            # Update new ranks and apply top_n filtering
            for i, result in enumerate(rerank_results):
                result.new_rank = i + 1
                # Preserve original rank info for debugging and consistency
                if (
                    result.debug_info is not None
                    and "original_rank_preserved" not in result.debug_info
                ):
                    result.debug_info["original_rank_preserved"] = result.original_rank

            # Apply score threshold if configured
            if self.config.score_threshold > 0:
                rerank_results = [
                    r
                    for r in rerank_results
                    if r.rerank_score >= self.config.score_threshold
                ]

            # Return top N results
            final_results = rerank_results[:top_n]

            logger.info(
                "BGE reranking completed: %s → %s documents",
                len(documents),
                len(final_results),
            )

            return final_results

        except Exception as e:
            logger.error(f"BGE reranking failed: {e}")
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """Get BGE reranker model information."""
        return {
            "type": "bge_local",
            "model_name": self.config.local_model,
            "device": self.device,
            "batch_size": self.config.local_batch_size,
            "max_length": self.config.local_max_length,
            "score_threshold": self.config.score_threshold,
            "initialized": self._is_initialized,
        }

    async def cleanup(self) -> None:
        """Clean up BGE reranker resources."""
        if self.model is not None:
            # Clear CUDA cache if using GPU
            if _torch and hasattr(_torch, "cuda") and _torch.cuda.is_available():
                _torch.cuda.empty_cache()

            # Explicitly delete model as suggested by FlagEmbedding developers
            del self.model
            self.model = None
            self._is_initialized = False
            logger.info("BGE reranker cleaned up")

    def _truncate_by_tokens(self, content: str, query: str, max_length: int) -> str:
        """
        Truncate content based on actual token count using BGE tokenizer.

        This addresses the issue where character-based truncation can be inaccurate
        for multi-byte characters (Chinese, emojis, etc.) and doesn't align with
        the model's actual token limits.
        """
        try:
            # Try to use the model's tokenizer for precise token counting
            if self.model and hasattr(self.model, "tokenizer"):
                tokenizer = self.model.tokenizer

                # Encode query and content separately to manage total length
                query_tokens = tokenizer.encode(query, add_special_tokens=False)
                query_length = len(query_tokens)

                # Reserve space for special tokens and query
                available_content_length = (
                    max_length - query_length - 10
                )  # Buffer for [CLS], [SEP], etc.

                if available_content_length <= 50:  # Minimum useful content length
                    logger.warning(
                        f"Query too long ({query_length} tokens), using minimal content"
                    )
                    available_content_length = 50

                # Tokenize and truncate content
                content_tokens = tokenizer.encode(content, add_special_tokens=False)

                if len(content_tokens) <= available_content_length:
                    # Content fits within limits
                    return content

                # Truncate content tokens
                truncated_tokens = content_tokens[:available_content_length]

                # Decode back to text
                truncated_content = tokenizer.decode(
                    truncated_tokens, skip_special_tokens=True
                )

                logger.debug(
                    f"Token-based truncation: {len(content_tokens)} → "
                    f"{len(truncated_tokens)} tokens"
                )

                return str(truncated_content)

        except Exception as e:
            logger.debug(
                f"Tokenizer-based truncation failed: {e}, falling back to char-based"
            )

        # Fallback: Enhanced character-based approximation
        return str(self._fallback_char_truncation(content, query, max_length))

    def _fallback_char_truncation(
        self, content: str, query: str, max_length: int
    ) -> str:
        """
        Fallback character-based truncation with better multi-byte handling.
        """

        # Better estimation for different text types
        # Estimate tokens more accurately based on text characteristics
        def estimate_token_count(text: str) -> int:
            # More sophisticated estimation
            import re

            # Count different character types
            ascii_chars = len(re.findall(r"[a-zA-Z0-9\s]", text))
            cjk_chars = len(
                re.findall(r"[\u4e00-\u9fff\u3400-\u4dbf]", text)
            )  # Chinese
            other_chars = len(text) - ascii_chars - cjk_chars

            # Different tokenization rates for different scripts
            estimated_tokens = (
                ascii_chars * 0.75  # English: ~0.75 tokens per char
                + cjk_chars * 1.2  # Chinese: ~1.2 tokens per char
                + other_chars * 1.0  # Other: ~1.0 tokens per char
            )

            return int(estimated_tokens)

        query_estimated_tokens = estimate_token_count(query)
        available_content_tokens = max_length - query_estimated_tokens - 10

        if available_content_tokens <= 0:
            available_content_tokens = 50  # Minimum

        # Find character length that approximates the token limit
        content_chars = len(content)
        estimated_content_tokens = estimate_token_count(content)

        if estimated_content_tokens <= available_content_tokens:
            return content

        # Calculate truncation ratio
        truncation_ratio = available_content_tokens / estimated_content_tokens
        target_char_length = int(content_chars * truncation_ratio)

        # Truncate at word/sentence boundaries when possible
        truncated = content[:target_char_length]

        # Try to end at a reasonable boundary
        for boundary in [". ", "。", "! ", "！", "? ", "？", "\n", " "]:
            last_boundary = truncated.rfind(boundary)
            if last_boundary > target_char_length * 0.8:  # Don't lose too much content
                truncated = truncated[: last_boundary + len(boundary)].rstrip()
                break

        logger.debug(
            f"Char-based truncation: {content_chars} → {len(truncated)} chars "
            f"(est. {estimated_content_tokens} → "
            f"{estimate_token_count(truncated)} tokens)"
        )

        return truncated


class Qwen3Reranker(BaseReranker):
    """Qwen3 reranker using Hugging Face causal LM interface."""

    DEFAULT_INSTRUCTION = (
        "Given a web search query, retrieve relevant passages that answer the query"
    )
    SYSTEM_PROMPT = (
        "<|im_start|>system\n"
        "Judge whether the Document meets the requirements based on the Query and the "
        'Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n'
        "<|im_start|>user\n"
    )
    ASSISTANT_PREFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    def __init__(self, config: RerankConfig):
        if _torch is None:
            raise RuntimeError("Torch is required for Qwen3 reranker")

        self.config = config
        self.device = _detect_torch_device()
        self.model: Any | None = None
        self.tokenizer: Any | None = None
        self.token_true_id: Optional[int] = None
        self.token_false_id: Optional[int] = None
        self.prefix_tokens: List[int] = []
        self.suffix_tokens: List[int] = []
        self.max_length = max(config.local_max_length, 512)
        self._body_max_length: int = 0
        self._is_initialized = False

    async def initialize(self) -> None:
        if self._is_initialized:
            return

        try:
            logger.info("Loading Qwen3 reranker: %s", self.config.local_model)

            from transformers import AutoModelForCausalLM, AutoTokenizer  # lazy import

            tokenizer_kwargs: Dict[str, Any] = {
                "padding_side": "left",
                "trust_remote_code": True,
            }
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.local_model,
                **tokenizer_kwargs,
            )

            model_kwargs: Dict[str, Any] = {"trust_remote_code": True}

            if self.config.local_flash_attention:
                model_kwargs["attn_implementation"] = "flash_attention_2"

            if self.config.local_use_fp16 and self.device != "cpu":
                model_kwargs["torch_dtype"] = _torch.float16

            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.local_model,
                **model_kwargs,
            )

            if self.device != "cpu":
                self.model.to(self.device)

            self.model.eval()

            # Prepare prompt tokens and score heads
            self.prefix_tokens = self.tokenizer.encode(
                self.SYSTEM_PROMPT, add_special_tokens=False
            )
            self.suffix_tokens = self.tokenizer.encode(
                self.ASSISTANT_PREFIX, add_special_tokens=False
            )

            self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
            self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")

            if self.token_false_id is None or self.token_true_id is None:
                raise RuntimeError(
                    "Failed to locate 'yes'/'no' tokens for Qwen3 reranker"
                )

            tokenizer_max_length = getattr(
                self.tokenizer, "model_max_length", self.max_length
            )
            self.max_length = min(max(self.max_length, 512), tokenizer_max_length)

            prefix_len = len(self.prefix_tokens)
            suffix_len = len(self.suffix_tokens)
            self._body_max_length = max(self.max_length - prefix_len - suffix_len, 128)

            self._is_initialized = True
            logger.info("Qwen3 reranker loaded on device: %s", self.device)

        except Exception as exc:  # pragma: no cover - heavy dependency
            logger.error("Failed to initialize Qwen3 reranker: %s", exc)
            raise

    def _format_instruction(self, query: str, document: str) -> str:
        instruction = self.config.local_instruction or self.DEFAULT_INSTRUCTION
        return (
            f"<Instruct>: {instruction}\n"
            f"<Query>: {query}\n"
            f"<Document>: {document}"
        )

    def _truncate_document(self, text: str) -> str:
        if not self.tokenizer:
            return text

        try:
            token_ids = cast(
                List[int], self.tokenizer.encode(text, add_special_tokens=False)
            )
            if len(token_ids) <= self._body_max_length:
                return text

            truncated = token_ids[: self._body_max_length]
            return cast(str, self.tokenizer.decode(truncated, skip_special_tokens=True))
        except Exception:  # pragma: no cover - tokenizer edge cases
            return text[: self._body_max_length * 3]  # simple char fallback

    def _prepare_inputs(self, pairs: List[str]) -> Dict[str, Any]:
        assert self.tokenizer is not None  # for mypy

        body = self.tokenizer(
            pairs,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=self._body_max_length,
        )

        input_ids: List[List[int]] = []
        for encoded in body["input_ids"]:
            token_list = list(cast(List[int], encoded))
            combined = self.prefix_tokens + token_list + self.suffix_tokens
            input_ids.append(combined[: self.max_length])

        padded = self.tokenizer.pad(
            {"input_ids": input_ids},
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        tensor_map: Dict[str, Any] = {k: v for k, v in padded.items()}
        return {k: cast(Any, v).to(self.device) for k, v in tensor_map.items()}

    def _compute_scores(self, inputs: Dict[str, Any]) -> List[float]:
        assert self.model is not None
        assert self.token_true_id is not None
        assert self.token_false_id is not None

        if _torch is None:
            raise RuntimeError("Torch is required for Qwen3 reranker")

        with _torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[:, -1, :]
            true_vec = logits[:, self.token_true_id]
            false_vec = logits[:, self.token_false_id]
            stacked = _torch.stack([false_vec, true_vec], dim=1)
            log_probs = _torch.nn.functional.log_softmax(stacked, dim=1)
            return cast(List[float], log_probs[:, 1].exp().tolist())

    async def rerank(
        self, query: str, documents: List[RetrievalResult], top_n: int = 8
    ) -> List[RerankResult]:
        if not self._is_initialized:
            await self.initialize()

        if not documents:
            return []

        pairs: List[str] = []
        originals: List[RetrievalResult] = []

        for doc in documents:
            content = self._truncate_document(doc.document.content)
            pairs.append(self._format_instruction(query, content))
            originals.append(doc)

        batch_size = self.config.local_batch_size
        scores: List[float] = []

        for idx in range(0, len(pairs), batch_size):
            batch_pairs = pairs[idx : idx + batch_size]
            inputs = self._prepare_inputs(batch_pairs)
            batch_scores = self._compute_scores(inputs)
            scores.extend(batch_scores)

        reranked: List[RerankResult] = []

        for i, (original, score) in enumerate(zip(originals, scores)):
            reranked.append(
                RerankResult(
                    document=original.document,
                    original_score=original.score,
                    rerank_score=float(score),
                    original_rank=original.rank,
                    new_rank=i + 1,
                    reranker_used="qwen_local",
                    debug_info={
                        "model": self.config.local_model,
                        "device": self.device,
                        "instruction": self.config.local_instruction
                        or self.DEFAULT_INSTRUCTION,
                        "backend": "qwen3",
                    },
                )
            )

        reranked.sort(key=lambda r: (-r.rerank_score, r.original_rank))

        for i, result in enumerate(reranked):
            result.new_rank = i + 1

        if self.config.score_threshold > 0:
            reranked = [
                r for r in reranked if r.rerank_score >= self.config.score_threshold
            ]

        final = reranked[:top_n]

        logger.info(
            "Qwen3 reranking completed: %s → %s documents", len(documents), len(final)
        )

        return final

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "type": "qwen_local",
            "model_name": self.config.local_model,
            "device": self.device,
            "batch_size": self.config.local_batch_size,
            "max_length": self.max_length,
            "instruction": self.config.local_instruction or self.DEFAULT_INSTRUCTION,
            "score_threshold": self.config.score_threshold,
            "initialized": self._is_initialized,
        }

    async def cleanup(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None

        self.tokenizer = None
        self._is_initialized = False
        logger.info("Qwen3 reranker cleaned up")


class RerankingService:
    """Main reranking service supporting configurable local rerankers."""

    def __init__(self, config: RerankConfig):
        self.config = config
        self.local_backend = config.local_backend
        self.local_reranker: Optional[BaseReranker]

        if not config.local_enabled:
            self.local_reranker = None
        elif self.local_backend == "qwen_local":
            self.local_reranker = Qwen3Reranker(config)
        elif self.local_backend == "bge_local":
            self.local_reranker = BGEReranker(config)
        else:
            raise ValueError(
                f"Unsupported local reranker backend: {self.local_backend}. "
                "Supported backends: qwen_local, bge_local"
            )
        self._fallback_enabled = False  # No cloud fallback in local-only mode

    async def initialize(self) -> None:
        """Initialize the local-only reranking service."""
        try:
            if not self.local_reranker:
                raise RuntimeError(
                    "No local reranker configured. Enable local reranker in settings."
                )

            await self.local_reranker.initialize()
            logger.info(
                "%s reranker initialized",
                "Qwen3" if isinstance(self.local_reranker, Qwen3Reranker) else "BGE",
            )

            logger.info("Local reranking service initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize reranking service: {e}")
            raise

    async def rerank(
        self,
        query: str,
        documents: List[RetrievalResult],
        top_n: Optional[int] = None,
        prefer_local: bool = True,  # Kept for API compatibility but ignored
    ) -> List[RerankResult]:
        """
        Rerank documents using the configured local reranker.

        Args:
            query: Query string for reranking
            documents: Retrieved documents to rerank
            top_n: Number of top documents to return (uses config default if None)
            prefer_local: Ignored - only local reranker available

        Returns:
            List of reranked results
        """
        if not documents:
            return []

        top_n = top_n or self.config.top_n

        if not self.local_reranker:
            raise RuntimeError("No local reranker available")

        try:
            logger.debug("Using local reranker backend: %s", self.local_backend)
            return await self.local_reranker.rerank(query, documents, top_n)

        except Exception as e:
            logger.error(f"Local reranker failed: {e}")
            raise RuntimeError(f"Reranking failed with local reranker: {e}") from e

    async def rerank_by_method(
        self,
        query: str,
        documents: List[RetrievalResult],
        method: str = "bge_local",
        top_n: Optional[int] = None,
    ) -> List[RerankResult]:
        """
        Rerank documents using a specific local method.

        Args:
            query: Query string for reranking
            documents: Retrieved documents to rerank
        method: Reranking method ("qwen_local" or "bge_local")
            top_n: Number of top documents to return

        Returns:
            List of reranked results
        """
        top_n = top_n or self.config.top_n

        if method == "bge_local":
            if self.local_backend != "bge_local" or self.local_reranker is None:
                raise ValueError("BGE local reranker not available")
            return await self.local_reranker.rerank(query, documents, top_n)

        if method == "qwen_local":
            if self.local_backend != "qwen_local" or self.local_reranker is None:
                raise ValueError("Qwen3 local reranker not available")
            return await self.local_reranker.rerank(query, documents, top_n)

        raise ValueError(
            f"Unknown reranking method: {method}. Supported: 'qwen_local', 'bge_local'."
        )

    def get_available_methods(self) -> List[str]:
        """Get list of available local reranking methods."""
        methods = []
        if self.local_reranker:
            methods.append(self.local_backend)
        return methods

    def get_service_info(self) -> Dict[str, Any]:
        """Get reranking service information (local-only)."""
        info = {
            "available_methods": self.get_available_methods(),
            "fallback_enabled": self._fallback_enabled,
            "mode": "local_only",
            "config": {
                "top_n": self.config.top_n,
                "score_threshold": self.config.score_threshold,
                "local_enabled": self.config.local_enabled,
            },
        }

        # Add local reranker info only
        if self.local_reranker:
            info["local_backend"] = self.local_backend
            info["local_model"] = self.local_reranker.get_model_info()
        else:
            info["warning"] = "No local reranker configured"

        return info

    async def cleanup(self) -> None:
        """Clean up local reranking service resources."""
        try:
            if self.local_reranker:
                await self.local_reranker.cleanup()

            logger.info("Local reranking service cleaned up")

        except Exception as e:
            logger.error(f"Error during reranking service cleanup: {e}")
            raise
