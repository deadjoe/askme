"""
Document reranking service with BGE local and Cohere fallback.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from askme.core.config import RerankConfig
from askme.retriever.base import Document, RetrievalResult

# 可选依赖：按需延迟导入，避免在未安装环境下阻塞其它功能
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
    async def rerank(
        self, query: str, documents: List[RetrievalResult], top_n: int = 8
    ) -> List[RerankResult]:
        """Rerank documents based on query relevance."""
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the reranker model."""
        pass


class BGEReranker(BaseReranker):
    """BGE reranker using FlagEmbedding local models."""

    def __init__(self, config: RerankConfig):
        self.config = config
        self.model: Any | None = None
        self.device = (
            "cuda"
            if (_torch and hasattr(_torch, "cuda") and _torch.cuda.is_available())
            else "cpu"
        )
        self._is_initialized = False

    async def initialize(self) -> None:
        """Initialize the BGE reranker model."""
        if self._is_initialized:
            return

        try:
            logger.info(f"Loading BGE reranker: {self.config.local_model}")

            # 延迟导入模型，以减少未安装依赖时的导入错误
            from FlagEmbedding import FlagReranker

            # Load BGE reranker model；若遇到 trust_remote_code 提示，回退到基础模型
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

            # Sort by rerank score (descending) with stable tie-breaking using original rank
            rerank_results.sort(key=lambda x: (-x.rerank_score, x.original_rank))

            # Update new ranks and apply top_n filtering
            for i, result in enumerate(rerank_results):
                result.new_rank = i + 1
                # Preserve original rank info for debugging and consistency
                if "original_rank_preserved" not in result.debug_info:
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
                    f"Token-based truncation: {len(content_tokens)} → {len(truncated_tokens)} tokens"
                )

                return truncated_content

        except Exception as e:
            logger.debug(
                f"Tokenizer-based truncation failed: {e}, falling back to char-based"
            )

        # Fallback: Enhanced character-based approximation
        return self._fallback_char_truncation(content, query, max_length)

    def _fallback_char_truncation(
        self, content: str, query: str, max_length: int
    ) -> str:
        """
        Fallback character-based truncation with better multi-byte handling.
        """
        # Better estimation for different text types
        query_char_count = len(query)

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
            f"(est. {estimated_content_tokens} → {estimate_token_count(truncated)} tokens)"
        )

        return truncated


# Cohere reranker removed - using local-only approach as per requirements


class RerankingService:
    """Main reranking service - local BGE-only implementation."""

    def __init__(self, config: RerankConfig, cohere_api_key: Optional[str] = None):
        self.config = config

        # Initialize local BGE reranker only
        self.bge_reranker = BGEReranker(config) if config.local_enabled else None

        # Cohere removed - warn if configuration still references it
        if config.cohere_enabled or cohere_api_key:
            logger.warning(
                "Cohere reranker configuration detected but not supported in local-only mode. "
                "Only BGE local reranker will be used."
            )

        self._fallback_enabled = False  # No cloud fallback in local-only mode

    async def initialize(self) -> None:
        """Initialize the local-only reranking service."""
        try:
            if self.bge_reranker:
                await self.bge_reranker.initialize()
                logger.info("BGE local reranker initialized")
            else:
                raise RuntimeError(
                    "No local reranker configured. BGE reranker is required."
                )

            logger.info("Local-only reranking service initialized successfully")

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
        Rerank documents using local BGE reranker only.

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

        # Use only local BGE reranker
        if not self.bge_reranker:
            raise RuntimeError("No local BGE reranker available")

        try:
            logger.debug("Using local BGE reranker")
            return await self.bge_reranker.rerank(query, documents, top_n)

        except Exception as e:
            logger.error(f"Local BGE reranker failed: {e}")
            raise RuntimeError(f"Reranking failed with local BGE reranker: {e}") from e

    async def rerank_by_method(
        self,
        query: str,
        documents: List[RetrievalResult],
        method: str = "bge_local",
        top_n: Optional[int] = None,
    ) -> List[RerankResult]:
        """
        Rerank documents using a specific method (local BGE only).

        Args:
            query: Query string for reranking
            documents: Retrieved documents to rerank
            method: Reranking method (only "bge_local" supported)
            top_n: Number of top documents to return

        Returns:
            List of reranked results
        """
        top_n = top_n or self.config.top_n

        if method == "bge_local":
            if not self.bge_reranker:
                raise ValueError("BGE local reranker not available")
            return await self.bge_reranker.rerank(query, documents, top_n)

        elif method == "cohere":
            raise ValueError("Cohere reranker not supported in local-only mode")

        else:
            raise ValueError(
                f"Unknown reranking method: {method}. Only 'bge_local' supported."
            )

    def get_available_methods(self) -> List[str]:
        """Get list of available reranking methods (local BGE only)."""
        methods = []
        if self.bge_reranker:
            methods.append("bge_local")
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
                "cohere_enabled": False,  # Always False in local-only mode
            },
        }

        # Add local reranker info only
        if self.bge_reranker:
            info["bge_reranker"] = self.bge_reranker.get_model_info()
        else:
            info["warning"] = "No local reranker configured"

        return info

    async def cleanup(self) -> None:
        """Clean up local reranking service resources."""
        try:
            if self.bge_reranker:
                await self.bge_reranker.cleanup()

            logger.info("Local reranking service cleaned up")

        except Exception as e:
            logger.error(f"Error during reranking service cleanup: {e}")
            raise
