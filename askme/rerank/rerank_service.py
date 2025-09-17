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
                # 轻量长度控制：用字符近似模型的 token 上限，避免超长文本影响速度与稳定性
                content = doc_result.document.content
                try:
                    char_limit = max(512, int(self.config.local_max_length) * 4)
                except Exception:
                    char_limit = 2048
                if len(content) > char_limit:
                    content = content[:char_limit]
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

            # Sort by rerank score (descending)
            rerank_results.sort(key=lambda x: x.rerank_score, reverse=True)

            # Update new ranks and apply top_n filtering
            for i, result in enumerate(rerank_results):
                result.new_rank = i + 1

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

            self.model = None
            self._is_initialized = False
            logger.info("BGE reranker cleaned up")


class CohereReranker(BaseReranker):
    """Cohere reranker for cloud-based fallback."""

    def __init__(self, config: RerankConfig, api_key: Optional[str] = None):
        self.config = config
        self.api_key = api_key
        self.client = None

        if api_key:
            # 延迟导入 cohere SDK
            import cohere

            self.client = cohere.Client(api_key)

    async def rerank(
        self, query: str, documents: List[RetrievalResult], top_n: int = 8
    ) -> List[RerankResult]:
        """Rerank documents using Cohere API."""
        if not self.client:
            raise RuntimeError("Cohere API client not initialized. Check API key.")

        if not documents:
            return []

        try:
            # Prepare documents for Cohere API
            cohere_docs = []
            original_docs = []

            for doc_result in documents:
                # Truncate content if too long
                content = doc_result.document.content
                if len(content) > 4000:  # Cohere has content length limits
                    content = content[:4000] + "..."

                cohere_docs.append(content)
                original_docs.append(doc_result)

            # Call Cohere rerank API
            logger.debug(f"Calling Cohere rerank API with {len(cohere_docs)} documents")

            response = self.client.rerank(
                model=self.config.cohere_model,
                query=query,
                documents=cohere_docs,
                top_n=min(top_n, len(cohere_docs)),
                return_documents=self.config.cohere_return_documents,
                max_chunks_per_doc=self.config.cohere_max_chunks_per_doc,
            )

            # Process Cohere response
            rerank_results = []

            for i, result in enumerate(response.results):
                original_idx = result.index
                original_result = original_docs[original_idx]

                rerank_results.append(
                    RerankResult(
                        document=original_result.document,
                        original_score=original_result.score,
                        rerank_score=result.relevance_score,
                        original_rank=original_result.rank,
                        new_rank=i + 1,
                        reranker_used="cohere",
                        debug_info={
                            "model": self.config.cohere_model,
                            "original_method": original_result.retrieval_method,
                            "api_response_index": original_idx,
                        },
                    )
                )

            logger.info(
                "Cohere reranking completed: %s → %s documents",
                len(documents),
                len(rerank_results),
            )

            return rerank_results

        except Exception as e:
            logger.error(f"Cohere reranking failed: {e}")
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """Get Cohere reranker model information."""
        return {
            "type": "cohere",
            "model_name": self.config.cohere_model,
            "max_chunks_per_doc": self.config.cohere_max_chunks_per_doc,
            "return_documents": self.config.cohere_return_documents,
            "api_key_configured": self.client is not None,
        }


class RerankingService:
    """Main reranking service with fallback logic."""

    def __init__(self, config: RerankConfig, cohere_api_key: Optional[str] = None):
        self.config = config

        # Initialize rerankers
        self.bge_reranker = BGEReranker(config) if config.local_enabled else None
        self.cohere_reranker = (
            CohereReranker(config, cohere_api_key) if config.cohere_enabled else None
        )

        self._fallback_enabled = config.cohere_enabled and cohere_api_key is not None

    async def initialize(self) -> None:
        """Initialize the reranking service."""
        try:
            if self.bge_reranker:
                await self.bge_reranker.initialize()
                logger.info("BGE reranker initialized")

            if self.cohere_reranker:
                logger.info("Cohere reranker configured")

            logger.info("Reranking service initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize reranking service: {e}")
            raise

    async def rerank(
        self,
        query: str,
        documents: List[RetrievalResult],
        top_n: Optional[int] = None,
        prefer_local: bool = True,
    ) -> List[RerankResult]:
        """
        Rerank documents using available rerankers.

        Args:
            query: Query string for reranking
            documents: Retrieved documents to rerank
            top_n: Number of top documents to return (uses config default if None)
            prefer_local: Whether to prefer local reranker over cloud

        Returns:
            List of reranked results
        """
        if not documents:
            return []

        top_n = top_n or self.config.top_n

        # Determine which reranker to use
        primary_reranker: BaseReranker | None = None
        fallback_reranker: BaseReranker | None = None

        if prefer_local and self.bge_reranker:
            primary_reranker = self.bge_reranker
            fallback_reranker = self.cohere_reranker if self._fallback_enabled else None
        elif self.cohere_reranker:
            primary_reranker = self.cohere_reranker
            fallback_reranker = self.bge_reranker if self.bge_reranker else None
        elif self.bge_reranker:
            primary_reranker = self.bge_reranker
        else:
            raise RuntimeError("No rerankers available")

        # Try primary reranker
        try:
            logger.debug(
                f"Using primary reranker: {primary_reranker.__class__.__name__}"
            )
            return await primary_reranker.rerank(query, documents, top_n)

        except Exception as e:
            logger.warning(f"Primary reranker failed: {e}")

            # Try fallback if available
            if fallback_reranker:
                try:
                    logger.info(
                        f"Falling back to: {fallback_reranker.__class__.__name__}"
                    )
                    return await fallback_reranker.rerank(query, documents, top_n)

                except Exception as fallback_error:
                    logger.error(f"Fallback reranker also failed: {fallback_error}")
                    raise
            else:
                # No fallback available
                raise

    async def rerank_by_method(
        self,
        query: str,
        documents: List[RetrievalResult],
        method: str = "bge_local",
        top_n: Optional[int] = None,
    ) -> List[RerankResult]:
        """
        Rerank documents using a specific method.

        Args:
            query: Query string for reranking
            documents: Retrieved documents to rerank
            method: Reranking method ("bge_local" or "cohere")
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
            if not self.cohere_reranker:
                raise ValueError("Cohere reranker not available")
            return await self.cohere_reranker.rerank(query, documents, top_n)

        else:
            raise ValueError(f"Unknown reranking method: {method}")

    def get_available_methods(self) -> List[str]:
        """Get list of available reranking methods."""
        methods = []
        if self.bge_reranker:
            methods.append("bge_local")
        if self.cohere_reranker:
            methods.append("cohere")
        return methods

    def get_service_info(self) -> Dict[str, Any]:
        """Get reranking service information."""
        info = {
            "available_methods": self.get_available_methods(),
            "fallback_enabled": self._fallback_enabled,
            "config": {
                "top_n": self.config.top_n,
                "score_threshold": self.config.score_threshold,
                "local_enabled": self.config.local_enabled,
                "cohere_enabled": self.config.cohere_enabled,
            },
        }

        # Add reranker-specific info
        if self.bge_reranker:
            info["bge_reranker"] = self.bge_reranker.get_model_info()

        if self.cohere_reranker:
            info["cohere_reranker"] = self.cohere_reranker.get_model_info()

        return info

    async def cleanup(self) -> None:
        """Clean up reranking service resources."""
        try:
            if self.bge_reranker:
                await self.bge_reranker.cleanup()

            # Cohere client doesn't need cleanup

            logger.info("Reranking service cleaned up")

        except Exception as e:
            logger.error(f"Error during reranking service cleanup: {e}")
            raise
