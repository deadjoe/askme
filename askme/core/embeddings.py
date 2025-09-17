"""
BGE-M3 embedding service with sparse and dense vector generation.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from askme.core.config import EmbeddingConfig

# Optional symbol for tests to patch without importing heavy deps at module import time
try:  # pragma: no cover - provide patch point for tests
    from FlagEmbedding import BGEM3FlagModel as _BGEM3FlagModel

    BGEM3FlagModel = _BGEM3FlagModel
except Exception:  # pragma: no cover
    BGEM3FlagModel = None

logger = logging.getLogger(__name__)


class BGEEmbeddingService:
    """BGE-M3 embedding service with dense, sparse, and multi-vector support."""

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.model = None
        # 允许在未安装 torch 的环境下工作（跳过 GPU 检测）
        self.device = (
            "cuda" if (hasattr(torch, "cuda") and torch.cuda.is_available()) else "cpu"
        )
        self._is_initialized = False
        # Thread pool for CPU-intensive embedding operations
        self._executor = ThreadPoolExecutor(
            max_workers=1,  # BGE models don't support true multithreading
            thread_name_prefix="bge-embed",
        )

    async def initialize(self) -> None:
        """Initialize the BGE-M3 model."""
        if self._is_initialized:
            return

        try:
            logger.info(f"Loading BGE-M3 model: {self.config.model}")

            # Load BGE-M3 model
            global BGEM3FlagModel
            if BGEM3FlagModel is None:
                from FlagEmbedding import BGEM3FlagModel as _BGEM3FlagModel

                BGEM3FlagModel = _BGEM3FlagModel
            self.model = BGEM3FlagModel(
                self.config.model,
                use_fp16=(self.device != "cpu" and self.config.use_fp16),
                device=self.device,
            )

            self._is_initialized = True
            logger.info(f"BGE-M3 model loaded on device: {self.device}")

        except Exception as e:
            logger.error(f"Failed to initialize BGE-M3 model: {e}")
            raise

    async def encode_documents(
        self, texts: List[str], batch_size: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Encode documents to get dense and sparse embeddings.

        Args:
            texts: List of document texts to encode
            batch_size: Batch size for processing (uses config default if None)

        Returns:
            List of dictionaries containing dense and sparse embeddings
        """
        if not self._is_initialized:
            await self.initialize()

        # Handle empty input early
        if not texts:
            return []

        # Check model is initialized before processing
        if self.model is None:
            raise RuntimeError("Embedding model not initialized")

        batch_size = batch_size or self.config.batch_size

        try:
            # Process in batches to manage memory
            results: List[Dict[str, Any]] = []

            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]

                logger.debug(
                    f"Processing batch {i//batch_size + 1}: {len(batch_texts)} texts"
                )

                loop = asyncio.get_event_loop()
                batch_results = await loop.run_in_executor(
                    self._executor,
                    self._encode_batch_sync,
                    batch_texts,
                    len(batch_texts),
                    self.config.max_length,
                )

                # Process results
                for j, text in enumerate(batch_texts):
                    dense_embedding = batch_results["dense_vecs"][j]
                    sparse_embedding = batch_results["sparse_vecs"][j]

                    # Normalize dense embedding if configured
                    if self.config.normalize_embeddings:
                        dense_embedding = dense_embedding / np.linalg.norm(
                            dense_embedding
                        )

                    # Convert sparse embedding to dictionary format
                    sparse_dict = self._convert_sparse_embedding(sparse_embedding)

                    results.append(
                        {
                            "text": text,
                            "dense_embedding": dense_embedding.tolist(),
                            "sparse_embedding": sparse_dict,
                            "embedding_dim": len(dense_embedding),
                        }
                    )

            logger.info(f"Encoded {len(texts)} documents successfully")
            return results

        except Exception as e:
            logger.error(f"Failed to encode documents: {e}")
            raise

    def _encode_batch_sync(
        self, texts: List[str], batch_size: int, max_length: int
    ) -> Dict[str, Any]:
        """Synchronous batch encoding for thread pool execution."""
        return self.model.encode(
            texts,
            batch_size=batch_size,
            max_length=max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )

    async def get_dense_embedding(self, text: str) -> List[float]:
        """Return a single dense embedding for the given text."""
        if not self._is_initialized:
            await self.initialize()

        try:
            if self.model is None:
                raise RuntimeError("Embedding model not initialized")

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self._executor,
                self._encode_batch_sync,
                [text],
                1,
                self.config.max_length,
            )
            vec = result.get("dense_vecs")
            if vec is None:
                vec = result.get("dense_vectors")
            # vec can be list-of-arrays or numpy array with shape (1, dim)
            if hasattr(vec, "shape") and hasattr(vec, "tolist"):
                # numpy array
                arr = vec
                if len(arr.shape) == 2 and arr.shape[0] >= 1:
                    return arr[0].tolist()
                return arr.tolist()
            if isinstance(vec, list) and vec:
                dense = vec[0]
                return dense.tolist() if hasattr(dense, "tolist") else list(dense)
            raise RuntimeError("Dense embedding not returned by model")
        except Exception as e:
            logger.error(f"Failed to get dense embedding: {e}")
            raise

    async def get_sparse_embedding(
        self, text: str
    ) -> Dict[str, float] | Dict[int, float]:
        """Return a single sparse embedding (lexical weights) for the given text.

        Returns either token->weight mapping or index->weight depending on model output.
        """
        if not self._is_initialized:
            await self.initialize()

        try:
            if self.model is None:
                raise RuntimeError("Embedding model not initialized")

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self._executor,
                lambda: self.model.encode(
                    [text],
                    batch_size=1,
                    max_length=self.config.max_length,
                    return_dense=False,
                    return_sparse=True,
                    return_colbert_vecs=False,
                ),
            )

            # Support multiple return shapes used in tests and real model
            if "lexical_weights" in result:
                weights_list = result["lexical_weights"]
                if isinstance(weights_list, list) and weights_list:
                    return weights_list[0]
            if "sparse_vecs" in result:
                sparse = result["sparse_vecs"][0]
                return self._convert_sparse_embedding(sparse)

            # Fallback to empty
            return {}
        except Exception as e:
            logger.error(f"Failed to get sparse embedding: {e}")
            raise

    async def encode_query(
        self, query: str, instruction: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Encode a query to get dense and sparse embeddings.

        Args:
            query: Query text to encode
            instruction: Optional instruction prefix for the query

        Returns:
            Dictionary containing dense and sparse embeddings
        """
        if not self._is_initialized:
            await self.initialize()

        try:
            # Add instruction if provided
            if instruction:
                query_text = f"{instruction} {query}"
            else:
                query_text = query

            # Encode query in thread pool
            if self.model is None:
                raise RuntimeError("Embedding model not initialized")

            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                self._executor,
                lambda: self.model.encode(
                    [query_text],
                    batch_size=1,
                    max_length=self.config.max_length,
                    return_dense=True,
                    return_sparse=True,
                    return_colbert_vecs=False,
                ),
            )

            dense_embedding = results.get("dense_vecs")[0]
            # 兼容无 sparse_vecs 的形态（如仅返回 lexical_weights 或未返回稀疏向量）
            if "sparse_vecs" in results and results["sparse_vecs"]:
                sparse_embedding = results["sparse_vecs"][0]
            else:
                # 若仅有 lexical_weights，暂不转换索引，Weaviate 路径不使用该值
                sparse_embedding = {}

            # Normalize dense embedding if configured
            if self.config.normalize_embeddings:
                dense_embedding = dense_embedding / np.linalg.norm(dense_embedding)

            # Convert sparse embedding
            sparse_dict = (
                self._convert_sparse_embedding(sparse_embedding)
                if isinstance(sparse_embedding, (list, dict))
                or hasattr(sparse_embedding, "indices")
                else {}
            )

            logger.debug(f"Encoded query: {query[:100]}...")

            return {
                "query": query,
                "dense_embedding": dense_embedding.tolist(),
                "sparse_embedding": sparse_dict,
                "embedding_dim": len(dense_embedding),
            }

        except Exception as e:
            logger.error(f"Failed to encode query: {e}")
            raise

    def _convert_sparse_embedding(self, sparse_array: Any) -> Dict[int, float]:
        """Convert sparse embedding array to dictionary format."""
        if hasattr(sparse_array, "indices") and hasattr(sparse_array, "values"):
            # Handle scipy sparse format
            return {
                int(idx): float(val)
                for idx, val in zip(sparse_array.indices, sparse_array.values)
                if val != 0
            }
        elif isinstance(sparse_array, dict):
            # Already in dictionary format
            return {int(k): float(v) for k, v in sparse_array.items() if v != 0}
        elif isinstance(sparse_array, (list, np.ndarray)):
            # Dense array format, convert to sparse
            return {
                int(i): float(val) for i, val in enumerate(sparse_array) if val != 0
            }
        else:
            logger.warning(f"Unknown sparse embedding format: {type(sparse_array)}")
            return {}

    async def compute_similarity(
        self,
        query_embedding: Dict[str, Any],
        doc_embeddings: List[Dict[str, Any]],
        method: str = "hybrid",
        alpha: float = 0.5,
    ) -> List[float]:
        """
        Compute similarity scores between query and document embeddings.

        Args:
            query_embedding: Query embedding dictionary
            doc_embeddings: List of document embedding dictionaries
            method: Similarity method ("dense", "sparse", "hybrid")
            alpha: Weight for dense similarity in hybrid mode

        Returns:
            List of similarity scores
        """
        try:
            scores = []

            query_dense = np.array(query_embedding["dense_embedding"])
            query_sparse = query_embedding["sparse_embedding"]

            for doc_emb in doc_embeddings:
                doc_dense = np.array(doc_emb["dense_embedding"])
                doc_sparse = doc_emb["sparse_embedding"]

                if method == "dense":
                    # Cosine similarity for dense vectors
                    score = np.dot(query_dense, doc_dense)
                elif method == "sparse":
                    # Sparse similarity (overlap)
                    score = self._sparse_similarity(query_sparse, doc_sparse)
                elif method == "hybrid":
                    # Weighted combination
                    dense_sim = np.dot(query_dense, doc_dense)
                    sparse_sim = self._sparse_similarity(query_sparse, doc_sparse)
                    score = alpha * dense_sim + (1 - alpha) * sparse_sim
                else:
                    raise ValueError(f"Unknown similarity method: {method}")

                scores.append(float(score))

            return scores

        except Exception as e:
            logger.error(f"Failed to compute similarities: {e}")
            raise

    def _sparse_similarity(
        self, sparse1: Dict[int, float], sparse2: Dict[int, float]
    ) -> float:
        """Compute similarity between sparse embeddings (dot product)."""
        common_indices = set(sparse1.keys()) & set(sparse2.keys())

        if not common_indices:
            return 0.0

        similarity = sum(sparse1[idx] * sparse2[idx] for idx in common_indices)
        return similarity

    async def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if not self._is_initialized:
            await self.initialize()

        return {
            "model_name": self.config.model,
            "embedding_dim": self.config.dimension,
            "max_length": self.config.max_length,
            "device": self.device,
            "normalize_embeddings": self.config.normalize_embeddings,
            "supports_sparse": True,
            "supports_dense": True,
            "batch_size": self.config.batch_size,
        }

    async def cleanup(self) -> None:
        """Clean up model resources."""
        try:
            if self.model is not None:
                # Clear CUDA cache if using GPU
                if hasattr(torch, "cuda") and torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Explicitly delete model as suggested by FlagEmbedding developers
                del self.model
                self.model = None
                self._is_initialized = False

            # Shutdown thread pool with timeout (restore original logic)
            if hasattr(self, "_executor") and self._executor is not None:
                logger.info("Shutting down BGE embedding thread pool...")
                # Use original non-blocking approach with timeout
                self._executor.shutdown(wait=False)

                # Give it a moment to shutdown gracefully
                import time

                start_time = time.time()
                while not self._executor._shutdown and (time.time() - start_time) < 2.0:
                    await asyncio.sleep(0.1)  # Use async sleep in async context

                if not self._executor._shutdown:
                    logger.warning(
                        "BGE embedding thread pool did not shutdown gracefully"
                    )

            logger.info("BGE embedding service cleaned up")

        except Exception as e:
            logger.warning(f"Error during BGE embedding service cleanup: {e}")
            # Don't re-raise to avoid blocking shutdown


class EmbeddingManager:
    """Manager for embedding operations with caching and batch processing."""

    def __init__(self, embedding_service: BGEEmbeddingService):
        self.embedding_service = embedding_service
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_size_limit = 10000  # Maximum cached embeddings

    async def get_document_embeddings(
        self,
        documents: List[str],
        use_cache: bool = True,
        batch_size: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Get embeddings for documents with optional caching.

        Uses per-document service methods to allow easy mocking in tests.
        """
        results: List[Optional[Dict[str, Any]]] = [None] * len(documents)

        # Resolve which docs to compute
        to_compute: list[tuple[int, str]] = []
        for i, doc in enumerate(documents):
            if not use_cache:
                to_compute.append((i, doc))
                continue
            doc_hash = self._hash_text(doc)
            cached = self._cache.get(doc_hash)
            if cached is not None:
                results[i] = cached
            else:
                to_compute.append((i, doc))

        # Compute embeddings for pending docs (sequential to simplify tests)
        for idx, text in to_compute:
            dense = await self.embedding_service.get_dense_embedding(text)
            sparse = await self.embedding_service.get_sparse_embedding(text)
            emb = {
                "text": text,
                "dense_embedding": dense,
                "sparse_embedding": sparse,
                "embedding_dim": len(dense),
            }
            if use_cache and len(self._cache) < self._cache_size_limit:
                self._cache[self._hash_text(text)] = emb
            results[idx] = emb

        # Type ignore: we ensure all filled above
        valid_results: List[Dict[str, Any]] = [r for r in results if r is not None]
        return valid_results

    async def get_query_embedding(
        self, query: str, instruction: Optional[str] = None, use_cache: bool = True
    ) -> Dict[str, Any]:
        """Get embedding for query with optional caching."""
        if not use_cache:
            return await self.embedding_service.encode_query(query, instruction)

        # Check cache
        cache_key = f"query: {instruction or ''}: {query}"
        cache_hash = self._hash_text(cache_key)

        if cache_hash in self._cache:
            return self._cache[cache_hash]

        # Generate embedding
        embedding = await self.embedding_service.encode_query(query, instruction)

        # Add to cache if not at limit
        if len(self._cache) < self._cache_size_limit:
            self._cache[cache_hash] = embedding

        return embedding

    def _hash_text(self, text: str) -> str:
        """Generate hash for text caching."""
        import hashlib

        return hashlib.md5(text.encode(), usedforsecurity=False).hexdigest()

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._cache.clear()
        logger.info("Embedding cache cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "cache_size": len(self._cache),
            "cache_limit": self._cache_size_limit,
            "cache_usage": len(self._cache) / self._cache_size_limit,
        }
