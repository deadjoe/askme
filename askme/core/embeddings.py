"""
BGE-M3 embedding service with sparse and dense vector generation.
"""

import asyncio
import importlib
import logging
import warnings
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np

from askme.core.config import EmbeddingConfig

# Suppress tokenizer performance warnings
warnings.filterwarnings(
    "ignore", message=".*XLMRobertaTokenizerFast.*", category=UserWarning
)

torch_module: ModuleType | None
try:
    torch_module = importlib.import_module("torch")
except Exception:  # pragma: no cover
    torch_module = None

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
        self.model: Any | None = None
        # Allow operation without torch installed (skip GPU detection)
        self.device = "cpu"  # fallback default
        if torch_module is not None:
            # Check for NVIDIA CUDA (preferred for BGE-M3 stability)
            if hasattr(torch_module, "cuda"):
                cuda_mod = getattr(torch_module, "cuda")
                is_available = getattr(cuda_mod, "is_available", None)
                if callable(is_available) and is_available():
                    self.device = "cuda"
            # Check for Apple Silicon MPS (known memory leak issues with BGE-M3)
            if (
                self.device == "cpu"
                and hasattr(torch_module, "backends")
                and hasattr(torch_module.backends, "mps")
            ):
                mps_mod = getattr(torch_module.backends, "mps")
                is_available = getattr(mps_mod, "is_available", None)
                if callable(is_available) and is_available():
                    # Use MPS but with memory management warnings
                    self.device = "mps"
                    logger.warning(
                        "Using MPS backend for BGE-M3 - known memory leak issues. "
                        "Monitor memory usage, consider CPU for large datasets."
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
                devices=self.device,
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

                # Debug: Log available keys in batch_results
                logger.info(f"BGE-M3 batch_results keys: {list(batch_results.keys())}")

                # Process results
                for j, text in enumerate(batch_texts):
                    dense_embedding = batch_results["dense_vecs"][j]

                    # Handle sparse vectors with fallback
                    if "sparse_vecs" in batch_results:
                        sparse_embedding = batch_results["sparse_vecs"][j]
                    elif "lexical_weights" in batch_results:
                        # Alternative key name for sparse vectors in BGE-M3 versions
                        sparse_embedding = batch_results["lexical_weights"][j]
                    else:
                        # Fallback: create empty sparse embedding
                        logger.warning(
                            f"No sparse vectors found in BGE-M3 output, "
                            f"available keys: {list(batch_results.keys())}"
                        )
                        sparse_embedding = []

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

    def _require_model(self) -> Any:
        """Ensure built-in model is initialized and return instance."""
        if self.model is None:
            raise RuntimeError("Embedding model not initialized")
        return self.model

    def _mps_memory_cleanup(self, force: bool = False) -> None:
        """
        Perform selective memory cleanup for MPS backend to mitigate known memory leaks.

        BGE-M3 has known memory leak issues on Apple Silicon MPS backend.
        This method attempts to mitigate the issue with minimal GPU utilization impact.
        """
        try:
            if torch_module is not None and hasattr(torch_module, "mps"):
                # Only perform cleanup every N operations or when forced
                if not hasattr(self, "_mps_cleanup_counter"):
                    self._mps_cleanup_counter = 0

                self._mps_cleanup_counter += 1

                # Cleanup every 10 operations or when forced
                if force or self._mps_cleanup_counter % 10 == 0:
                    import gc

                    # Lightweight garbage collection
                    gc.collect()

                    # Clear MPS cache if available (less frequent)
                    if hasattr(torch_module.mps, "empty_cache"):
                        torch_module.mps.empty_cache()

                    logger.debug(
                        f"MPS memory cleanup completed (cycle: "
                        f"{self._mps_cleanup_counter})"
                    )

        except Exception as e:
            logger.warning(f"MPS memory cleanup failed: {e}")

    def _encode_batch_sync(
        self, texts: List[str], batch_size: int, max_length: int
    ) -> Dict[str, Any]:
        """Synchronous batch encoding for thread pool execution."""
        model = self._require_model()

        # Suppress tokenizer warnings during encoding
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*fast tokenizer.*__call__.*method.*faster.*",
                category=UserWarning,
            )
            warnings.filterwarnings(
                "ignore", message=".*XLMRobertaTokenizerFast.*", category=UserWarning
            )
            result = cast(
                Dict[str, Any],
                model.encode(
                    texts,
                    batch_size=batch_size,
                    max_length=max_length,
                    return_dense=True,
                    return_sparse=True,
                ),
            )

            # Memory management for MPS backend (known memory leak mitigation)
            if self.device == "mps":
                self._mps_memory_cleanup()

            return result

    def _encode_single_batch(self, text: str, max_length: int) -> Dict[str, Any]:
        model = self._require_model()

        # Suppress tokenizer warnings during encoding
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*fast tokenizer.*__call__.*method.*faster.*",
                category=UserWarning,
            )
            warnings.filterwarnings(
                "ignore", message=".*XLMRobertaTokenizerFast.*", category=UserWarning
            )
            result = cast(
                Dict[str, Any],
                model.encode(
                    [text],
                    batch_size=1,
                    max_length=max_length,
                    return_dense=True,
                    return_sparse=True,
                ),
            )

            # Memory management for MPS backend
            if self.device == "mps":
                self._mps_memory_cleanup()

            return result

    def _encode_sparse_single(self, text: str, max_length: int) -> Dict[str, Any]:
        model = self._require_model()

        # Suppress tokenizer warnings during encoding
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*fast tokenizer.*__call__.*method.*faster.*",
                category=UserWarning,
            )
            warnings.filterwarnings(
                "ignore", message=".*XLMRobertaTokenizerFast.*", category=UserWarning
            )
            result = cast(
                Dict[str, Any],
                model.encode(
                    [text],
                    batch_size=1,
                    max_length=max_length,
                    return_dense=False,
                    return_sparse=True,
                ),
            )

            # Memory management for MPS backend
            if self.device == "mps":
                self._mps_memory_cleanup()

            return result

    def _encode_query_single(self, text: str, max_length: int) -> Dict[str, Any]:
        model = self._require_model()

        # Suppress tokenizer warnings during encoding
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*fast tokenizer.*__call__.*method.*faster.*",
                category=UserWarning,
            )
            warnings.filterwarnings(
                "ignore", message=".*XLMRobertaTokenizerFast.*", category=UserWarning
            )
            result = cast(
                Dict[str, Any],
                model.encode(
                    [text],
                    batch_size=1,
                    max_length=max_length,
                    return_dense=True,
                    return_sparse=True,
                ),
            )

            # Memory management for MPS backend
            if self.device == "mps":
                self._mps_memory_cleanup()

            return result

    def _extract_dense_vector(self, result: Dict[str, Any]) -> List[float]:
        """Extract 1D dense embedding from model results."""
        vec_data = result.get("dense_vecs")
        if vec_data is None:
            vec_data = result.get("dense_vectors")
        if vec_data is None:
            raise RuntimeError("Dense embedding not returned by model")

        if hasattr(vec_data, "shape") and hasattr(vec_data, "tolist"):
            arr = cast(Any, vec_data)
            if len(getattr(arr, "shape", [])) == 2 and arr.shape[0] >= 1:
                dense_values = arr[0].tolist()
            else:
                dense_values = arr.tolist()
            return [float(x) for x in list(dense_values)]

        if isinstance(vec_data, list) and vec_data:
            dense_item = vec_data[0]
            if hasattr(dense_item, "tolist"):
                dense_values = cast(Any, dense_item).tolist()
            else:
                dense_values = dense_item
            return [float(x) for x in list(dense_values)]

        raise RuntimeError("Dense embedding not returned by model")

    async def get_dense_embedding(self, text: str) -> List[float]:
        """Return a single dense embedding for the given text."""
        if not self._is_initialized:
            await self.initialize()

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self._executor,
                self._encode_single_batch,
                text,
                self.config.max_length,
            )
            return self._extract_dense_vector(result)
        except Exception as e:
            logger.error(f"Failed to get dense embedding: {e}")
            raise

    async def get_sparse_embedding(self, text: str) -> Dict[int | str, float]:
        """Return a single sparse embedding (lexical weights) for the given text.

        Returns either token->weight mapping or index->weight depending on model output.
        """
        if not self._is_initialized:
            await self.initialize()

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self._executor,
                self._encode_sparse_single,
                text,
                self.config.max_length,
            )

            # Support multiple return shapes used in tests and real model
            if "lexical_weights" in result:
                weights_list = result["lexical_weights"]
                if isinstance(weights_list, list) and weights_list:
                    raw_weights = weights_list[0]
                    if isinstance(raw_weights, dict):
                        return {
                            str(k): float(v)
                            for k, v in raw_weights.items()
                            if float(v) != 0.0
                        }
            if "sparse_vecs" in result:
                sparse_vecs = result["sparse_vecs"]
                if isinstance(sparse_vecs, list) and sparse_vecs:
                    sparse = sparse_vecs[0]
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
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                self._executor,
                self._encode_query_single,
                query_text,
                self.config.max_length,
            )

            dense_values = self._extract_dense_vector(results)
            dense_array = np.array(dense_values, dtype=float)

            # Handle cases without sparse_vecs (lexical_weights only or no sparse)
            sparse_embedding_raw: Any = {}
            sparse_vecs = results.get("sparse_vecs")
            if isinstance(sparse_vecs, list) and sparse_vecs:
                sparse_embedding_raw = sparse_vecs[0]
            else:
                lexical_weights = results.get("lexical_weights")
                if isinstance(lexical_weights, list) and lexical_weights:
                    sparse_embedding_raw = lexical_weights[0]

            # Normalize dense embedding if configured
            if self.config.normalize_embeddings:
                norm = float(np.linalg.norm(dense_array))
                if norm > 0:
                    dense_array = dense_array / norm

            dense_list = dense_array.tolist()

            # Convert sparse embedding
            sparse_dict = (
                self._convert_sparse_embedding(sparse_embedding_raw)
                if isinstance(sparse_embedding_raw, (list, dict))
                or hasattr(sparse_embedding_raw, "indices")
                else {}
            )

            logger.debug(f"Encoded query: {query[:100]}...")

            return {
                "query": query,
                "dense_embedding": dense_list,
                "sparse_embedding": sparse_dict,
                "embedding_dim": len(dense_list),
            }

        except Exception as e:
            logger.error(f"Failed to encode query: {e}")
            raise

    def _convert_sparse_embedding(self, sparse_array: Any) -> Dict[int | str, float]:
        """Convert sparse embedding array to dictionary format."""
        if hasattr(sparse_array, "indices") and hasattr(sparse_array, "values"):
            # Handle scipy sparse format
            values = zip(sparse_array.indices, sparse_array.values)
            return {int(idx): float(val) for idx, val in values if val != 0}

        if isinstance(sparse_array, dict):
            converted: Dict[int | str, float] = {}
            for key, value in sparse_array.items():
                try:
                    numeric_key = int(key)
                except (TypeError, ValueError):
                    converted[str(key)] = float(value)
                else:
                    converted[numeric_key] = float(value)

            return {k: v for k, v in converted.items() if v != 0}
        if isinstance(sparse_array, (list, np.ndarray)):
            return {
                int(i): float(val) for i, val in enumerate(sparse_array) if val != 0
            }

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
                if torch_module is not None and hasattr(torch_module, "cuda"):
                    cuda_mod = getattr(torch_module, "cuda")
                    is_available = getattr(cuda_mod, "is_available", None)
                    if callable(is_available) and is_available():
                        empty_cache = getattr(cuda_mod, "empty_cache", None)
                        if callable(empty_cache):
                            empty_cache()

                # Explicitly delete model as suggested by FlagEmbedding developers
                del self.model
                self.model = None
                self._is_initialized = False

            # Shutdown thread pool with timeout
            if hasattr(self, "_executor") and self._executor is not None:
                logger.info("Shutting down BGE embedding thread pool...")
                try:
                    # Try modern shutdown with timeout (Python 3.9+)
                    import inspect

                    shutdown_sig = inspect.signature(self._executor.shutdown)
                    if "timeout" in shutdown_sig.parameters:
                        self._executor.shutdown(wait=True, timeout=2.0)  # type: ignore
                        logger.info("BGE embedding thread pool shut down gracefully")
                    else:
                        # Fallback for older Python versions
                        self._executor.shutdown(wait=True)
                        logger.info(
                            "BGE embedding thread pool shut down (no timeout support)"
                        )
                except Exception as e:
                    logger.warning(f"Graceful shutdown failed: {e}, forcing shutdown")
                    try:
                        self._executor.shutdown(wait=False)
                    except Exception as force_e:
                        logger.error(
                            f"Failed to force shutdown BGE thread pool: {force_e}"
                        )
                finally:
                    # Mark as None to prevent further use
                    self._executor = None  # type: ignore

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
        """Get embeddings for documents with caching and true batch processing."""
        if not documents:
            return []

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

        if not to_compute:
            # All results from cache
            return [r for r in results if r is not None]

        # Extract texts to compute and their indices
        indices_to_compute = [idx for idx, _ in to_compute]
        texts_to_compute = [text for _, text in to_compute]

        # Use true batch processing via encode_documents
        batch_embeddings = await self.embedding_service.encode_documents(
            texts_to_compute, batch_size=batch_size
        )

        # Update results and cache
        for i, embedding in enumerate(batch_embeddings):
            original_idx = indices_to_compute[i]
            results[original_idx] = embedding

            # Cache the result if caching enabled and under limit
            if use_cache and len(self._cache) < self._cache_size_limit:
                text = embedding["text"]
                self._cache[self._hash_text(text)] = embedding

        # Ensure all results are filled
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
