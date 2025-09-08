"""
BGE-M3 embedding service with sparse and dense vector generation.
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple, Union
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from FlagEmbedding import BGEM3FlagModel

from askme.core.config import EmbeddingConfig


logger = logging.getLogger(__name__)


class BGEEmbeddingService:
    """BGE-M3 embedding service supporting dense, sparse, and multi-vector representations."""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._is_initialized = False
        
    async def initialize(self) -> None:
        """Initialize the BGE-M3 model."""
        if self._is_initialized:
            return
        
        try:
            logger.info(f"Loading BGE-M3 model: {self.config.model}")
            
            # Load BGE-M3 model
            self.model = BGEM3FlagModel(
                self.config.model,
                use_fp16=self.config.use_fp16,
                device=self.device
            )
            
            self._is_initialized = True
            logger.info(f"BGE-M3 model loaded on device: {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to initialize BGE-M3 model: {e}")
            raise
    
    async def encode_documents(
        self, 
        texts: List[str],
        batch_size: Optional[int] = None
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
        
        batch_size = batch_size or self.config.batch_size
        
        try:
            # Process in batches to manage memory
            results = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                logger.debug(f"Processing batch {i//batch_size + 1}: {len(batch_texts)} texts")
                
                # Encode batch with BGE-M3
                batch_results = self.model.encode(
                    batch_texts,
                    batch_size=len(batch_texts),
                    max_length=self.config.max_length,
                    return_dense=True,
                    return_sparse=True,
                    return_colbert_vecs=False  # Skip ColBERT vectors for now
                )
                
                # Process results
                for j, text in enumerate(batch_texts):
                    dense_embedding = batch_results['dense_vecs'][j]
                    sparse_embedding = batch_results['sparse_vecs'][j]
                    
                    # Normalize dense embedding if configured
                    if self.config.normalize_embeddings:
                        dense_embedding = dense_embedding / np.linalg.norm(dense_embedding)
                    
                    # Convert sparse embedding to dictionary format
                    sparse_dict = self._convert_sparse_embedding(sparse_embedding)
                    
                    results.append({
                        'text': text,
                        'dense_embedding': dense_embedding.tolist(),
                        'sparse_embedding': sparse_dict,
                        'embedding_dim': len(dense_embedding)
                    })
            
            logger.info(f"Encoded {len(texts)} documents successfully")
            return results
            
        except Exception as e:
            logger.error(f"Failed to encode documents: {e}")
            raise
    
    async def encode_query(
        self, 
        query: str,
        instruction: Optional[str] = None
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
            
            # Encode query
            results = self.model.encode(
                [query_text],
                batch_size=1,
                max_length=self.config.max_length,
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=False
            )
            
            dense_embedding = results['dense_vecs'][0]
            sparse_embedding = results['sparse_vecs'][0]
            
            # Normalize dense embedding if configured
            if self.config.normalize_embeddings:
                dense_embedding = dense_embedding / np.linalg.norm(dense_embedding)
            
            # Convert sparse embedding
            sparse_dict = self._convert_sparse_embedding(sparse_embedding)
            
            logger.debug(f"Encoded query: {query[:100]}...")
            
            return {
                'query': query,
                'dense_embedding': dense_embedding.tolist(),
                'sparse_embedding': sparse_dict,
                'embedding_dim': len(dense_embedding)
            }
            
        except Exception as e:
            logger.error(f"Failed to encode query: {e}")
            raise
    
    def _convert_sparse_embedding(self, sparse_array) -> Dict[int, float]:
        """Convert sparse embedding array to dictionary format."""
        if hasattr(sparse_array, 'indices') and hasattr(sparse_array, 'values'):
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
                int(i): float(val) 
                for i, val in enumerate(sparse_array) 
                if val != 0
            }
        else:
            logger.warning(f"Unknown sparse embedding format: {type(sparse_array)}")
            return {}
    
    async def compute_similarity(
        self,
        query_embedding: Dict[str, Any],
        doc_embeddings: List[Dict[str, Any]],
        method: str = "hybrid",
        alpha: float = 0.5
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
            
            query_dense = np.array(query_embedding['dense_embedding'])
            query_sparse = query_embedding['sparse_embedding']
            
            for doc_emb in doc_embeddings:
                doc_dense = np.array(doc_emb['dense_embedding'])
                doc_sparse = doc_emb['sparse_embedding']
                
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
    
    def _sparse_similarity(self, sparse1: Dict[int, float], sparse2: Dict[int, float]) -> float:
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
            "batch_size": self.config.batch_size
        }
    
    async def cleanup(self) -> None:
        """Clean up model resources."""
        if self.model is not None:
            # Clear CUDA cache if using GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.model = None
            self._is_initialized = False
            logger.info("BGE embedding service cleaned up")


class EmbeddingManager:
    """Manager for embedding operations with caching and batch processing."""
    
    def __init__(self, embedding_service: BGEEmbeddingService):
        self.embedding_service = embedding_service
        self._cache = {}
        self._cache_size_limit = 10000  # Maximum cached embeddings
    
    async def get_document_embeddings(
        self,
        documents: List[str],
        use_cache: bool = True,
        batch_size: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get embeddings for documents with optional caching.
        
        Args:
            documents: List of document texts
            use_cache: Whether to use embedding cache
            batch_size: Batch size for processing
            
        Returns:
            List of embedding dictionaries
        """
        if not use_cache:
            return await self.embedding_service.encode_documents(documents, batch_size)
        
        # Check cache for existing embeddings
        uncached_docs = []
        uncached_indices = []
        results = [None] * len(documents)
        
        for i, doc in enumerate(documents):
            doc_hash = self._hash_text(doc)
            if doc_hash in self._cache:
                results[i] = self._cache[doc_hash]
            else:
                uncached_docs.append(doc)
                uncached_indices.append(i)
        
        # Process uncached documents
        if uncached_docs:
            logger.debug(f"Processing {len(uncached_docs)} uncached documents")
            
            new_embeddings = await self.embedding_service.encode_documents(
                uncached_docs, batch_size
            )
            
            # Update cache and results
            for i, embedding in enumerate(new_embeddings):
                doc_idx = uncached_indices[i]
                doc = uncached_docs[i]
                doc_hash = self._hash_text(doc)
                
                # Add to cache if not at limit
                if len(self._cache) < self._cache_size_limit:
                    self._cache[doc_hash] = embedding
                
                results[doc_idx] = embedding
        
        return results
    
    async def get_query_embedding(
        self,
        query: str,
        instruction: Optional[str] = None,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """Get embedding for query with optional caching."""
        if not use_cache:
            return await self.embedding_service.encode_query(query, instruction)
        
        # Check cache
        cache_key = f"query:{instruction or ''}:{query}"
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
        return hashlib.md5(text.encode()).hexdigest()
    
    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._cache.clear()
        logger.info("Embedding cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "cache_size": len(self._cache),
            "cache_limit": self._cache_size_limit,
            "cache_usage": len(self._cache) / self._cache_size_limit
        }