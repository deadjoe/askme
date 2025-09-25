"""
Base classes for vector database retrievers.
"""

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Document:
    """Document representation for retrieval."""

    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    sparse_embedding: Optional[Dict[int, float]] = None


@dataclass
class RetrievalResult:
    """Single retrieval result with scoring information."""

    document: Document
    score: float
    rank: int
    retrieval_method: str  # "dense", "sparse", "hybrid"
    debug_info: Optional[Dict[str, Any]] = None


@dataclass
class HybridSearchParams:
    """Parameters for hybrid search configuration."""

    alpha: float = 0.5  # 0=sparse only, 1=dense only
    use_rrf: bool = True  # Use reciprocal rank fusion
    rrf_k: int = 60  # RRF parameter
    topk: int = 50  # Number of candidates to retrieve
    filters: Optional[Dict[str, Any]] = None
    original_query: Optional[
        str
    ] = None  # For backends needing raw text (e.g., Weaviate)


class VectorRetriever(ABC):
    """Abstract base class for vector database retrievers."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.collection_name = config.get("collection_name", "askme_default")

    @property
    def supports_native_upsert(self) -> bool:
        """Whether this retriever supports native upsert operations.

        If True, insert_documents() will automatically overwrite existing documents
        with the same ID without requiring explicit deletion.
        """
        return False

    @abstractmethod
    async def connect(self) -> None:
        """Connect to the vector database."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the vector database."""
        pass

    @abstractmethod
    async def create_collection(
        self, dimension: int, metric: str = "cosine", **kwargs: Any
    ) -> None:
        """Create a collection/index for documents."""
        pass

    @abstractmethod
    async def insert_documents(self, documents: List[Document]) -> List[str]:
        """Insert documents into the vector database."""
        pass

    @abstractmethod
    async def dense_search(
        self,
        query_embedding: List[float],
        topk: int = 50,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalResult]:
        """Perform dense vector search."""
        pass

    @abstractmethod
    async def sparse_search(
        self,
        query_terms: Dict[int, float],  # or query string depending on implementation
        topk: int = 50,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalResult]:
        """Perform sparse/BM25 search."""
        pass

    @abstractmethod
    async def hybrid_search(
        self,
        query_embedding: List[float],
        query_terms: Dict[int, float],
        params: HybridSearchParams,
    ) -> List[RetrievalResult]:
        """Perform hybrid search combining dense and sparse."""
        pass

    @abstractmethod
    async def get_document(self, doc_id: str) -> Optional[Document]:
        """Get a document by ID."""
        pass

    @abstractmethod
    async def delete_document(self, doc_id: str) -> bool:
        """Delete a document by ID."""
        pass

    @abstractmethod
    async def update_document(self, doc_id: str, document: Document) -> bool:
        """Update an existing document."""
        pass

    @abstractmethod
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        pass


class SearchFusion:
    """Utilities for fusing search results from multiple sources."""

    @staticmethod
    def reciprocal_rank_fusion(
        results_lists: List[List[RetrievalResult]], k: int = 60
    ) -> List[RetrievalResult]:
        """
        Combine multiple ranked lists using Reciprocal Rank Fusion.

        Args:
            results_lists: List of ranked result lists to combine
            k: RRF parameter (typical values: 20-100)

        Returns:
            Fused and re-ranked results
        """
        doc_scores = {}
        doc_objects = {}

        for results in results_lists:
            for rank, result in enumerate(results, 1):
                doc_id = result.document.id
                doc_objects[doc_id] = result.document

                if doc_id not in doc_scores:
                    doc_scores[doc_id] = 0.0

                doc_scores[doc_id] += 1.0 / (k + rank)

        # Sort by fused score
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        # Create fused results
        fused_results = []
        for rank, (doc_id, score) in enumerate(sorted_docs):
            fused_results.append(
                RetrievalResult(
                    document=doc_objects[doc_id],
                    score=score,
                    rank=rank + 1,
                    retrieval_method="rrf_fusion",
                    debug_info={
                        "rrf_score": score,
                        "rrf_k": k,
                        "num_sources": len(results_lists),
                    },
                )
            )

        return fused_results

    @staticmethod
    def alpha_fusion(
        dense_results: List[RetrievalResult],
        sparse_results: List[RetrievalResult],
        alpha: float = 0.5,
    ) -> List[RetrievalResult]:
        """
        Combine dense and sparse results using alpha weighting.

        Args:
            dense_results: Results from dense search
            sparse_results: Results from sparse search
            alpha: Weight for dense results (1-alpha for sparse)

        Returns:
            Alpha-weighted fused results
        """

        # Normalize scores to [0, 1] range
        def normalize_scores(results: List[RetrievalResult]) -> List[RetrievalResult]:
            if not results:
                return results

            scores = [r.score for r in results]
            min_score, max_score = min(scores), max(scores)

            if max_score == min_score:
                # All scores are the same
                for result in results:
                    result.score = 1.0
                return results

            normalized = []
            for result in results:
                normalized_score = (result.score - min_score) / (max_score - min_score)
                normalized.append(
                    RetrievalResult(
                        document=result.document,
                        score=normalized_score,
                        rank=result.rank,
                        retrieval_method=result.retrieval_method,
                        debug_info=result.debug_info,
                    )
                )
            return normalized

        # Normalize both result sets
        dense_norm = normalize_scores(dense_results)
        sparse_norm = normalize_scores(sparse_results)

        # Create combined score mapping
        doc_scores = {}
        doc_objects = {}

        # Add dense scores
        for result in dense_norm:
            doc_id = result.document.id
            doc_objects[doc_id] = result.document
            doc_scores[doc_id] = alpha * result.score

        # Add sparse scores
        for result in sparse_norm:
            doc_id = result.document.id
            doc_objects[doc_id] = result.document
            if doc_id in doc_scores:
                doc_scores[doc_id] += (1 - alpha) * result.score
            else:
                doc_scores[doc_id] = (1 - alpha) * result.score

        # Sort by combined score
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        # Create fused results
        fused_results = []
        for rank, (doc_id, score) in enumerate(sorted_docs):
            fused_results.append(
                RetrievalResult(
                    document=doc_objects[doc_id],
                    score=score,
                    rank=rank + 1,
                    retrieval_method="alpha_fusion",
                    debug_info={"alpha": alpha, "combined_score": score},
                )
            )

        return fused_results
