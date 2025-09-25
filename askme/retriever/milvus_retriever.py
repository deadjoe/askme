"""
Milvus 2.5+ retriever implementation with hybrid search support.
"""

import inspect
import json
import logging
from typing import Any, Dict, List, Optional

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

# Try to import newer features for Milvus 2.5+
try:
    from pymilvus import AnnSearchRequest, RRFRanker

    HYBRID_SEARCH_AVAILABLE = True
except ImportError:
    HYBRID_SEARCH_AVAILABLE = False

try:
    _HYBRID_PARAMS = inspect.signature(Collection.hybrid_search).parameters
    HYBRID_SEARCH_USES_RERANK = "rerank" in _HYBRID_PARAMS
except (AttributeError, TypeError, ValueError):
    HYBRID_SEARCH_USES_RERANK = False

from .base import (
    Document,
    HybridSearchParams,
    RetrievalResult,
    SearchFusion,
    VectorRetriever,
)

logger = logging.getLogger(__name__)


class MilvusRetriever(VectorRetriever):
    """Milvus 2.5+ retriever with hybrid search capabilities."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.host = config.get("host", "localhost")
        self.port = config.get("port", 19530)
        self.username = config.get("username", "")
        self.password = config.get("password", "")
        self.secure = config.get("secure", False)

        self.connection_name = f"askme_conn_{hash(f'{self.host}: {self.port}')}"
        self.collection: Optional[Collection] = None
        self.dimension = config.get("dimension", 1024)

    @property
    def supports_native_upsert(self) -> bool:
        """Milvus 2.6+ supports native upsert operations."""
        return True

    async def connect(self) -> None:
        """Connect to Milvus server."""
        try:
            connections.connect(
                alias=self.connection_name,
                host=self.host,
                port=self.port,
                user=self.username,
                password=self.password,
                secure=self.secure,
            )
            logger.info(f"Connected to Milvus at {self.host}: {self.port}")

            # Try to load existing collection
            if utility.has_collection(self.collection_name, using=self.connection_name):
                self.collection = Collection(
                    name=self.collection_name, using=self.connection_name
                )
                self.collection.load()
                logger.info(f"Loaded existing collection: {self.collection_name}")

        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            raise

    async def disconnect(self) -> None:
        """Disconnect from Milvus."""
        try:
            if self.collection:
                self.collection.release()
            connections.disconnect(alias=self.connection_name)
            logger.info("Disconnected from Milvus")
        except Exception as e:
            logger.warning(f"Error disconnecting from Milvus: {e}")

    async def create_collection(
        self, dimension: int, metric: str = "cosine", **kwargs: Any
    ) -> None:
        """Create a new collection with hybrid search support."""
        try:
            if utility.has_collection(self.collection_name, using=self.connection_name):
                logger.info(f"Collection {self.collection_name} already exists")
                return

            # Define collection schema with support for dense, sparse, and metadata
            fields = [
                FieldSchema(
                    name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64
                ),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(
                    name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=dimension
                ),
                FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
                FieldSchema(name="metadata", dtype=DataType.JSON),
            ]

            schema = CollectionSchema(
                fields=fields,
                description="askme hybrid RAG collection with dense and sparse vectors",
                enable_dynamic_field=True,
            )

            # Create collection
            collection = Collection(
                name=self.collection_name, schema=schema, using=self.connection_name
            )

            # Create indexes
            # Dense vector index
            dense_index_params = {
                "metric_type": metric.upper(),
                "index_type": "HNSW",
                "params": {"M": 16, "efConstruction": 256},
            }
            collection.create_index(
                field_name="dense_vector",
                index_params=dense_index_params,
                index_name="dense_idx",
            )

            # Sparse vector index for BM25 search; metric is specified at query time
            sparse_index_params = {
                "index_type": "SPARSE_INVERTED_INDEX",
                "metric_type": "IP",
                "params": {},
            }
            collection.create_index(
                field_name="sparse_vector",
                index_params=sparse_index_params,
                index_name="sparse_idx",
            )

            self.collection = collection
            self.collection.load()

            logger.info(
                f"Created collection {self.collection_name} with hybrid search support"
            )

        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            raise

    async def insert_documents(self, documents: List[Document]) -> List[str]:
        """Insert documents with dense and sparse vectors."""
        if not self.collection:
            raise RuntimeError(
                "Collection not initialized. Call create_collection first."
            )

        try:
            # Prepare data for insertion
            data = []
            for doc in documents:
                if not doc.embedding:
                    raise ValueError(f"Document {doc.id} missing dense embedding")
                if not doc.sparse_embedding:
                    raise ValueError(f"Document {doc.id} missing sparse embedding")

                data.append(
                    {
                        "id": doc.id,
                        "content": doc.content,
                        "dense_vector": doc.embedding,
                        "sparse_vector": doc.sparse_embedding,
                        "metadata": doc.metadata,
                    }
                )

            # Use upsert for native overwrite support (Milvus 2.6+)
            result = self.collection.upsert(data)

            # Flush to ensure data is written
            self.collection.flush()

            logger.info(
                f"Upserted {len(documents)} documents into {self.collection_name}"
            )
            # Ensure primary keys are returned as strings
            try:
                return [str(pk) for pk in result.primary_keys]
            except Exception:
                return []

        except Exception as e:
            logger.error(f"Failed to insert documents: {e}")
            raise

    async def dense_search(
        self,
        query_embedding: List[float],
        topk: int = 50,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalResult]:
        """Perform dense vector search."""
        if not self.collection:
            raise RuntimeError("Collection not initialized")

        try:
            search_params = {"metric_type": "COSINE", "params": {"ef": 256}}

            # Build filter expression if provided
            expr = self._build_filter_expression(filters) if filters else None

            results = self.collection.search(
                data=[query_embedding],
                anns_field="dense_vector",
                param=search_params,
                limit=topk,
                expr=expr,
                output_fields=["id", "content", "metadata"],
            )

            retrieval_results = []
            for rank, hit in enumerate(results[0]):
                doc = Document(
                    id=hit.entity.get("id"),
                    content=hit.entity.get("content"),
                    metadata=hit.entity.get("metadata", {}),
                )

                retrieval_results.append(
                    RetrievalResult(
                        document=doc,
                        score=hit.score,
                        rank=rank + 1,
                        retrieval_method="dense",
                        debug_info={"distance": hit.distance},
                    )
                )

            logger.debug(f"Dense search returned {len(retrieval_results)} results")
            return retrieval_results

        except Exception as e:
            logger.error(f"Dense search failed: {e}")
            raise

    async def sparse_search(
        self,
        query_terms: Dict[int, float],
        topk: int = 50,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalResult]:
        """Perform sparse/BM25 search using Milvus 2.5 BM25 function."""
        if not self.collection:
            raise RuntimeError("Collection not initialized")

        try:
            search_params = {"metric_type": "IP", "params": {}}

            # Build filter expression if provided
            expr = self._build_filter_expression(filters) if filters else None

            # Convert query terms to sparse vector format
            sparse_query = query_terms

            results = self.collection.search(
                data=[sparse_query],
                anns_field="sparse_vector",
                param=search_params,
                limit=topk,
                expr=expr,
                output_fields=["id", "content", "metadata"],
            )

            retrieval_results = []
            for rank, hit in enumerate(results[0]):
                doc = Document(
                    id=hit.entity.get("id"),
                    content=hit.entity.get("content"),
                    metadata=hit.entity.get("metadata", {}),
                )

                retrieval_results.append(
                    RetrievalResult(
                        document=doc,
                        score=hit.score,
                        rank=rank + 1,
                        retrieval_method="sparse",
                        debug_info={"bm25_score": hit.score},
                    )
                )

            logger.debug(f"Sparse search returned {len(retrieval_results)} results")
            return retrieval_results

        except Exception as e:
            logger.error(f"Sparse search failed: {e}")
            raise

    async def hybrid_search(
        self,
        query_embedding: List[float],
        query_terms: Dict[int, float],
        params: HybridSearchParams,
    ) -> List[RetrievalResult]:
        """Perform hybrid search using Milvus 2.5 hybrid capabilities."""
        if not self.collection:
            raise RuntimeError("Collection not initialized")

        try:
            # Build filter expression if provided
            expr = (
                self._build_filter_expression(params.filters)
                if params.filters
                else None
            )

            if params.use_rrf:
                # Use Milvus native RRF hybrid search
                return await self._hybrid_search_rrf(
                    query_embedding, query_terms, params, expr
                )
            else:
                # Use alpha fusion
                return await self._hybrid_search_alpha(
                    query_embedding, query_terms, params, expr
                )

        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            raise

    async def _hybrid_search_rrf(
        self,
        query_embedding: List[float],
        query_terms: Dict[int, float],
        params: HybridSearchParams,
        expr: Optional[str],
    ) -> List[RetrievalResult]:
        """Perform RRF hybrid search using Milvus native capabilities or fallback."""

        if HYBRID_SEARCH_AVAILABLE:
            try:
                # Create search requests for dense and sparse
                dense_request = AnnSearchRequest(
                    data=[query_embedding],
                    anns_field="dense_vector",
                    param={"metric_type": "COSINE", "params": {"ef": 256}},
                    limit=params.topk,
                    expr=expr,
                )

                sparse_request = AnnSearchRequest(
                    data=[query_terms],
                    anns_field="sparse_vector",
                    param={"metric_type": "IP", "params": {}},
                    limit=params.topk,
                    expr=expr,
                )

                # Perform hybrid search with RRF
                ranker = RRFRanker(k=params.rrf_k)

                assert self.collection is not None
                hybrid_kwargs = {
                    "reqs": [dense_request, sparse_request],
                    "limit": params.topk,
                    "output_fields": ["id", "content", "metadata"],
                }

                if HYBRID_SEARCH_USES_RERANK:
                    hybrid_kwargs["rerank"] = ranker
                else:
                    hybrid_kwargs["ranker"] = ranker

                results = self.collection.hybrid_search(**hybrid_kwargs)

                retrieval_results = []
                for rank, hit in enumerate(results[0]):
                    doc = Document(
                        id=hit.entity.get("id"),
                        content=hit.entity.get("content"),
                        metadata=hit.entity.get("metadata", {}),
                    )

                    retrieval_results.append(
                        RetrievalResult(
                            document=doc,
                            score=hit.score,
                            rank=rank + 1,
                            retrieval_method="hybrid_rrf_native",
                            debug_info={"rrf_score": hit.score, "rrf_k": params.rrf_k},
                        )
                    )

                logger.debug(
                    f"Native RRF hybrid search returned "
                    f"{len(retrieval_results)} results"
                )
                return retrieval_results

            except Exception as e:
                logger.warning(
                    f"Native hybrid search failed, falling back to manual RRF: {e}"
                )

        # Fallback: manual RRF implementation
        return await self._hybrid_search_rrf_fallback(
            query_embedding, query_terms, params, expr
        )

    async def _hybrid_search_rrf_fallback(
        self,
        query_embedding: List[float],
        query_terms: Dict[int, float],
        params: HybridSearchParams,
        expr: Optional[str],
    ) -> List[RetrievalResult]:
        """Fallback RRF implementation using separate searches."""

        # Perform separate searches
        dense_results = await self.dense_search(
            query_embedding, params.topk, params.filters
        )
        sparse_results = await self.sparse_search(
            query_terms, params.topk, params.filters
        )

        # Apply RRF fusion
        fused_results = SearchFusion.reciprocal_rank_fusion(
            [dense_results, sparse_results], k=params.rrf_k
        )

        # Update retrieval method for debug info
        for result in fused_results:
            result.retrieval_method = "hybrid_rrf_fallback"

        logger.debug(
            f"Fallback RRF hybrid search returned {len(fused_results)} results"
        )
        return fused_results[: params.topk]

    async def _hybrid_search_alpha(
        self,
        query_embedding: List[float],
        query_terms: Dict[int, float],
        params: HybridSearchParams,
        expr: Optional[str],
    ) -> List[RetrievalResult]:
        """Perform alpha fusion hybrid search."""

        # Perform separate searches
        dense_results = await self.dense_search(
            query_embedding, params.topk, params.filters
        )
        sparse_results = await self.sparse_search(
            query_terms, params.topk, params.filters
        )

        # Apply alpha fusion
        fused_results = SearchFusion.alpha_fusion(
            dense_results, sparse_results, params.alpha
        )

        # Limit to topk
        return fused_results[: params.topk]

    def _build_filter_expression(self, filters: Dict[str, Any]) -> str:
        """Build Milvus filter expression from filters dict."""
        expressions = []

        for field, value in filters.items():
            if isinstance(value, str):
                expressions.append(f'metadata["{field}"] == "{value}"')
            elif isinstance(value, (int, float)):
                expressions.append(f'metadata["{field}"] == {value}')
            elif isinstance(value, list):
                # IN operator for lists
                value_str = ", ".join(
                    [f'"{v}"' if isinstance(v, str) else str(v) for v in value]
                )
                expressions.append(f'metadata["{field}"] in [{value_str}]')

        return " and ".join(expressions) if expressions else ""

    async def get_document(self, doc_id: str) -> Optional[Document]:
        """Get a document by ID."""
        if not self.collection:
            raise RuntimeError("Collection not initialized")

        try:
            results = self.collection.query(
                expr=f'id == "{doc_id}"', output_fields=["id", "content", "metadata"]
            )

            if results:
                result = results[0]
                return Document(
                    id=result["id"],
                    content=result["content"],
                    metadata=result.get("metadata", {}),
                )
            return None

        except Exception as e:
            logger.error(f"Failed to get document {doc_id}: {e}")
            raise

    async def delete_document(self, doc_id: str) -> bool:
        """Delete a document by ID."""
        if not self.collection:
            raise RuntimeError("Collection not initialized")

        try:
            self.collection.delete(expr=f'id == "{doc_id}"')
            self.collection.flush()

            logger.info(f"Deleted document {doc_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete document {doc_id}: {e}")
            return False

    async def update_document(self, doc_id: str, document: Document) -> bool:
        """Update an existing document (delete + insert)."""
        try:
            # Delete existing document
            await self.delete_document(doc_id)

            # Insert updated document
            await self.insert_documents([document])

            return True

        except Exception as e:
            logger.error(f"Failed to update document {doc_id}: {e}")
            return False

    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        if not self.collection:
            raise RuntimeError("Collection not initialized")

        try:
            stats = {
                "name": self.collection_name,
                "num_entities": self.collection.num_entities,
                "description": self.collection.description,
                "indexes": [],
            }

            # Get index information
            indexes = self.collection.indexes
            for index in indexes:
                stats["indexes"].append(
                    {
                        "field_name": index.field_name,
                        "index_name": index.index_name,
                        "params": index.params,
                    }
                )

            return stats

        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            raise
