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

            # Create indexes with optimized HNSW parameters
            # Dense vector index with 2024 best practices
            dense_index_params = {
                "metric_type": metric.upper(),
                "index_type": "HNSW",
                "params": self._get_optimized_hnsw_params(dimension),
            }
            collection.create_index(
                field_name="dense_vector",
                index_params=dense_index_params,
                index_name="dense_idx",
            )

            # Sparse vector index for BM25 search using optimal metric type
            sparse_index_params = {
                "index_type": "SPARSE_INVERTED_INDEX",
                "metric_type": "BM25",  # BM25 optimized for full-text search
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
        """Insert documents with dense and sparse vectors using optimized batching."""
        if not self.collection:
            raise RuntimeError(
                "Collection not initialized. Call create_collection first."
            )

        if not documents:
            return []

        try:
            # Use optimized batch insertion for better performance
            all_inserted_ids = []
            batches = self._create_optimal_batches(documents)

            logger.info(
                f"Inserting {len(documents)} documents in "
                f"{len(batches)} optimized batches"
            )

            for batch_idx, batch_docs in enumerate(batches):
                # Prepare data for this batch
                data = []
                doc_ids = []
                for doc in batch_docs:
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
                    doc_ids.append(doc.id)

                # Safe upsert for this batch
                batch_result = await self._safe_upsert(data, doc_ids)
                all_inserted_ids.extend(batch_result)

                logger.debug(
                    f"Completed batch {batch_idx + 1}/{len(batches)}: "
                    f"{len(batch_docs)} documents"
                )

            # Strategic flush: only flush at the end or for large insertions
            should_flush = await self._should_flush(len(documents))
            if should_flush:
                logger.debug("Performing strategic flush after batch insertion")
                self.collection.flush()

            logger.info(
                f"Successfully upserted {len(documents)} documents into "
                f"{self.collection_name}"
            )

            return all_inserted_ids

        except Exception as e:
            logger.error(f"Failed to insert documents: {e}")
            raise

    def _create_optimal_batches(
        self, documents: List[Document]
    ) -> List[List[Document]]:
        """Create optimal batches based on content size and Milvus recommendations."""
        MAX_BATCH_SIZE_MB = 35  # Conservative limit under 40MB recommendation
        MAX_DOCS_PER_BATCH = 5000  # Conservative limit for metadata-rich documents
        MIN_BATCH_SIZE = 10  # Minimum batch size to avoid too many tiny batches

        batches = []
        current_batch: List[Document] = []
        current_size_mb = 0.0

        for doc in documents:
            # Estimate document size (rough approximation)
            embedding_bytes = (
                len(doc.embedding) * 4 if doc.embedding else 0
            )  # float32 = 4 bytes
            sparse_bytes = (
                len(doc.sparse_embedding) * 8 if doc.sparse_embedding else 0
            )  # rough estimate

            doc_size_bytes = (
                len(doc.content.encode("utf-8"))
                + len(str(doc.metadata).encode("utf-8"))
                + embedding_bytes
                + sparse_bytes
            )
            doc_size_mb = doc_size_bytes / (1024 * 1024)

            # Check if adding this document would exceed limits
            would_exceed_size = (current_size_mb + doc_size_mb) > MAX_BATCH_SIZE_MB
            would_exceed_count = len(current_batch) >= MAX_DOCS_PER_BATCH

            if current_batch and (would_exceed_size or would_exceed_count):
                # Finalize current batch
                batches.append(current_batch)
                current_batch = [doc]
                current_size_mb = doc_size_mb
            else:
                # Add to current batch
                current_batch.append(doc)
                current_size_mb += doc_size_mb

        # Add the last batch if it has documents
        if current_batch:
            batches.append(current_batch)

        # Merge very small batches to avoid overhead
        if len(batches) > 1:
            merged_batches = []
            i = 0
            while i < len(batches):
                current = batches[i]
                # Try to merge small consecutive batches
                while (
                    i + 1 < len(batches)
                    and len(current) < MIN_BATCH_SIZE
                    and len(current) + len(batches[i + 1]) <= MAX_DOCS_PER_BATCH
                ):
                    current.extend(batches[i + 1])
                    i += 1
                merged_batches.append(current)
                i += 1
            batches = merged_batches

        logger.debug(
            f"Created {len(batches)} optimal batches from {len(documents)} documents"
        )
        return batches

    def _safe_calculate_score_range(
        self, all_hits: List[Any], current_score: Any
    ) -> str:
        """Safely calculate score range, handling mock objects in tests."""
        try:
            # Check if current_score is numeric
            if not isinstance(current_score, (int, float)):
                return "mock-mock"

            all_scores = [hit.score for hit in all_hits if hasattr(hit, "score")]
            all_scores.append(current_score)
            # Filter out non-numeric scores (e.g., mock objects)
            numeric_scores = [s for s in all_scores if isinstance(s, (int, float))]
            if numeric_scores:
                return f"{min(numeric_scores): .4f}-{max(numeric_scores): .4f}"
            else:
                return f"{current_score: .4f}-{current_score: .4f}"
        except (TypeError, AttributeError, ValueError):
            return "mock-mock"

    async def _should_flush(self, num_documents: int) -> bool:
        """Determine if flush should be called based on best practices."""
        # Flush strategies based on Milvus documentation:
        # 1. Don't flush for small insertions (let automatic 10-min flush handle it)
        # 2. Flush for large bulk insertions when no more data is coming
        # 3. Flush before backups or when immediate visibility is critical

        LARGE_INSERTION_THRESHOLD = 1000  # Documents

        # For large insertions, flush immediately for consistency
        if num_documents >= LARGE_INSERTION_THRESHOLD:
            logger.debug(f"Large insertion ({num_documents} docs), will flush")
            return True

        # For smaller insertions, rely on automatic flush (10-min interval)
        logger.debug(
            f"Small insertion ({num_documents} docs), skipping flush for performance"
        )
        return False

    async def _safe_upsert(self, data: List[Dict], doc_ids: List[str]) -> List[str]:
        """
        Perform safe upsert to avoid Milvus primary key duplicate issues.

        This method addresses the known Milvus issue where upsert doesn't properly
        handle duplicate primary keys, leading to inconsistent query results.
        """
        try:
            # Step 1: Check for existing documents to avoid duplicates
            existing_ids = []
            if doc_ids:
                # Query existing documents in batches to avoid large query expressions
                batch_size = 100  # Reasonable batch size for ID queries
                for i in range(0, len(doc_ids), batch_size):
                    batch_ids = doc_ids[i : i + batch_size]
                    id_expr = (
                        "id in [" + ",".join(f'"{id_}"' for id_ in batch_ids) + "]"
                    )

                    try:
                        if self.collection is None:
                            raise RuntimeError("Collection not initialized")
                        existing_results = self.collection.query(
                            expr=id_expr, output_fields=["id"], limit=batch_size
                        )
                        existing_ids.extend([r["id"] for r in existing_results])
                    except Exception as e:
                        logger.warning(f"Failed to query existing IDs: {e}")
                        # Continue with upsert if query fails
                        break

            # Step 2: Delete existing documents explicitly to ensure clean upsert
            if existing_ids:
                logger.debug(
                    f"Deleting {len(existing_ids)} existing documents before upsert"
                )
                try:
                    if self.collection is None:
                        raise RuntimeError("Collection not initialized")
                    delete_expr = (
                        "id in [" + ",".join(f'"{id_}"' for id_ in existing_ids) + "]"
                    )
                    self.collection.delete(expr=delete_expr)
                    # Small flush to ensure deletion is visible before insertion
                    self.collection.flush()
                except Exception as e:
                    logger.warning(f"Failed to delete existing documents: {e}")
                    # Continue with upsert anyway

            # Step 3: Perform clean insert
            if self.collection is None:
                raise RuntimeError("Collection not initialized")
            result = self.collection.upsert(data)

            # Return document IDs as strings
            if hasattr(result, "primary_keys") and result.primary_keys:
                return [str(pk) for pk in result.primary_keys]
            else:
                return doc_ids  # Fallback to input IDs

        except Exception as e:
            logger.error(f"Safe upsert failed: {e}")
            # Fallback to simple upsert if safe method fails
            logger.warning("Falling back to simple upsert operation")
            if self.collection is None:
                raise RuntimeError("Collection not initialized")
            result = self.collection.upsert(data)
            if hasattr(result, "primary_keys") and result.primary_keys:
                return [str(pk) for pk in result.primary_keys]
            else:
                return doc_ids

    async def dense_search(
        self,
        query_embedding: List[float],
        topk: int = 50,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalResult]:
        """Perform dense vector search with enhanced observability."""
        if not self.collection:
            raise RuntimeError("Collection not initialized")

        import time

        search_start = time.perf_counter()

        try:
            # Use adaptive ef parameter for optimal search performance
            optimal_ef = self._get_optimal_search_ef(topk)
            search_params = {"metric_type": "COSINE", "params": {"ef": optimal_ef}}

            # Build filter expression if provided
            expr = self._build_filter_expression(filters) if filters else None
            filter_count = len(filters) if filters else 0

            logger.debug(
                f"Dense search starting: topk={topk}, ef={optimal_ef}, "
                f"filters={filter_count}, expr_len={len(expr or '')}"
            )

            search_operation_start = time.perf_counter()
            results = self.collection.search(
                data=[query_embedding],
                anns_field="dense_vector",
                param=search_params,
                limit=topk,
                expr=expr,
                output_fields=["id", "content", "metadata"],
            )
            search_operation_time = (
                time.perf_counter() - search_operation_start
            ) * 1000

            retrieval_results = []
            processing_start = time.perf_counter()

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
                        debug_info={
                            "distance": hit.distance,
                            "ef_used": optimal_ef,
                            "score_range": self._safe_calculate_score_range(
                                results[0], hit.score
                            ),
                        },
                    )
                )

            processing_time = (time.perf_counter() - processing_start) * 1000
            total_time = (time.perf_counter() - search_start) * 1000

            # Enhanced logging with performance metrics
            logger.info(
                f"Dense search completed: {len(retrieval_results)}/{topk} results, "
                f"search={search_operation_time: .1f}ms, "
                f"processing={processing_time: .1f}ms, "
                f"total={total_time: .1f}ms, ef={optimal_ef}"
            )

            if filters:
                logger.debug(f"Applied filters: {list(filters.keys())}")

            return retrieval_results

        except Exception as e:
            search_time = (time.perf_counter() - search_start) * 1000
            logger.error(f"Dense search failed after {search_time: .1f}ms: {e}")
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
            search_params = {
                "metric_type": "BM25",
                "params": {},
            }  # Use BM25 for optimal sparse search

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
                # Use optimized ef parameter for hybrid dense search
                optimal_ef = self._get_optimal_search_ef(params.topk)
                dense_request = AnnSearchRequest(
                    data=[query_embedding],
                    anns_field="dense_vector",
                    param={"metric_type": "COSINE", "params": {"ef": optimal_ef}},
                    limit=params.topk,
                    expr=expr,
                )

                sparse_request = AnnSearchRequest(
                    data=[query_terms],
                    anns_field="sparse_vector",
                    param={
                        "metric_type": "BM25",
                        "params": {},
                    },  # Use BM25 for consistent sparse search
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
        """Build Milvus filter expression from filters dict with proper escaping."""
        if not filters:
            return ""

        expressions = []

        for field, value in filters.items():
            # Validate and escape field name
            escaped_field = self._escape_field_name(field)

            if isinstance(value, dict):
                # Support advanced operators: {"$gt": 10}, {"$in": [1,2,3]}, etc.
                expr = self._build_operator_expression(escaped_field, value)
                if expr:
                    expressions.append(expr)
            elif isinstance(value, str):
                # Escape string values to prevent injection
                escaped_value = self._escape_string_value(value)
                expressions.append(f'{escaped_field} == "{escaped_value}"')
            elif isinstance(value, (int, float)):
                expressions.append(f"{escaped_field} == {value}")
            elif isinstance(value, bool):
                expressions.append(f"{escaped_field} == {str(value).lower()}")
            elif isinstance(value, list) and value:
                # IN operator for non-empty lists
                escaped_values = []
                for v in value:
                    if isinstance(v, str):
                        escaped_values.append(f'"{self._escape_string_value(v)}"')
                    elif isinstance(v, (int, float)):
                        escaped_values.append(str(v))
                    elif isinstance(v, bool):
                        escaped_values.append(str(v).lower())
                    else:
                        logger.warning(f"Unsupported value type in list: {type(v)}")

                if escaped_values:
                    value_str = ", ".join(escaped_values)
                    expressions.append(f"{escaped_field} in [{value_str}]")
            elif value is None:
                # Support null checks
                expressions.append(f"{escaped_field} == null")
            else:
                logger.warning(f"Unsupported filter value type: {type(value)}")

        return " and ".join(expressions) if expressions else ""

    def _escape_field_name(self, field: str) -> str:
        """Escape field name for safe use in Milvus expressions."""
        # Validate field name contains only safe characters
        import re

        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", field):
            raise ValueError(f"Invalid field name: {field}")

        # Use metadata accessor for all fields except top-level fields
        if field in ["id", "content"]:
            return field
        else:
            return f'metadata["{field}"]'

    def _escape_string_value(self, value: str) -> str:
        """Escape string value to prevent injection attacks."""
        # Escape double quotes and backslashes
        return value.replace("\\", "\\\\").replace('"', '\\"')

    def _build_operator_expression(
        self, field: str, operator_dict: Dict[str, Any]
    ) -> str:
        """Build expression for advanced operators."""
        expressions = []

        for op, val in operator_dict.items():
            if op == "$eq":
                if isinstance(val, str):
                    escaped_val = self._escape_string_value(val)
                    expressions.append(f'{field} == "{escaped_val}"')
                else:
                    expressions.append(f"{field} == {val}")

            elif op == "$ne":
                if isinstance(val, str):
                    escaped_val = self._escape_string_value(val)
                    expressions.append(f'{field} != "{escaped_val}"')
                else:
                    expressions.append(f"{field} != {val}")

            elif op == "$gt":
                expressions.append(f"{field} > {val}")

            elif op == "$gte":
                expressions.append(f"{field} >= {val}")

            elif op == "$lt":
                expressions.append(f"{field} < {val}")

            elif op == "$lte":
                expressions.append(f"{field} <= {val}")

            elif op == "$in":
                if isinstance(val, list) and val:
                    escaped_values = []
                    for v in val:
                        if isinstance(v, str):
                            escaped_values.append(f'"{self._escape_string_value(v)}"')
                        else:
                            escaped_values.append(str(v))
                    value_str = ", ".join(escaped_values)
                    expressions.append(f"{field} in [{value_str}]")

            elif op == "$nin":
                if isinstance(val, list) and val:
                    escaped_values = []
                    for v in val:
                        if isinstance(v, str):
                            escaped_values.append(f'"{self._escape_string_value(v)}"')
                        else:
                            escaped_values.append(str(v))
                    value_str = ", ".join(escaped_values)
                    expressions.append(f"{field} not in [{value_str}]")

            elif op == "$like":
                if isinstance(val, str):
                    escaped_val = self._escape_string_value(val)
                    expressions.append(f'{field} like "{escaped_val}"')

            else:
                logger.warning(f"Unsupported operator: {op}")

        return " and ".join(expressions) if expressions else ""

    def _get_optimized_hnsw_params(self, dimension: int) -> Dict[str, Any]:
        """
        Calculate optimized HNSW parameters based on 2024 best practices.

        Parameters are optimized for the specific embedding dimension and
        balanced for production RAG workloads (speed vs. recall trade-off).
        """
        # Adaptive M parameter based on dimension
        # Higher dimensions benefit from more connections
        if dimension <= 384:
            # Small embeddings (e.g., BERT-base, smaller models)
            M = 16
        elif dimension <= 768:
            # Medium embeddings (e.g., BERT-large, BGE-base)
            M = 24
        elif dimension <= 1536:
            # Large embeddings (e.g., BGE-m3, OpenAI embeddings)
            M = 32
        else:
            # Very large embeddings
            M = 48

        # efConstruction based on quality-speed balance for production
        # Higher efConstruction = better quality but slower indexing
        # For production RAG, we favor reasonable quality with acceptable build time
        efConstruction = max(100, M * 8)  # Minimum 100, typically M * 8

        # Ensure efConstruction is within valid Milvus range
        efConstruction = min(efConstruction, 500)  # Cap at 500 for build time

        params = {
            "M": M,
            "efConstruction": efConstruction,
        }

        logger.info(
            f"Optimized HNSW params for dim={dimension}: M={M}, "
            f"efConstruction={efConstruction}"
        )

        return params

    def _get_optimal_search_ef(self, topk: int) -> int:
        """
        Calculate optimal ef parameter for search based on 2024 best practices.

        ef controls the search accuracy-speed tradeoff. Higher ef = better recall
        but slower search. For RAG systems, we balance quality and latency.
        """
        # Base ef should be at least 2x topk for good recall
        base_ef = max(topk * 2, 64)

        # Scale based on topk requirement
        if topk <= 10:
            # Small result sets: prioritize speed for interactive queries
            ef = min(base_ef, 128)
        elif topk <= 50:
            # Medium result sets: balance quality and speed for typical RAG
            ef = min(base_ef, 256)
        else:
            # Large result sets: prioritize quality for comprehensive retrieval
            ef = min(base_ef, 512)

        logger.debug(f"Optimal search ef={ef} for topk={topk}")
        return ef

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
        """Get comprehensive collection statistics with enhanced observability."""
        if not self.collection:
            raise RuntimeError("Collection not initialized")

        try:
            import time

            stats_start = time.perf_counter()

            # Basic collection info
            stats = {
                "name": self.collection_name,
                "num_entities": self.collection.num_entities,
                "description": self.collection.description,
                "indexes": [],
                "performance": {
                    "stats_collection_time_ms": 0,
                },
                "memory_usage": {},
                "search_performance": {},
            }

            # Get index information with enhanced details
            indexes = self.collection.indexes
            for index in indexes:
                index_info = {
                    "field_name": index.field_name,
                    "index_name": index.index_name,
                    "params": index.params,
                    "index_type": index.params.get("index_type", "unknown"),
                }

                # Add HNSW-specific information if applicable
                if index_info["index_type"] == "HNSW":
                    hnsw_params = index.params.get("params", {})
                    index_info["hnsw_details"] = {
                        "M": hnsw_params.get("M", "unknown"),
                        "efConstruction": hnsw_params.get("efConstruction", "unknown"),
                        "metric_type": index.params.get("metric_type", "unknown"),
                    }

                stats["indexes"].append(index_info)

            # Try to get memory usage statistics
            try:
                # Get collection loading status and memory info
                load_state = self.collection.load_state
                stats["memory_usage"]["load_state"] = str(load_state)

                # Get partition information if available
                partitions = self.collection.partitions
                stats["partitions"] = [
                    {"name": p.name, "description": p.description} for p in partitions
                ]

            except Exception as e:
                logger.debug(f"Could not get memory usage stats: {e}")
                stats["memory_usage"]["error"] = str(e)

            # Add search performance recommendations
            stats["search_performance"] = {
                "recommended_topk_ranges": {
                    "interactive": "1-20 (fast response)",
                    "standard_rag": "20-100 (balanced)",
                    "comprehensive": "100-500 (thorough)",
                },
                "current_hnsw_config": "adaptive based on dimension",
                "ef_optimization": "dynamic based on topk",
            }

            # Add data distribution insights
            if stats["num_entities"] > 0:
                stats["data_insights"] = {
                    "entity_count": stats["num_entities"],
                    "estimated_memory_mb": stats["num_entities"]
                    * 0.004,  # Rough estimate
                    "search_latency_estimate": "< 50ms for typical queries",
                }

            stats_time = (time.perf_counter() - stats_start) * 1000
            stats["performance"]["stats_collection_time_ms"] = round(stats_time, 2)

            logger.debug(f"Collection stats collected in {stats_time: .1f}ms")
            return stats

        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            raise
