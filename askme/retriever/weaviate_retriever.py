"""
Weaviate retriever implementation with hybrid search support
(alpha/relative score fusion).

Notes:
- We use Weaviate's built-in hybrid query combining BM25 and vector search.
- Dense vectors are stored on insert; BM25 operates on the `content` property.
- Filters are mapped to simple equality/in conditions.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import weaviate
from weaviate.classes import config as wcfg
from weaviate.classes.query import Filter

from .base import Document, HybridSearchParams, RetrievalResult, VectorRetriever

logger = logging.getLogger(__name__)


class WeaviateRetriever(VectorRetriever):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.url = config.get("url", "http://localhost:8080")
        self.api_key = config.get("api_key", "")
        self.class_name = config.get("class_name", "AskmeDocument")
        self.client: Optional[weaviate.WeaviateClient] = None
        self.collection = None
        self.dimension = config.get("dimension", 1024)

    @property
    def supports_native_upsert(self) -> bool:
        """Weaviate supports native upsert via deterministic UUID generation."""
        return True

    async def connect(self) -> None:
        try:
            if self.api_key:
                self.client = weaviate.connect_to_weaviate_cloud(
                    cluster_url=self.url,
                    auth_credentials=weaviate.AuthApiKey(self.api_key),
                )
            else:
                # local/self-hosted: parse URL into host/port for client v4
                from urllib.parse import urlparse

                u = urlparse(self.url)
                host = u.hostname or "localhost"
                port = u.port or (443 if (u.scheme == "https") else 8080)
                http_secure = u.scheme == "https"
                # weaviate-client v4 requires both HTTP and gRPC params
                grpc_host = host
                # Heuristic: if HTTP exposed at 8081, expose gRPC at 8082
                # (docker run mapping); else default 50051
                grpc_port = port + 1 if port in (8080, 8081) else 50051
                grpc_secure = http_secure
                self.client = weaviate.connect_to_custom(
                    http_host=host,
                    http_port=port,
                    http_secure=http_secure,
                    grpc_host=grpc_host,
                    grpc_port=grpc_port,
                    grpc_secure=grpc_secure,
                    skip_init_checks=True,
                )

            logger.info(f"Connected to Weaviate at {self.url}")

            # Ensure collection exists
            await self.create_collection(self.dimension)

        except Exception as e:
            logger.error(f"Failed to connect to Weaviate: {e}")
            raise

    async def disconnect(self) -> None:
        try:
            if self.client is not None:
                self.client.close()
                self.client = None
                logger.info("Disconnected from Weaviate")
        except Exception as e:
            logger.warning(f"Error disconnecting from Weaviate: {e}")

    async def create_collection(
        self, dimension: int, metric: str = "cosine", **kwargs: Any
    ) -> None:
        assert self.client is not None
        try:
            metric_enum = (
                wcfg.VectorDistances.COSINE
                if metric.lower() == "cosine"
                else wcfg.VectorDistances.DOT
            )

            # Create if missing
            existing_raw = self.client.collections.list_all()
            existing = [
                c if isinstance(c, str) else getattr(c, "name", str(c))
                for c in existing_raw
            ]
            if self.class_name not in existing:
                # 显式配置向量索引（HNSW + 选定距离度量），避免依赖服务端默认
                self.client.collections.create(
                    name=self.class_name,
                    properties=[
                        wcfg.Property(name="doc_id", data_type=wcfg.DataType.TEXT),
                        wcfg.Property(name="content", data_type=wcfg.DataType.TEXT),
                        wcfg.Property(name="title", data_type=wcfg.DataType.TEXT),
                    ],
                    vectorizer_config=wcfg.Configure.Vectorizer.none(),
                    vector_index_config=wcfg.Configure.VectorIndex.hnsw(
                        distance_metric=metric_enum
                    ),
                )
                logger.info(f"Created Weaviate collection: {self.class_name}")

            self.collection = self.client.collections.get(self.class_name)

        except Exception as e:
            logger.error(f"Failed creating Weaviate collection: {e}")
            raise

    async def insert_documents(self, documents: List[Document]) -> List[str]:
        col = await self._ensure_collection()
        try:
            ids: List[str] = []
            # dynamic batching
            import uuid as _uuid

            # dynamic() may be a coroutine in tests (AsyncMock) or return
            # an async/sync context manager.
            mgr = col.batch.dynamic()
            try:
                from inspect import iscoroutine
            except Exception:  # pragma: no cover

                def iscoroutine(_):
                    return False

            if iscoroutine(mgr):
                mgr = await mgr
            used_async = False
            try:
                async with mgr as batch:  # type: ignore
                    used_async = True
                    for doc in documents:
                        vec = doc.embedding
                        if hasattr(vec, "tolist"):
                            vec = vec.tolist()  # ensure Python list for client
                        uid = str(_uuid.uuid5(_uuid.NAMESPACE_URL, doc.id))
                        props = {
                            "doc_id": doc.id,
                            "content": doc.content,
                            "title": doc.metadata.get("title", doc.id),
                        }
                        # include metadata into properties
                        # (shallow merge; avoid collisions)
                        for k, v in doc.metadata.items():
                            if k not in props and isinstance(
                                v, (str, int, float, bool)
                            ):
                                props[k] = v
                        _res = batch.add_object(properties=props, uuid=uid, vector=vec)
                        try:
                            from inspect import isawaitable
                        except Exception:  # pragma: no cover

                            def isawaitable(_):
                                return False

                        if isawaitable(_res):
                            await _res
                        ids.append(uid)
            except TypeError:
                # Not an async context manager; use sync context
                pass
            if not used_async:
                with mgr as batch:
                    for doc in documents:
                        vec = doc.embedding
                        if hasattr(vec, "tolist"):
                            vec = vec.tolist()  # ensure Python list for client
                        uid = str(_uuid.uuid5(_uuid.NAMESPACE_URL, doc.id))
                        props = {
                            "doc_id": doc.id,
                            "content": doc.content,
                            "title": doc.metadata.get("title", doc.id),
                        }
                        # include metadata into properties
                        # (shallow merge; avoid collisions)
                        for k, v in doc.metadata.items():
                            if k not in props and isinstance(
                                v, (str, int, float, bool)
                            ):
                                props[k] = v
                        _res = batch.add_object(properties=props, uuid=uid, vector=vec)
                        try:
                            from inspect import isawaitable
                        except Exception:  # pragma: no cover

                            def isawaitable(_):
                                return False

                        if isawaitable(_res):
                            await _res
                        ids.append(uid)
            return ids
        except Exception as e:
            logger.error(f"Failed to insert into Weaviate: {e}")
            raise

    async def dense_search(
        self,
        query_embedding: List[float],
        topk: int = 50,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalResult]:
        col = await self._ensure_collection()
        try:
            where = self._build_where(filters) if filters else None
            res = col.query.near_vector(
                near_vector=query_embedding,
                limit=topk,
                filters=where,
                return_metadata=["score"],
                return_properties=["content", "title", "doc_id"],
            )
            out: List[RetrievalResult] = []
            for rank, o in enumerate(res.objects, 1):
                props = o.properties or {}
                doc = Document(
                    id=str(props.get("doc_id", o.uuid)),
                    content=str(props.get("content", "")),
                    metadata={
                        k: v
                        for k, v in props.items()
                        if k not in {"content", "title", "doc_id"}
                    },
                )
                score = float(getattr(o.metadata, "score", 0.0) or 0.0)
                out.append(
                    RetrievalResult(
                        document=doc, score=score, rank=rank, retrieval_method="dense"
                    )
                )
            return out
        except Exception as e:
            logger.error(f"Weaviate dense search failed: {e}")
            raise

    async def sparse_search(
        self,
        query_terms: Dict[int, float],
        topk: int = 50,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalResult]:
        # Weaviate BM25 expects a raw text query; we cannot reconstruct
        # from sparse terms reliably. Log warning and return empty result.
        # Hybrid search with original_query should be used instead.
        logger.warning(
            "Weaviate sparse_search cannot reconstruct text from sparse terms. "
            "Use hybrid_search with original_query for BM25 functionality."
        )
        return []

    async def hybrid_search(
        self,
        query_embedding: List[float],
        query_terms: Dict[int, float],
        params: HybridSearchParams,
    ) -> List[RetrievalResult]:
        col = await self._ensure_collection()
        try:
            query_text = params.original_query or ""
            if not query_text:
                logger.warning(
                    "Weaviate hybrid search requires original_query text for BM25. "
                    "Falling back to dense-only search."
                )
            where = self._build_where(params.filters) if params.filters else None
            # Use native hybrid with alpha as specified;
            # relative score fusion is default
            res = col.query.hybrid(
                query=query_text,
                vector=query_embedding,
                alpha=params.alpha,
                limit=params.topk,
                filters=where,
                return_metadata=["score"],
                return_properties=["content", "title", "doc_id"],
            )
            out: List[RetrievalResult] = []
            for rank, o in enumerate(res.objects, 1):
                props = o.properties or {}
                doc = Document(
                    id=str(props.get("doc_id", o.uuid)),
                    content=str(props.get("content", "")),
                    metadata={
                        k: v
                        for k, v in props.items()
                        if k not in {"content", "title", "doc_id"}
                    },
                )
                score = float(getattr(o.metadata, "score", 0.0) or 0.0)
                out.append(
                    RetrievalResult(
                        document=doc, score=score, rank=rank, retrieval_method="hybrid"
                    )
                )
            return out
        except Exception as e:
            logger.error(f"Weaviate hybrid search failed: {e}")
            raise

    async def get_document(self, doc_id: str) -> Optional[Document]:
        col = await self._ensure_collection()
        try:
            # Allow lookup by original doc_id (property) or by uuid;
            # prefer property lookup
            try:
                where = Filter.by_property("doc_id").equal(doc_id)
                res = col.query.fetch_objects(
                    filters=where,
                    limit=1,
                    return_properties=["content", "title", "doc_id"],
                )
                o = res.objects[0] if res.objects else None
            except Exception:
                o = None
            if not o:
                o = col.data.objects.get_by_id(doc_id)
            if not o:
                return None
            props = o.properties or {}
            return Document(
                id=str(props.get("doc_id", o.uuid)),
                content=str(props.get("content", "")),
                metadata={
                    k: v
                    for k, v in props.items()
                    if k not in {"content", "title", "doc_id"}
                },
            )
        except Exception as e:
            logger.error(f"Weaviate get_document failed: {e}")
            raise

    async def delete_document(self, doc_id: str) -> bool:
        col = await self._ensure_collection()
        try:
            # Prefer deletion by original doc_id property (may not equal UUID)
            try:
                where = Filter.by_property("doc_id").equal(doc_id)
                res = col.query.fetch_objects(filters=where, limit=10)
                if res.objects:
                    for o in res.objects:
                        try:
                            col.data.objects.delete_by_id(o.uuid)
                        except Exception:
                            pass
                    return True
            except Exception:
                pass

            # Fallback: attempt delete by assuming provided id is UUID
            col.data.objects.delete_by_id(doc_id)
            return True
        except Exception as e:
            logger.error(f"Weaviate delete_document failed: {e}")
            return False

    async def update_document(self, doc_id: str, document: Document) -> bool:
        try:
            await self.delete_document(doc_id)
            await self.insert_documents([document])
            return True
        except Exception as e:
            logger.error(f"Weaviate update_document failed: {e}")
            return False

    async def get_collection_stats(self) -> Dict[str, Any]:
        col = await self._ensure_collection()
        try:
            info = col.config.get()
            total = None
            try:
                agg = col.aggregate.over_all(total_count=True)
                total = getattr(agg, "total_count", None)
                # Some versions may return dict-like
                if total is None and isinstance(agg, dict):
                    total = agg.get("total_count")
            except Exception:
                pass
            return {
                "name": self.class_name,
                "description": "Weaviate collection",
                "indexes": [
                    {
                        "vector_index": getattr(
                            info.vector_index_config, "__class__", type(info)
                        ).__name__
                    }
                ],
                "num_entities": int(total) if isinstance(total, (int, float)) else 0,
            }
        except Exception as e:
            logger.error(f"Weaviate get_collection_stats failed: {e}")
            raise

    async def _ensure_collection(self):
        if self.client is None:
            await self.connect()
        assert self.client is not None
        if self.collection is None:
            try:
                self.collection = self.client.collections.get(self.class_name)
            except Exception:
                # Try to create then get
                await self.create_collection(self.dimension)
                self.collection = self.client.collections.get(self.class_name)
        if self.collection is None:
            raise RuntimeError("Collection not initialized")
        return self.collection

    def _build_where(self, filters: Dict[str, Any]) -> Optional[Filter]:
        # Support basic equality / IN filters on top-level properties
        exprs: List[Filter] = []
        for field, value in filters.items():
            if isinstance(value, list) and value:
                ors = [Filter.by_property(field).equal(v) for v in value]
                # Combine ORs (chain)
                f = ors[0]
                for nxt in ors[1:]:
                    f = f | nxt
                exprs.append(f)
            else:
                exprs.append(Filter.by_property(field).equal(value))
        if not exprs:
            return None
        f = exprs[0]
        for nxt in exprs[1:]:
            f = f & nxt
        return f
