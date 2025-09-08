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

    async def connect(self) -> None:
        try:
            if self.api_key:
                self.client = weaviate.connect_to_weaviate_cloud(
                    cluster_url=self.url,
                    auth_credentials=weaviate.AuthApiKey(self.api_key),
                )
            else:
                # local/self-hosted
                self.client = weaviate.connect_to_custom(url=self.url)

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
            existing = [c.name for c in self.client.collections.list_all()]
            if self.class_name not in existing:
                self.client.collections.create(
                    name=self.class_name,
                    properties=[
                        wcfg.Property(name="content", data_type=wcfg.DataType.TEXT),
                        wcfg.Property(name="title", data_type=wcfg.DataType.TEXT),
                    ],
                    vectorizer_config=wcfg.Configure.Vectorizer.none(),
                    vector_index_config=wcfg.Configure.VectorIndex.hnsw(
                        distance=metric_enum
                    ),
                )
                logger.info(f"Created Weaviate collection: {self.class_name}")

            self.collection = self.client.collections.get(self.class_name)

        except Exception as e:
            logger.error(f"Failed creating Weaviate collection: {e}")
            raise

    async def insert_documents(self, documents: List[Document]) -> List[str]:
        if not self.collection:
            raise RuntimeError("Collection not initialized")
        try:
            ids: List[str] = []
            # dynamic batching
            with self.collection.batch.dynamic() as batch:
                for doc in documents:
                    vec = doc.embedding
                    props = {
                        "content": doc.content,
                        "title": doc.metadata.get("title", doc.id),
                    }
                    # include metadata into properties (shallow merge; avoid collisions)
                    for k, v in doc.metadata.items():
                        if k not in props and isinstance(
                            v, (str, int, float, bool)
                        ):
                            props[k] = v
                    batch.add_object(properties=props, uuid=doc.id, vector=vec)
                    ids.append(doc.id)
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
        if not self.collection:
            raise RuntimeError("Collection not initialized")
        try:
            where = self._build_where(filters) if filters else None
            res = self.collection.query.near_vector(
                near_vector=query_embedding,
                limit=topk,
                filters=where,
                return_metadata=["score"],
                return_properties=["content", "title"],
            )
            out: List[RetrievalResult] = []
            for rank, o in enumerate(res.objects, 1):
                doc = Document(
                    id=str(o.uuid),
                    content=str(o.properties.get("content", "")),
                    metadata={
                        k: v
                        for k, v in (o.properties or {}).items()
                        if k not in {"content", "title"}
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
        # Weaviate BM25 expects a raw text query; we cannot reconstruct from sparse terms reliably.
        # Use empty result to avoid misleading behavior; hybrid() path should be preferred.
        return []

    async def hybrid_search(
        self,
        query_embedding: List[float],
        query_terms: Dict[int, float],
        params: HybridSearchParams,
    ) -> List[RetrievalResult]:
        if not self.collection:
            raise RuntimeError("Collection not initialized")
        try:
            query_text = params.original_query or ""
            where = self._build_where(params.filters) if params.filters else None
            # Use native hybrid with alpha as specified;
            # relative score fusion is default
            res = self.collection.query.hybrid(
                query=query_text,
                vector=query_embedding,
                alpha=params.alpha,
                limit=params.topk,
                filters=where,
                return_metadata=["score"],
                return_properties=["content", "title"],
            )
            out: List[RetrievalResult] = []
            for rank, o in enumerate(res.objects, 1):
                doc = Document(
                    id=str(o.uuid),
                    content=str(o.properties.get("content", "")),
                    metadata={
                        k: v
                        for k, v in (o.properties or {}).items()
                        if k not in {"content", "title"}
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
        if not self.collection:
            raise RuntimeError("Collection not initialized")
        try:
            o = self.collection.data.objects.get_by_id(doc_id)
            if not o:
                return None
            return Document(
                id=str(o.uuid),
                content=str((o.properties or {}).get("content", "")),
                metadata={
                    k: v
                    for k, v in (o.properties or {}).items()
                    if k not in {"content", "title"}
                },
            )
        except Exception as e:
            logger.error(f"Weaviate get_document failed: {e}")
            raise

    async def delete_document(self, doc_id: str) -> bool:
        if not self.collection:
            raise RuntimeError("Collection not initialized")
        try:
            self.collection.data.objects.delete_by_id(doc_id)
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
        if not self.collection:
            raise RuntimeError("Collection not initialized")
        try:
            info = self.collection.config.get()
            # Weaviate does not expose entity count here without aggregate query;
            # return minimal config
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
            }
        except Exception as e:
            logger.error(f"Weaviate get_collection_stats failed: {e}")
            raise

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
