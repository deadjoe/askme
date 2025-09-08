"""
Contract test for Weaviate hybrid retrieval.

Skips if WEAVIATE_URL not set or weaviate client not available.
"""

import os
from typing import Any

import pytest


@pytest.mark.skipif(not os.getenv("WEAVIATE_URL"), reason="WEAVIATE_URL not configured")
def test_weaviate_hybrid_contract() -> None:
    try:
        import weaviate
        from weaviate.classes import config as wcfg
    except Exception:  # pragma: no cover
        pytest.skip("weaviate client not available")

    url = os.environ["WEAVIATE_URL"]
    client = weaviate.connect_to_custom(url=url)

    try:
        cname = "AskmeContract"
        # Drop if exists
        try:
            client.collections.delete(cname)
        except Exception:
            pass
        # Create collection
        client.collections.create(
            name=cname,
            properties=[
                wcfg.Property(name="content", data_type=wcfg.DataType.TEXT),
                wcfg.Property(name="title", data_type=wcfg.DataType.TEXT),
            ],
            vectorizer_config=wcfg.Configure.Vectorizer.none(),
            vector_index_config=wcfg.Configure.VectorIndex.hnsw(
                distance=wcfg.VectorDistances.COSINE
            ),
        )
        col = client.collections.get(cname)
        # Insert two docs with precomputed vectors (zero vectors for smoke)
        with col.batch.dynamic() as batch:
            batch.add_object(
                properties={"content": "hybrid bm25 and dense", "title": "doc1"},
                uuid="d1",
                vector=[0.0] * 4,
            )
            batch.add_object(
                properties={"content": "semantic vector search", "title": "doc2"},
                uuid="d2",
                vector=[0.0] * 4,
            )

        res = col.query.hybrid(
            query="hybrid search",
            vector=[0.0] * 4,
            alpha=0.5,
            limit=2,
            return_metadata=["score"],
            return_properties=["content", "title"],
        )
        assert len(res.objects) >= 1
        titles = [str(o.properties.get("title", "")) for o in res.objects]
        assert any(t in titles for t in ["doc1", "doc2"])  # at least one match
    finally:
        try:
            client.close()
        except Exception:
            pass
