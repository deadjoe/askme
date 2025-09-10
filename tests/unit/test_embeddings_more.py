"""
Additional tests to raise coverage for BGEEmbeddingService: compute_similarity,
conversion utilities, normalization and cleanup.
"""

from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np
import pytest

from askme.core.config import EmbeddingConfig
from askme.core.embeddings import BGEEmbeddingService


@pytest.mark.asyncio
async def test_encode_query_normalization_default():
    cfg = EmbeddingConfig(normalize_embeddings=True)
    svc = BGEEmbeddingService(cfg)
    # Patch model and mark initialized
    class _M:
        def encode(self, *args: Any, **kwargs: Any):  # type: ignore[no-redef]
            return {"dense_vecs": [np.array([3.0, 4.0])], "sparse_vecs": [{}]}

    svc.model = _M()
    svc._is_initialized = True
    out = await svc.encode_query("q")
    # 3-4-5 normalized -> length 1
    norm = math.sqrt(sum(x * x for x in out["dense_embedding"]))
    assert abs(norm - 1.0) < 1e-6


def test_convert_sparse_embedding_variants():
    cfg = EmbeddingConfig()
    svc = BGEEmbeddingService(cfg)
    # dict
    d = svc._convert_sparse_embedding({"1": 0.5, 2: 0.3})
    assert d[1] == 0.5 and d[2] == 0.3
    # list -> indices of nonzero
    l = svc._convert_sparse_embedding([0.0, 0.1, 0.0, 0.2])
    assert l == {1: 0.1, 3: 0.2}
    # unknown -> {}
    class _X: ...
    assert svc._convert_sparse_embedding(_X()) == {}


@pytest.mark.asyncio
async def test_compute_similarity_dense_sparse_hybrid_and_errors():
    cfg = EmbeddingConfig()
    svc = BGEEmbeddingService(cfg)
    q = {"dense_embedding": [1.0, 0.0], "sparse_embedding": {0: 1.0, 2: 0.5}}
    docs = [
        {"dense_embedding": [0.0, 1.0], "sparse_embedding": {1: 0.2}},  # orthogonal dense
        {"dense_embedding": [1.0, 0.0], "sparse_embedding": {0: 0.5}},  # dense match
    ]
    # dense
    ds = await svc.compute_similarity(q, docs, method="dense")
    assert ds == [0.0, 1.0]
    # sparse
    ss = await svc.compute_similarity(q, docs, method="sparse")
    assert abs(ss[0] - 0.0) < 1e-6 and abs(ss[1] - 0.5) < 1e-6
    # hybrid alpha=0.2
    hs = await svc.compute_similarity(q, docs, method="hybrid", alpha=0.2)
    assert abs(hs[1] - (0.2 * 1.0 + 0.8 * 0.5)) < 1e-6
    # unknown method
    with pytest.raises(ValueError):
        await svc.compute_similarity(q, docs, method="unknown")


@pytest.mark.asyncio
async def test_cleanup_resets_state(monkeypatch):
    cfg = EmbeddingConfig()
    svc = BGEEmbeddingService(cfg)
    class _M: ...
    svc.model = _M()
    svc._is_initialized = True
    # Simulate no torch.cuda available path
    monkeypatch.setattr("askme.core.embeddings.torch", None, raising=False)
    await svc.cleanup()
    assert svc.model is None and svc._is_initialized is False

