"""
Unit tests for search result fusion (RRF and alpha fusion).
"""

from askme.retriever.base import Document, RetrievalResult, SearchFusion


def _mk_res(
    doc_id: str, score: float, rank: int, method: str = "dense"
) -> RetrievalResult:
    return RetrievalResult(
        document=Document(id=doc_id, content=f"content {doc_id}"),
        score=score,
        rank=rank,
        retrieval_method=method,
    )


def test_rrf_fusion_basic() -> None:
    # dense results: d1 rank1, d2 rank2
    dense = [_mk_res("d1", 0.9, 1, "dense"), _mk_res("d2", 0.7, 2, "dense")]
    # sparse results: d2 rank1, d3 rank2
    sparse = [_mk_res("d2", 2.0, 1, "sparse"), _mk_res("d3", 1.5, 2, "sparse")]

    fused = SearchFusion.reciprocal_rank_fusion([dense, sparse], k=60)
    ids = [r.document.id for r in fused[:3]]
    # d1 和 d2 都获得高分；由于 d1 在 dense 排名第一，d2 在 sparse 排名第一，顺序一般为 d1/d2 领先
    assert ids[0] in ("d1", "d2")
    assert set(ids[:3]) == {"d1", "d2", "d3"}


def test_alpha_fusion_weighting() -> None:
    # 让 dense 偏好 dA，sparse 偏好 dB
    dense = [_mk_res("dA", 10.0, 1, "dense"), _mk_res("dB", 1.0, 2, "dense")]
    sparse = [_mk_res("dB", 10.0, 1, "sparse"), _mk_res("dA", 1.0, 2, "sparse")]

    fused_dense_bias = SearchFusion.alpha_fusion(dense, sparse, alpha=0.8)
    fused_sparse_bias = SearchFusion.alpha_fusion(dense, sparse, alpha=0.2)

    assert fused_dense_bias[0].document.id == "dA"
    assert fused_sparse_bias[0].document.id == "dB"
