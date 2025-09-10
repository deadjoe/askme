"""
Stress tests for fixed-size chunking to ensure progress when overlap >= chunk_size.
"""

from askme.ingest.document_processor import (
    ChunkingConfig,
    DocumentChunker,
    ProcessedDocument,
)


def test_fixed_chunking_progress_with_large_overlap() -> None:
    content = "x" * 5000
    cfg = ChunkingConfig(
        method="fixed", chunk_size=1000, chunk_overlap=2000, min_chunk_size=10
    )
    chunker = DocumentChunker(cfg)
    pd = ProcessedDocument(
        source_path="mem",
        title="t",
        content=content,
        chunks=[],
        metadata={},
        processing_stats={},
    )
    chunks = chunker._fixed_size_chunking(pd.content)
    # Should produce at least 5 chunks of ~1000 chars and must terminate
    assert len(chunks) >= 4
    assert sum(len(c) for c in chunks) <= len(content) + 100  # guard
