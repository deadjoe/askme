"""
Unit tests for document processing pipeline - Fixed version.
"""

import tempfile
from pathlib import Path
from typing import Any, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from askme.ingest.document_processor import (
    ChunkingConfig,
    DocumentChunker,
    DocumentProcessingPipeline,
    HTMLProcessor,
    MarkdownProcessor,
    PDFProcessor,
    ProcessedDocument,
    TextProcessor,
)
from askme.retriever.base import Document


class TestChunkingConfig:
    """Test chunking configuration."""

    def test_default_values(self: Any) -> None:
        """Test default chunking configuration values."""
        config = ChunkingConfig()
        assert config.method == "semantic"
        assert config.chunk_size == 1000
        assert config.chunk_overlap == 200
        assert config.min_chunk_size == 100
        assert config.max_chunk_size == 2000
        assert config.preserve_structure is True

    def test_custom_values(self: Any) -> None:
        """Test custom chunking configuration."""
        config = ChunkingConfig(
            method="fixed",
            chunk_size=500,
            chunk_overlap=100,
            min_chunk_size=50,
            max_chunk_size=1500,
            preserve_structure=False,
        )
        assert config.method == "fixed"
        assert config.chunk_size == 500
        assert config.chunk_overlap == 100
        assert config.min_chunk_size == 50
        assert config.max_chunk_size == 1500
        assert config.preserve_structure is False


class TestDocumentChunker:
    """Test document chunking functionality."""

    def test_fixed_size_chunking_normal(self: Any) -> None:
        """Test normal fixed-size chunking."""
        config = ChunkingConfig(
            method="fixed", chunk_size=100, chunk_overlap=20, min_chunk_size=10
        )
        chunker = DocumentChunker(config)

        content = "This is a test document. " * 20  # ~500 characters
        chunks = chunker._fixed_size_chunking(content)

        assert len(chunks) >= 4  # Should create multiple chunks
        assert all(len(chunk) >= config.min_chunk_size for chunk in chunks)

        # Test overlap behavior
        if len(chunks) > 1:
            # Some content should overlap between adjacent chunks
            overlap_found = False
            for i in range(len(chunks) - 1):
                current_end = chunks[i][-10:]  # Last 10 chars
                next_start = chunks[i + 1][:10]  # First 10 chars
                if any(word in next_start for word in current_end.split() if word):
                    overlap_found = True
                    break
            assert overlap_found or config.chunk_overlap == 0

    def test_fixed_size_chunking_large_overlap(self: Any) -> None:
        """Test fixed-size chunking with overlap >= chunk_size (regression test)."""
        config = ChunkingConfig(
            method="fixed", chunk_size=100, chunk_overlap=150, min_chunk_size=10
        )
        chunker = DocumentChunker(config)

        content = "x" * 1000  # Long content
        chunks = chunker._fixed_size_chunking(content)

        # Should not get stuck in infinite loop
        assert len(chunks) >= 5
        assert len(chunks) <= 15  # Should terminate reasonably

    def test_semantic_chunking_paragraphs(self: Any) -> None:
        """Test semantic chunking based on paragraphs."""
        config = ChunkingConfig(
            method="semantic", chunk_size=50, chunk_overlap=10, min_chunk_size=20
        )
        chunker = DocumentChunker(config)

        content = (
            "First paragraph with enough content to trigger chunking.\n\n"
            "Second paragraph with enough content to trigger chunking.\n\n"
            "Third paragraph with enough content to trigger chunking and exceed the chunk size limit significantly."
        )
        chunks = chunker._semantic_chunking(content)

        assert len(chunks) >= 2
        assert all(len(chunk) >= config.min_chunk_size for chunk in chunks)
        # Paragraphs should be preserved in chunks
        assert any("First paragraph" in chunk for chunk in chunks)

    def test_recursive_chunking(self: Any) -> None:
        """Test recursive chunking with multiple separators."""
        config = ChunkingConfig(
            method="recursive", chunk_size=100, chunk_overlap=20, min_chunk_size=10
        )
        chunker = DocumentChunker(config)

        content = (
            "Section 1.\n\nParagraph 1. Sentence 1. Sentence 2.\n\n"
            "Paragraph 2. More content here."
        )
        chunks = chunker._recursive_chunking(content)

        assert len(chunks) >= 1
        assert all(len(chunk) >= config.min_chunk_size for chunk in chunks)

    def test_chunk_document_metadata(self: Any) -> None:
        """Test chunk document with proper metadata."""
        config = ChunkingConfig(method="fixed", chunk_size=50, min_chunk_size=10)
        chunker = DocumentChunker(config)

        processed_doc = ProcessedDocument(
            source_path="test.txt",
            title="Test Document",
            content="This is a test document with some content for chunking.",
            chunks=[],
            metadata={"author": "test", "tags": ["test"]},
            processing_stats={"processor": "TestProcessor"},
        )

        chunk_docs = chunker.chunk_document(processed_doc)

        assert len(chunk_docs) >= 1
        for i, chunk_doc in enumerate(chunk_docs):
            assert chunk_doc["id"] == f"test.txt#chunk_{i}"
            assert len(chunk_doc["content"]) >= config.min_chunk_size

            metadata = chunk_doc["metadata"]
            assert metadata["chunk_index"] == i
            assert metadata["chunk_count"] == len(chunk_docs)
            assert metadata["source_document"] == "test.txt"
            assert metadata["chunk_method"] == "fixed"
            assert metadata["author"] == "test"
            assert metadata["tags"] == ["test"]


class TestPDFProcessor:
    """Test PDF document processor."""

    def test_can_process_pdf(self: Any) -> None:
        """Test PDF file recognition."""
        processor = PDFProcessor()
        assert processor.can_process(Path("test.pdf")) is True
        assert processor.can_process(Path("test.PDF")) is True
        assert processor.can_process(Path("test.txt")) is False

    @pytest.mark.asyncio
    async def test_process_pdf_mock(self: Any) -> None:
        """Test PDF processing with mocked pypdf."""
        processor = PDFProcessor()

        # Mock PDF reader and metadata
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Test PDF content"

        mock_reader = MagicMock()
        mock_reader.pages = [mock_page, mock_page]  # 2 pages
        mock_reader.metadata = {"/Title": "Test PDF", "/Author": "Test Author"}

        with patch("builtins.open", MagicMock()):
            with patch("askme.ingest.document_processor.pypdf.PdfReader") as mock_pdf:
                mock_pdf.return_value = mock_reader

                # Mock Path.stat() for file size
                with patch.object(Path, "stat") as mock_stat:
                    mock_stat.return_value.st_size = 1024

                    result = await processor.process(Path("test.pdf"))

                    assert result.title == "Test PDF"
                    assert "Test PDF content" in result.content
                    assert result.metadata["source_type"] == "pdf"
                    assert result.metadata["page_count"] == 2
                    assert result.metadata["pdf_title"] == "Test PDF"
                    assert result.metadata["pdf_author"] == "Test Author"


class TestMarkdownProcessor:
    """Test Markdown document processor."""

    def test_can_process_markdown(self: Any) -> None:
        """Test Markdown file recognition."""
        processor = MarkdownProcessor()
        assert processor.can_process(Path("test.md")) is True
        assert processor.can_process(Path("test.markdown")) is True
        assert processor.can_process(Path("test.mdown")) is True
        assert processor.can_process(Path("test.mkd")) is True
        assert processor.can_process(Path("test.txt")) is False

    @pytest.mark.asyncio
    async def test_process_markdown_with_title(self: Any) -> None:
        """Test Markdown processing with title extraction."""
        processor = MarkdownProcessor()

        content = (
            "# Main Title\n\nThis is markdown content.\n\n## Section\n\nMore content."
        )

        # Create a proper async context manager mock
        mock_file = MagicMock()
        mock_file.read = AsyncMock(return_value=content)

        async_context_manager = AsyncMock()
        async_context_manager.__aenter__.return_value = mock_file

        with patch(
            "askme.ingest.document_processor.aiofiles.open",
            return_value=async_context_manager,
        ):
            with patch.object(Path, "stat") as mock_stat:
                mock_stat.return_value.st_size = len(content)

                result = await processor.process(Path("test.md"))

                assert result.title == "Main Title"
                assert result.content == content
                assert result.metadata["source_type"] == "markdown"
                assert result.metadata["encoding"] == "utf-8"

    @pytest.mark.asyncio
    async def test_process_markdown_without_title(self: Any) -> None:
        """Test Markdown processing without H1 title."""
        processor = MarkdownProcessor()

        content = (
            "This is markdown content without a title.\n\n## Section\n\nMore content."
        )

        mock_file = MagicMock()
        mock_file.read = AsyncMock(return_value=content)

        async_context_manager = AsyncMock()
        async_context_manager.__aenter__.return_value = mock_file

        with patch(
            "askme.ingest.document_processor.aiofiles.open",
            return_value=async_context_manager,
        ):
            with patch.object(Path, "stat") as mock_stat:
                mock_stat.return_value.st_size = len(content)

                result = await processor.process(Path("test.md"))

                assert result.title == "test"  # Uses filename stem
                assert result.content == content


class TestHTMLProcessor:
    """Test HTML document processor."""

    def test_can_process_html(self: Any) -> None:
        """Test HTML file recognition."""
        processor = HTMLProcessor()
        assert processor.can_process(Path("test.html")) is True
        assert processor.can_process(Path("test.htm")) is True
        assert processor.can_process(Path("test.txt")) is False

    @pytest.mark.asyncio
    async def test_process_html_with_title(self: Any) -> None:
        """Test HTML processing with title extraction."""
        processor = HTMLProcessor()

        html_content = """
        <html>
        <head>
            <title>Test HTML Page</title>
            <meta name="description" content="Test description">
            <meta name="keywords" content="test,html">
        </head>
        <body>
            <h1>Main Heading</h1>
            <p>This is HTML content.</p>
        </body>
        </html>
        """

        mock_file = MagicMock()
        mock_file.read = AsyncMock(return_value=html_content)

        async_context_manager = AsyncMock()
        async_context_manager.__aenter__.return_value = mock_file

        with patch(
            "askme.ingest.document_processor.aiofiles.open",
            return_value=async_context_manager,
        ):
            with patch.object(Path, "stat") as mock_stat:
                mock_stat.return_value.st_size = len(html_content)

                result = await processor.process(Path("test.html"))

                assert result.title == "Test HTML Page"
                assert "Main Heading" in result.content
                assert result.metadata["source_type"] == "html"
                assert result.metadata["html_title"] == "Test HTML Page"
                assert "description" in result.metadata["html_meta"]


class TestTextProcessor:
    """Test plain text document processor."""

    def test_can_process_text(self: Any) -> None:
        """Test text file recognition."""
        processor = TextProcessor()
        assert processor.can_process(Path("test.txt")) is True
        assert processor.can_process(Path("test.text")) is True

        # Should handle files with text/ MIME types
        with patch("askme.ingest.document_processor.mimetypes.guess_type") as mock_mime:
            mock_mime.return_value = ("text/plain", None)
            assert processor.can_process(Path("test.log")) is True

            mock_mime.return_value = ("application/pdf", None)
            assert processor.can_process(Path("test.log")) is False

    @pytest.mark.asyncio
    async def test_process_text_utf8(self: Any) -> None:
        """Test text processing with UTF-8 encoding."""
        processor = TextProcessor()

        content = "This is plain text content.\nWith multiple lines.\nAnd unicode: café"

        mock_file = MagicMock()
        mock_file.read = AsyncMock(return_value=content)

        async_context_manager = AsyncMock()
        async_context_manager.__aenter__.return_value = mock_file

        with patch(
            "askme.ingest.document_processor.aiofiles.open",
            return_value=async_context_manager,
        ):
            with patch.object(Path, "stat") as mock_stat:
                mock_stat.return_value.st_size = len(content)

                result = await processor.process(Path("test.txt"))

                assert result.title == "test"
                assert result.content == content
                assert result.metadata["source_type"] == "text"
                assert result.metadata["encoding"] == "utf-8"

    @pytest.mark.asyncio
    async def test_process_text_encoding_fallback(self: Any) -> None:
        """Test text processing with encoding fallback."""
        processor = TextProcessor()

        content = "This is plain text content."

        def mock_open_side_effect(*args: Any, **kwargs: Any) -> AsyncMock:
            mock_file = MagicMock()
            async_context_manager = AsyncMock()

            if kwargs.get("encoding") == "utf-8":
                mock_file.read = AsyncMock(
                    side_effect=UnicodeDecodeError("utf-8", b"", 0, 1, "error")
                )
            else:
                mock_file.read = AsyncMock(return_value=content)

            async_context_manager.__aenter__.return_value = mock_file
            return async_context_manager

        with patch(
            "askme.ingest.document_processor.aiofiles.open",
            side_effect=mock_open_side_effect,
        ):
            with patch.object(Path, "stat") as mock_stat:
                mock_stat.return_value.st_size = len(content)

                result = await processor.process(Path("test.txt"))

                assert result.content == content
                assert result.metadata["encoding"] in ["utf-8-sig", "latin-1", "cp1252"]


class TestDocumentProcessingPipeline:
    """Test main document processing pipeline."""

    def test_get_processor_selection(self: Any) -> None:
        """Test processor selection based on file extension."""
        pipeline = DocumentProcessingPipeline()

        assert isinstance(pipeline.get_processor(Path("test.pdf")), PDFProcessor)
        assert isinstance(pipeline.get_processor(Path("test.md")), MarkdownProcessor)
        assert isinstance(pipeline.get_processor(Path("test.html")), HTMLProcessor)
        assert isinstance(pipeline.get_processor(Path("test.txt")), TextProcessor)
        assert pipeline.get_processor(Path("test.unknown")) is None

    @pytest.mark.asyncio
    async def test_process_file_with_metadata(self: Any) -> None:
        """Test file processing with custom metadata and tags."""
        pipeline = DocumentProcessingPipeline()

        # Mock a longer text file that will meet min_chunk_size requirements
        content = (
            "This is test content for processing. " * 20
        )  # Make it long enough to chunk
        mock_file = MagicMock()
        mock_file.read = AsyncMock(return_value=content)

        async_context_manager = AsyncMock()
        async_context_manager.__aenter__.return_value = mock_file

        with patch(
            "askme.ingest.document_processor.aiofiles.open",
            return_value=async_context_manager,
        ):
            with patch.object(Path, "exists", return_value=True):
                with patch.object(Path, "stat") as mock_stat:
                    mock_stat.return_value.st_size = len(content)

                    documents = await pipeline.process_file(
                        Path("test.txt"),
                        metadata={"author": "test"},
                        tags=["important", "test"],
                    )

                    assert len(documents) >= 1
                    assert all(isinstance(doc, Document) for doc in documents)

                    # Check metadata propagation
                    first_doc = documents[0]
                    assert first_doc.metadata["author"] == "test"
                    assert first_doc.metadata["tags"] == ["important", "test"]
                    assert first_doc.metadata["source_type"] == "text"

    @pytest.mark.asyncio
    async def test_process_file_not_found(self: Any) -> None:
        """Test processing non-existent file."""
        pipeline = DocumentProcessingPipeline()

        with patch.object(Path, "exists", return_value=False):
            with pytest.raises(FileNotFoundError):
                await pipeline.process_file("nonexistent.txt")

    @pytest.mark.asyncio
    async def test_process_file_unsupported_type(self: Any) -> None:
        """Test processing unsupported file type."""
        pipeline = DocumentProcessingPipeline()

        with patch.object(Path, "exists", return_value=True):
            with pytest.raises(ValueError, match="No processor available"):
                await pipeline.process_file("test.unknown")

    @pytest.mark.asyncio
    async def test_process_directory_recursive(self: Any) -> None:
        """Test directory processing with recursive search."""
        pipeline = DocumentProcessingPipeline()

        # Mock directory structure
        test_files = [Path("dir/file1.txt"), Path("dir/subdir/file2.md")]

        # Make content long enough to meet min_chunk_size requirements
        content = "Test content for directory processing. " * 25
        mock_file = MagicMock()
        mock_file.read = AsyncMock(return_value=content)

        async_context_manager = AsyncMock()
        async_context_manager.__aenter__.return_value = mock_file

        with patch.object(Path, "exists", return_value=True):
            with patch.object(Path, "is_dir", return_value=True):
                with patch.object(Path, "rglob") as mock_rglob:
                    with patch.object(Path, "stat") as mock_stat:
                        mock_stat.return_value.st_size = len(content)

                        # Mock file discovery
                        def rglob_side_effect(pattern: str) -> List[Path]:
                            if pattern == "*.txt":
                                return [test_files[0]]
                            elif pattern == "*.md":
                                return [test_files[1]]
                            else:
                                return []

                        mock_rglob.side_effect = rglob_side_effect

                        with patch(
                            "askme.ingest.document_processor.aiofiles.open",
                            return_value=async_context_manager,
                        ):
                            with patch.object(Path, "relative_to") as mock_relative:
                                mock_relative.return_value = Path("relative_path")

                                documents = await pipeline.process_directory(
                                    Path("dir"),
                                    recursive=True,
                                    tags=["batch"],
                                    file_patterns=["*.txt", "*.md"],
                                )

                                assert len(documents) >= 2  # At least 2 files processed

                                # Check directory context in metadata
                                for doc in documents:
                                    assert doc.metadata["source_directory"] == str(
                                        Path("dir")
                                    )
                                    assert doc.metadata["tags"] == ["batch"]

    @pytest.mark.asyncio
    async def test_process_directory_invalid_path(self: Any) -> None:
        """Test processing invalid directory path."""
        pipeline = DocumentProcessingPipeline()

        with patch.object(Path, "exists", return_value=False):
            with pytest.raises(ValueError, match="Invalid directory"):
                await pipeline.process_directory("nonexistent")
