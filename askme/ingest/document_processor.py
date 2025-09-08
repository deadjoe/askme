"""
Document processing and chunking functionality.
"""

import hashlib
import logging
import mimetypes
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import aiofiles
import pypdf
from bs4 import BeautifulSoup
from markdownify import markdownify

from askme.retriever.base import Document

logger = logging.getLogger(__name__)


@dataclass
class ProcessedDocument:
    """Document after processing and chunking."""

    source_path: str
    title: str
    content: str
    chunks: List[str]
    metadata: Dict[str, Any]
    processing_stats: Dict[str, Any]


@dataclass
class ChunkingConfig:
    """Configuration for document chunking."""

    method: str = "semantic"  # "fixed", "semantic", "recursive"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    min_chunk_size: int = 100
    max_chunk_size: int = 2000
    preserve_structure: bool = True


class DocumentProcessor(ABC):
    """Abstract base class for document processors."""

    @abstractmethod
    def can_process(self, file_path: Path) -> bool:
        """Check if this processor can handle the given file."""
        pass

    @abstractmethod
    async def process(
        self, file_path: Path, metadata: Optional[Dict[str, Any]] = None
    ) -> ProcessedDocument:
        """Process a document and return structured content."""
        pass


class PDFProcessor(DocumentProcessor):
    """PDF document processor using pypdf."""

    def can_process(self, file_path: Path) -> bool:
        """Check if file is a PDF."""
        return file_path.suffix.lower() == ".pdf"

    async def process(
        self, file_path: Path, metadata: Optional[Dict[str, Any]] = None
    ) -> ProcessedDocument:
        """Process PDF document."""
        try:
            with open(file_path, "rb") as file:
                pdf_reader = pypdf.PdfReader(file)

                # Extract text from all pages
                full_text = ""
                page_texts = []

                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    page_texts.append(page_text)
                    full_text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"

                # Extract PDF metadata
                pdf_info = pdf_reader.metadata if pdf_reader.metadata else {}

                doc_metadata = {
                    "source_type": "pdf",
                    "file_path": str(file_path),
                    "file_size": file_path.stat().st_size,
                    "page_count": len(pdf_reader.pages),
                    "created_at": datetime.now().isoformat(),
                    "pdf_title": pdf_info.get("/Title", ""),
                    "pdf_author": pdf_info.get("/Author", ""),
                    "pdf_subject": pdf_info.get("/Subject", ""),
                    "pdf_creator": pdf_info.get("/Creator", ""),
                    **(metadata or {}),
                }

                # Determine title
                title = (
                    doc_metadata.get("title")
                    or pdf_info.get("/Title")
                    or file_path.stem
                )

                processing_stats = {
                    "processor": "PDFProcessor",
                    "pages_processed": len(pdf_reader.pages),
                    "total_characters": len(full_text),
                    "processing_time": datetime.now().isoformat(),
                }

                return ProcessedDocument(
                    source_path=str(file_path),
                    title=title,
                    content=full_text.strip(),
                    chunks=[],  # Will be populated by chunker
                    metadata=doc_metadata,
                    processing_stats=processing_stats,
                )

        except Exception as e:
            logger.error(f"Failed to process PDF {file_path}: {e}")
            raise


class MarkdownProcessor(DocumentProcessor):
    """Markdown document processor."""

    def can_process(self, file_path: Path) -> bool:
        """Check if file is Markdown."""
        return file_path.suffix.lower() in [".md", ".markdown", ".mdown", ".mkd"]

    async def process(
        self, file_path: Path, metadata: Optional[Dict[str, Any]] = None
    ) -> ProcessedDocument:
        """Process Markdown document."""
        try:
            async with aiofiles.open(file_path, "r", encoding="utf-8") as file:
                content = await file.read()

            # Extract title from first H1 if available
            lines = content.split("\n")
            title = None
            for line in lines:
                if line.startswith("# "):
                    title = line[2:].strip()
                    break

            if not title:
                title = file_path.stem

            doc_metadata = {
                "source_type": "markdown",
                "file_path": str(file_path),
                "file_size": file_path.stat().st_size,
                "created_at": datetime.now().isoformat(),
                "encoding": "utf-8",
                **(metadata or {}),
            }

            processing_stats = {
                "processor": "MarkdownProcessor",
                "lines_processed": len(lines),
                "total_characters": len(content),
                "processing_time": datetime.now().isoformat(),
            }

            return ProcessedDocument(
                source_path=str(file_path),
                title=title,
                content=content,
                chunks=[],
                metadata=doc_metadata,
                processing_stats=processing_stats,
            )

        except Exception as e:
            logger.error(f"Failed to process Markdown {file_path}: {e}")
            raise


class HTMLProcessor(DocumentProcessor):
    """HTML document processor using BeautifulSoup."""

    def can_process(self, file_path: Path) -> bool:
        """Check if file is HTML."""
        return file_path.suffix.lower() in [".html", ".htm"]

    async def process(
        self, file_path: Path, metadata: Optional[Dict[str, Any]] = None
    ) -> ProcessedDocument:
        """Process HTML document."""
        try:
            async with aiofiles.open(file_path, "r", encoding="utf-8") as file:
                html_content = await file.read()

            # Parse HTML
            soup = BeautifulSoup(html_content, "html.parser")

            # Extract title
            title_tag = soup.find("title")
            title = title_tag.get_text().strip() if title_tag else file_path.stem

            # Convert to markdown for better structure preservation
            markdown_content = markdownify(html_content, heading_style="ATX")

            # Extract metadata from HTML
            meta_tags = soup.find_all("meta")
            html_meta = {}
            for tag in meta_tags:
                name = tag.get("name") or tag.get("property")
                content = tag.get("content")
                if name and content:
                    html_meta[name] = content

            doc_metadata = {
                "source_type": "html",
                "file_path": str(file_path),
                "file_size": file_path.stat().st_size,
                "created_at": datetime.now().isoformat(),
                "html_title": title,
                "html_meta": html_meta,
                "encoding": "utf-8",
                **(metadata or {}),
            }

            processing_stats = {
                "processor": "HTMLProcessor",
                "original_size": len(html_content),
                "markdown_size": len(markdown_content),
                "total_characters": len(markdown_content),
                "processing_time": datetime.now().isoformat(),
            }

            return ProcessedDocument(
                source_path=str(file_path),
                title=title,
                content=markdown_content,
                chunks=[],
                metadata=doc_metadata,
                processing_stats=processing_stats,
            )

        except Exception as e:
            logger.error(f"Failed to process HTML {file_path}: {e}")
            raise


class TextProcessor(DocumentProcessor):
    """Plain text document processor."""

    def can_process(self, file_path: Path) -> bool:
        """Check if file is plain text."""
        if file_path.suffix.lower() in [".txt", ".text"]:
            return True

        # Check MIME type for other potential text files
        mime_type, _ = mimetypes.guess_type(str(file_path))
        return bool(mime_type and mime_type.startswith("text/"))

    async def process(
        self, file_path: Path, metadata: Optional[Dict[str, Any]] = None
    ) -> ProcessedDocument:
        """Process plain text document."""
        try:
            # Try different encodings
            encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
            content = None
            used_encoding = None

            for encoding in encodings:
                try:
                    async with aiofiles.open(file_path, "r", encoding=encoding) as file:
                        content = await file.read()
                    used_encoding = encoding
                    break
                except UnicodeDecodeError:
                    continue

            if content is None:
                raise ValueError(
                    f"Could not decode file {file_path} with any supported encoding"
                )

            # Use filename as title
            title = file_path.stem

            doc_metadata = {
                "source_type": "text",
                "file_path": str(file_path),
                "file_size": file_path.stat().st_size,
                "created_at": datetime.now().isoformat(),
                "encoding": used_encoding,
                **(metadata or {}),
            }

            processing_stats = {
                "processor": "TextProcessor",
                "encoding_used": used_encoding,
                "total_characters": len(content),
                "processing_time": datetime.now().isoformat(),
            }

            return ProcessedDocument(
                source_path=str(file_path),
                title=title,
                content=content,
                chunks=[],
                metadata=doc_metadata,
                processing_stats=processing_stats,
            )

        except Exception as e:
            logger.error(f"Failed to process text file {file_path}: {e}")
            raise


class DocumentChunker:
    """Document chunking with multiple strategies."""

    def __init__(self, config: ChunkingConfig):
        self.config = config

    def chunk_document(self, processed_doc: ProcessedDocument) -> List[Dict[str, Any]]:
        """Chunk a processed document based on configuration."""
        content = processed_doc.content

        if self.config.method == "fixed":
            chunks = self._fixed_size_chunking(content)
        elif self.config.method == "semantic":
            chunks = self._semantic_chunking(content)
        elif self.config.method == "recursive":
            chunks = self._recursive_chunking(content)
        else:
            logger.warning(
                f"Unknown chunking method: {self.config.method}. Using fixed size."
            )
            chunks = self._fixed_size_chunking(content)

        # Create chunk documents with metadata
        chunk_docs = []
        for i, chunk_text in enumerate(chunks):
            chunk_id = f"{processed_doc.source_path}#chunk_{i}"

            chunk_metadata = {
                **processed_doc.metadata,
                "chunk_index": i,
                "chunk_count": len(chunks),
                "chunk_id": chunk_id,
                "source_document": processed_doc.source_path,
                "chunk_method": self.config.method,
            }

            chunk_docs.append(
                {"id": chunk_id, "content": chunk_text, "metadata": chunk_metadata}
            )

        return chunk_docs

    def _fixed_size_chunking(self, content: str) -> List[str]:
        """Simple fixed-size chunking with overlap."""
        chunks = []
        start = 0
        content_len = len(content)

        while start < content_len:
            end = start + self.config.chunk_size
            chunk = content[start:end]

            # Try to break at word boundaries
            if end < content_len:
                last_space = chunk.rfind(" ")
                if last_space > start + self.config.min_chunk_size:
                    chunk = chunk[:last_space]
                    end = start + last_space

            if len(chunk.strip()) >= self.config.min_chunk_size:
                chunks.append(chunk.strip())

            # Move start position with overlap
            start = end - self.config.chunk_overlap
            if start <= 0:
                break

        return chunks

    def _semantic_chunking(self, content: str) -> List[str]:
        """Semantic chunking based on paragraphs and structure."""
        # Split by double newlines (paragraphs)
        paragraphs = content.split("\n\n")

        chunks = []
        current_chunk = ""

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            # If adding this paragraph would exceed chunk size
            if (
                len(current_chunk) + len(paragraph) > self.config.chunk_size
                and current_chunk
            ):
                # Save current chunk if it meets minimum size
                if len(current_chunk) >= self.config.min_chunk_size:
                    chunks.append(current_chunk.strip())

                # Start new chunk with overlap from previous chunk
                if self.config.chunk_overlap > 0 and chunks:
                    overlap_text = current_chunk[-self.config.chunk_overlap :]
                    current_chunk = overlap_text + "\n\n" + paragraph
                else:
                    current_chunk = paragraph
            else:
                # Add paragraph to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph

        # Add final chunk
        if current_chunk and len(current_chunk.strip()) >= self.config.min_chunk_size:
            chunks.append(current_chunk.strip())

        return chunks

    def _recursive_chunking(self, content: str) -> List[str]:
        """Recursive chunking with multiple separators."""
        separators = ["\n\n", "\n", ". ", " "]
        return self._recursive_split(content, separators, 0)

    def _recursive_split(
        self, text: str, separators: List[str], sep_index: int
    ) -> List[str]:
        """Recursively split text using different separators."""
        if len(text) <= self.config.chunk_size:
            return [text] if text.strip() else []

        if sep_index >= len(separators):
            # No more separators, do character-based splitting
            return self._fixed_size_chunking(text)

        separator = separators[sep_index]
        parts = text.split(separator)

        chunks = []
        current_chunk = ""

        for part in parts:
            if not part.strip():
                continue

            potential_chunk = (
                current_chunk + separator + part if current_chunk else part
            )

            if len(potential_chunk) <= self.config.chunk_size:
                current_chunk = potential_chunk
            else:
                # Current chunk is good, save it
                if (
                    current_chunk
                    and len(current_chunk.strip()) >= self.config.min_chunk_size
                ):
                    chunks.append(current_chunk.strip())

                # If this part is too big, recursively split it
                if len(part) > self.config.chunk_size:
                    sub_chunks = self._recursive_split(part, separators, sep_index + 1)
                    chunks.extend(sub_chunks)
                    current_chunk = ""
                else:
                    current_chunk = part

        # Add final chunk
        if current_chunk and len(current_chunk.strip()) >= self.config.min_chunk_size:
            chunks.append(current_chunk.strip())

        return chunks


class DocumentProcessingPipeline:
    """Main document processing pipeline."""

    def __init__(self, chunking_config: Optional[ChunkingConfig] = None):
        self.processors = [
            PDFProcessor(),
            MarkdownProcessor(),
            HTMLProcessor(),
            TextProcessor(),
        ]
        self.chunker = DocumentChunker(chunking_config or ChunkingConfig())

    def get_processor(self, file_path: Path) -> Optional[DocumentProcessor]:
        """Get appropriate processor for file type."""
        for processor in self.processors:
            if processor.can_process(file_path):
                return processor
        return None

    async def process_file(
        self,
        file_path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ) -> List[Document]:
        """Process a single file and return chunked documents."""
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        processor = self.get_processor(file_path)
        if not processor:
            raise ValueError(
                f"No processor available for file type: {file_path.suffix}"
            )

        # Add tags to metadata
        if tags:
            metadata = metadata or {}
            metadata["tags"] = tags

        # Process document
        processed_doc = await processor.process(file_path, metadata)

        # Chunk document
        chunk_data = self.chunker.chunk_document(processed_doc)

        # Convert to Document objects
        documents = []
        for chunk in chunk_data:
            doc = Document(
                id=chunk["id"], content=chunk["content"], metadata=chunk["metadata"]
            )
            documents.append(doc)

        logger.info(f"Processed {file_path} into {len(documents)} chunks")
        return documents

    async def process_directory(
        self,
        dir_path: Union[str, Path],
        recursive: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        file_patterns: Optional[List[str]] = None,
    ) -> List[Document]:
        """Process all supported files in a directory."""
        dir_path = Path(dir_path)

        if not dir_path.exists() or not dir_path.is_dir():
            raise ValueError(f"Invalid directory: {dir_path}")

        # Default file patterns if not provided
        if file_patterns is None:
            file_patterns = ["*.pdf", "*.md", "*.markdown", "*.html", "*.htm", "*.txt"]

        all_documents = []
        processed_files = 0
        failed_files = []

        # Find all matching files
        for pattern in file_patterns:
            if recursive:
                files = dir_path.rglob(pattern)
            else:
                files = dir_path.glob(pattern)

            for file_path in files:
                try:
                    # Add directory context to metadata
                    file_metadata = {
                        **(metadata or {}),
                        "source_directory": str(dir_path),
                        "relative_path": str(file_path.relative_to(dir_path)),
                    }

                    documents = await self.process_file(file_path, file_metadata, tags)
                    all_documents.extend(documents)
                    processed_files += 1

                    logger.info(f"Processed file {processed_files}: {file_path.name}")

                except Exception as e:
                    logger.error(f"Failed to process {file_path}: {e}")
                    failed_files.append(str(file_path))

        logger.info(
            f"Directory processing complete: {processed_files} files processed, "
            f"{len(failed_files)} failed, {len(all_documents)} total chunks"
        )

        if failed_files:
            logger.warning(f"Failed files: {failed_files}")

        return all_documents
