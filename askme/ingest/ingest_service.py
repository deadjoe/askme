"""
Document ingestion service connecting processing pipeline with vector database.
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from askme.core.config import Settings
from askme.core.embeddings import BGEEmbeddingService, EmbeddingManager
from askme.ingest.document_processor import ChunkingConfig, DocumentProcessingPipeline
from askme.retriever.base import Document, VectorRetriever

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    """Ingestion task status."""

    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class IngestionTask:
    """Ingestion task tracking."""

    task_id: str
    source_type: str  # "file", "dir", "url"
    source_path: str
    status: TaskStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    total_files: Optional[int] = None
    processed_files: int = 0
    total_chunks: int = 0
    processed_chunks: int = 0
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class IngestionStats:
    """Ingestion statistics."""

    total_documents: int
    total_chunks: int
    total_size_bytes: int
    processing_time_seconds: float
    files_by_type: Dict[str, int]
    chunks_per_document: float
    errors: List[str]


class IngestionService:
    """Service for ingesting documents into the vector database."""

    def __init__(
        self,
        vector_retriever: VectorRetriever,
        embedding_service: BGEEmbeddingService,
        settings: Settings,
    ):
        self.vector_retriever = vector_retriever
        self.embedding_manager = EmbeddingManager(embedding_service)
        self.settings = settings

        # Initialize processing pipeline
        chunking_config = ChunkingConfig(
            method=settings.document.chunking.method,
            chunk_size=settings.document.chunking.chunk_size,
            chunk_overlap=settings.document.chunking.chunk_overlap,
            min_chunk_size=settings.document.chunking.min_chunk_size,
            max_chunk_size=settings.document.chunking.max_chunk_size,
        )
        self.processing_pipeline = DocumentProcessingPipeline(chunking_config)

        # Task tracking
        self._tasks: Dict[str, IngestionTask] = {}
        self._running_tasks: Dict[str, asyncio.Task] = {}

    async def initialize(self) -> None:
        """Initialize the ingestion service."""
        try:
            # Initialize vector database connection
            await self.vector_retriever.connect()

            # Initialize embedding service
            await self.embedding_manager.embedding_service.initialize()

            # Create collection if it doesn't exist
            await self.vector_retriever.create_collection(
                dimension=self.settings.embedding.dimension, metric="cosine"
            )

            logger.info("Ingestion service initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize ingestion service: {e}")
            raise

    async def ingest_file(
        self,
        file_path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        overwrite: bool = False,
    ) -> str:
        """
        Ingest a single file.

        Args:
            file_path: Path to the file to ingest
            metadata: Additional metadata for the file
            tags: Tags to associate with the file
            overwrite: Whether to overwrite existing documents

        Returns:
            Task ID for tracking the ingestion
        """
        task_id = str(uuid.uuid4())

        # Create ingestion task
        task = IngestionTask(
            task_id=task_id,
            source_type="file",
            source_path=str(file_path),
            status=TaskStatus.QUEUED,
            created_at=datetime.utcnow(),
            total_files=1,
            metadata={
                "user_metadata": metadata or {},
                "tags": tags or [],
                "overwrite": overwrite,
            },
        )

        self._tasks[task_id] = task

        # Start processing task
        processing_task = asyncio.create_task(
            self._process_file_task(task, Path(file_path), metadata, tags, overwrite)
        )
        self._running_tasks[task_id] = processing_task

        logger.info(f"Started file ingestion task {task_id}: {file_path}")
        return task_id

    async def ingest_directory(
        self,
        dir_path: Union[str, Path],
        recursive: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        file_patterns: Optional[List[str]] = None,
        overwrite: bool = False,
    ) -> str:
        """
        Ingest all supported files in a directory.

        Args:
            dir_path: Path to the directory to ingest
            recursive: Whether to process subdirectories
            metadata: Additional metadata for files
            tags: Tags to associate with files
            file_patterns: File patterns to match (defaults to supported formats)
            overwrite: Whether to overwrite existing documents

        Returns:
            Task ID for tracking the ingestion
        """
        task_id = str(uuid.uuid4())

        # Count files to process
        dir_path = Path(dir_path)
        if file_patterns is None:
            file_patterns = [
                f"*.{fmt}" for fmt in self.settings.document.supported_formats
            ]

        file_count = 0
        for pattern in file_patterns:
            if recursive:
                files = list(dir_path.rglob(pattern))
            else:
                files = list(dir_path.glob(pattern))
            file_count += len(files)

        # Create ingestion task
        task = IngestionTask(
            task_id=task_id,
            source_type="dir",
            source_path=str(dir_path),
            status=TaskStatus.QUEUED,
            created_at=datetime.utcnow(),
            total_files=file_count,
            metadata={
                "user_metadata": metadata or {},
                "tags": tags or [],
                "recursive": recursive,
                "file_patterns": file_patterns,
                "overwrite": overwrite,
            },
        )

        self._tasks[task_id] = task

        # Start processing task
        processing_task = asyncio.create_task(
            self._process_directory_task(
                task, dir_path, recursive, metadata, tags, file_patterns, overwrite
            )
        )
        self._running_tasks[task_id] = processing_task

        logger.info(
            f"Started directory ingestion task {task_id}: {dir_path} ({file_count} files)"
        )
        return task_id

    async def _process_file_task(
        self,
        task: IngestionTask,
        file_path: Path,
        metadata: Optional[Dict[str, Any]],
        tags: Optional[List[str]],
        overwrite: bool,
    ) -> None:
        """Process a single file ingestion task."""
        try:
            task.status = TaskStatus.PROCESSING
            task.started_at = datetime.utcnow()

            # Process the file
            documents = await self.processing_pipeline.process_file(
                file_path, metadata, tags
            )

            task.total_chunks = len(documents)

            # Generate embeddings and ingest
            await self._ingest_documents(task, documents, overwrite)

            # Mark as completed
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.utcnow()
            task.processed_files = 1

            logger.info(
                f"File ingestion task {task.task_id} completed: {len(documents)} chunks"
            )

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
            task.completed_at = datetime.utcnow()
            logger.error(f"File ingestion task {task.task_id} failed: {e}")
            raise
        finally:
            # Clean up running task
            if task.task_id in self._running_tasks:
                del self._running_tasks[task.task_id]

    async def _process_directory_task(
        self,
        task: IngestionTask,
        dir_path: Path,
        recursive: bool,
        metadata: Optional[Dict[str, Any]],
        tags: Optional[List[str]],
        file_patterns: List[str],
        overwrite: bool,
    ) -> None:
        """Process a directory ingestion task."""
        try:
            task.status = TaskStatus.PROCESSING
            task.started_at = datetime.utcnow()

            # Process the directory
            documents = await self.processing_pipeline.process_directory(
                dir_path, recursive, metadata, tags, file_patterns
            )

            task.total_chunks = len(documents)

            # Group documents by source file for progress tracking
            files_processed = set()
            for doc in documents:
                source_file = doc.metadata.get("source_document", "unknown")
                files_processed.add(source_file)

            task.processed_files = len(files_processed)

            # Generate embeddings and ingest
            await self._ingest_documents(task, documents, overwrite)

            # Mark as completed
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.utcnow()

            logger.info(
                f"Directory ingestion task {task.task_id} completed: "
                f"{task.processed_files} files, {len(documents)} chunks"
            )

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
            task.completed_at = datetime.utcnow()
            logger.error(f"Directory ingestion task {task.task_id} failed: {e}")
            raise
        finally:
            # Clean up running task
            if task.task_id in self._running_tasks:
                del self._running_tasks[task.task_id]

    async def _ingest_documents(
        self, task: IngestionTask, documents: List[Document], overwrite: bool
    ) -> None:
        """Ingest processed documents into the vector database."""
        if not documents:
            logger.warning("No documents to ingest")
            return

        batch_size = self.settings.performance.batch.embedding_batch_size

        # Process in batches
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            batch_texts = [doc.content for doc in batch]

            logger.debug(
                f"Processing embedding batch {i//batch_size + 1}: {len(batch)} documents"
            )

            # Generate embeddings
            embeddings = await self.embedding_manager.get_document_embeddings(
                batch_texts, use_cache=True, batch_size=len(batch_texts)
            )

            # Update documents with embeddings
            vector_docs = []
            for doc, embedding in zip(batch, embeddings):
                doc.embedding = embedding["dense_embedding"]
                doc.sparse_embedding = embedding["sparse_embedding"]
                vector_docs.append(doc)

            # Handle overwrite logic
            if overwrite:
                # Delete existing documents with same IDs
                for doc in vector_docs:
                    try:
                        await self.vector_retriever.delete_document(doc.id)
                    except Exception:
                        pass  # Document might not exist

            # Insert into vector database
            try:
                await self.vector_retriever.insert_documents(vector_docs)
                task.processed_chunks += len(vector_docs)

                logger.debug(
                    f"Ingested batch {i//batch_size + 1}: {len(vector_docs)} documents "
                    f"({task.processed_chunks}/{task.total_chunks} total)"
                )

            except Exception as e:
                logger.error(f"Failed to insert document batch: {e}")
                raise

    async def get_task_status(self, task_id: str) -> Optional[IngestionTask]:
        """Get the status of an ingestion task."""
        return self._tasks.get(task_id)

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a running ingestion task."""
        if task_id not in self._running_tasks:
            return False

        try:
            # Cancel the asyncio task
            self._running_tasks[task_id].cancel()

            # Update task status
            if task_id in self._tasks:
                self._tasks[task_id].status = TaskStatus.CANCELLED
                self._tasks[task_id].completed_at = datetime.utcnow()

            logger.info(f"Cancelled ingestion task {task_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to cancel task {task_id}: {e}")
            return False

    async def get_ingestion_stats(self) -> IngestionStats:
        """Get overall ingestion statistics."""
        try:
            # Get collection stats from vector database
            collection_stats = await self.vector_retriever.get_collection_stats()

            # Calculate stats from completed tasks
            total_files = 0
            total_processing_time: float = 0.0
            files_by_type: Dict[str, int] = {}
            errors = []

            for task in self._tasks.values():
                if task.status == TaskStatus.COMPLETED:
                    total_files += task.processed_files

                    if task.started_at and task.completed_at:
                        processing_time = (
                            task.completed_at - task.started_at
                        ).total_seconds()
                        total_processing_time += processing_time
                elif task.status == TaskStatus.FAILED and task.error_message:
                    errors.append(task.error_message)

            total_chunks = collection_stats.get("num_entities", 0)
            chunks_per_doc = total_chunks / total_files if total_files > 0 else 0.0

            return IngestionStats(
                total_documents=total_files,
                total_chunks=total_chunks,
                total_size_bytes=0,  # TODO: Calculate from task metadata
                processing_time_seconds=total_processing_time,
                files_by_type=files_by_type,
                chunks_per_document=chunks_per_doc,
                errors=errors[-10:],  # Last 10 errors
            )

        except Exception as e:
            logger.error(f"Failed to get ingestion stats: {e}")
            raise

    async def cleanup_completed_tasks(self, older_than_hours: int = 24) -> int:
        """Clean up completed tasks older than specified hours."""
        cutoff_time = datetime.utcnow().timestamp() - (older_than_hours * 3600)

        cleaned_count = 0
        tasks_to_remove = []

        for task_id, task in self._tasks.items():
            if (
                task.status
                in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]
                and task.created_at.timestamp() < cutoff_time
            ):
                tasks_to_remove.append(task_id)

        for task_id in tasks_to_remove:
            del self._tasks[task_id]
            cleaned_count += 1

        logger.info(f"Cleaned up {cleaned_count} completed tasks")
        return cleaned_count

    async def shutdown(self) -> None:
        """Shutdown the ingestion service."""
        try:
            # Cancel all running tasks
            for task_id in list(self._running_tasks.keys()):
                await self.cancel_task(task_id)

            # Wait for tasks to complete
            if self._running_tasks:
                await asyncio.gather(
                    *self._running_tasks.values(), return_exceptions=True
                )

            # Cleanup embedding service
            await self.embedding_manager.embedding_service.cleanup()

            # Disconnect from vector database
            await self.vector_retriever.disconnect()

            logger.info("Ingestion service shutdown completed")

        except Exception as e:
            logger.error(f"Error during ingestion service shutdown: {e}")
            raise
