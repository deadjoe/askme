"""Document ingestion service connecting processing pipeline with vector database."""

import asyncio
import logging
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

from askme.core.config import Settings
from askme.core.embeddings import BGEEmbeddingService, EmbeddingManager
from askme.ingest.document_processor import ChunkingConfig, DocumentProcessingPipeline
from askme.retriever.base import Document, VectorRetriever

logger = logging.getLogger(__name__)


def get_utc_now() -> datetime:
    """Get current UTC datetime."""
    return datetime.now(timezone.utc)


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
    # Enhanced statistics
    total_size_bytes: int = 0
    files_by_type: Optional[Dict[str, int]] = None
    processing_stages: Optional[Dict[str, float]] = None  # stage -> time_seconds

    def __post_init__(self) -> None:
        if self.files_by_type is None:
            self.files_by_type = {}
        if self.processing_stages is None:
            self.processing_stages = {}


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
    # Enhanced statistics
    stage_timings: Optional[Dict[str, float]] = None  # stage -> total_time_seconds
    avg_processing_time_per_file: float = 0.0
    avg_file_size_bytes: float = 0.0

    def __post_init__(self) -> None:
        if self.stage_timings is None:
            self.stage_timings = {}


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

        # Thread pool for CPU-intensive operations
        ingestion_cfg = getattr(settings.performance, "ingestion", None)
        configured_workers = getattr(ingestion_cfg, "max_workers", None)
        max_workers = self._resolve_max_workers(configured_workers)
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="askme-ingest",
        )
        logger.info("Ingestion thread pool initialized with %s workers", max_workers)

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

    def _resolve_max_workers(self, configured: Optional[int]) -> int:
        """Determine thread pool size with sensible defaults."""
        if configured is not None and configured > 0:
            return max(1, configured)

        cpu_count = os.cpu_count() or 2
        if cpu_count <= 4:
            return max(2, cpu_count)

        # Reserve a core for the event loop / other components and cap upper bound
        return min(max(4, cpu_count - 1), 32)

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
            created_at=get_utc_now(),
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

        # Create ingestion task
        task = IngestionTask(
            task_id=task_id,
            source_type="dir",
            source_path=str(dir_path),
            status=TaskStatus.QUEUED,
            created_at=get_utc_now(),
            total_files=0,
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
            "Started directory ingestion task %s: %s (recursive=%s, patterns=%s)",
            task_id,
            dir_path,
            recursive,
            ",".join(file_patterns),
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
            task.started_at = get_utc_now()
            processing_start = get_utc_now()

            # Collect file statistics
            file_size = file_path.stat().st_size
            file_extension = file_path.suffix.lower()
            task.total_size_bytes += file_size
            if task.files_by_type is not None:
                task.files_by_type[file_extension] = (
                    task.files_by_type.get(file_extension, 0) + 1
                )

            # Process the file in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            documents = await loop.run_in_executor(
                self._executor, self._process_file_sync, file_path, metadata, tags
            )

            # Track document processing time
            processing_end = get_utc_now()
            doc_processing_time = (processing_end - processing_start).total_seconds()
            if task.processing_stages is not None:
                task.processing_stages["document_processing"] = (
                    task.processing_stages.get("document_processing", 0)
                    + doc_processing_time
                )

            task.total_chunks = len(documents)

            # Generate embeddings and ingest with yield points
            embedding_start = get_utc_now()
            await self._ingest_documents_non_blocking(task, documents, overwrite)
            embedding_end = get_utc_now()

            # Track embedding and ingestion time
            embedding_time = (embedding_end - embedding_start).total_seconds()
            if task.processing_stages is not None:
                task.processing_stages["embedding_and_ingestion"] = (
                    task.processing_stages.get("embedding_and_ingestion", 0)
                    + embedding_time
                )

            # Mark as completed
            task.status = TaskStatus.COMPLETED
            task.completed_at = get_utc_now()
            task.processed_files = 1

            logger.info(
                f"File ingestion task {task.task_id} completed: "
                f"{len(documents)} chunks, {file_size} bytes, "
                f"{doc_processing_time: .2f}s processing, "
                f"{embedding_time: .2f}s embedding"
            )

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
            task.completed_at = get_utc_now()
            logger.error(f"File ingestion task {task.task_id} failed: {e}")
            raise
        finally:
            # Clean up running task
            if task.task_id in self._running_tasks:
                del self._running_tasks[task.task_id]

    def _process_file_sync(
        self,
        file_path: Path,
        metadata: Optional[Dict[str, Any]],
        tags: Optional[List[str]],
    ) -> List[Document]:
        """Synchronous wrapper for file processing to run in thread pool."""
        import asyncio

        # Create new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            return loop.run_until_complete(
                self.processing_pipeline.process_file(file_path, metadata, tags)
            )
        finally:
            loop.close()

    def _iter_directory_files(
        self,
        dir_path: Path,
        recursive: bool,
        file_patterns: List[str],
    ) -> Iterable[Path]:
        """Yield unique files under directory matching configured patterns."""
        seen: set[str] = set()

        for pattern in file_patterns:
            iterator = dir_path.rglob(pattern) if recursive else dir_path.glob(pattern)
            for file_path in iterator:
                if not file_path.is_file():
                    continue
                try:
                    key = str(file_path.resolve())
                except Exception:
                    key = str(file_path)
                if key in seen:
                    continue
                seen.add(key)
                yield file_path

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
            task.started_at = get_utc_now()
            files_processed = 0
            total_doc_processing = 0.0
            total_embedding = 0.0

            loop = asyncio.get_event_loop()

            for file_path in self._iter_directory_files(
                dir_path, recursive, file_patterns
            ):
                file_processing_start = get_utc_now()

                try:
                    relative_path = str(file_path.relative_to(dir_path))
                except ValueError:
                    relative_path = file_path.name

                file_metadata = {
                    **(metadata or {}),
                    "source_directory": str(dir_path),
                    "relative_path": relative_path,
                }

                documents = await loop.run_in_executor(
                    self._executor,
                    self._process_file_sync,
                    file_path,
                    file_metadata,
                    tags,
                )

                total_doc_processing += (
                    get_utc_now() - file_processing_start
                ).total_seconds()

                # Collect file statistics
                try:
                    file_size = file_path.stat().st_size
                except OSError:
                    file_size = 0
                task.total_size_bytes += file_size
                file_extension = file_path.suffix.lower()
                if task.files_by_type is not None:
                    task.files_by_type[file_extension] = (
                        task.files_by_type.get(file_extension, 0) + 1
                    )

                chunk_count = len(documents)
                task.total_chunks += chunk_count

                embedding_start = get_utc_now()
                await self._ingest_documents_non_blocking(task, documents, overwrite)
                total_embedding += (get_utc_now() - embedding_start).total_seconds()

                files_processed += 1
                task.processed_files = files_processed
                task.total_files = files_processed

            if task.processing_stages is not None:
                if total_doc_processing:
                    task.processing_stages["document_processing"] = (
                        task.processing_stages.get("document_processing", 0.0)
                        + total_doc_processing
                    )
                if total_embedding:
                    task.processing_stages["embedding_and_ingestion"] = (
                        task.processing_stages.get("embedding_and_ingestion", 0.0)
                        + total_embedding
                    )

            task.status = TaskStatus.COMPLETED
            task.completed_at = get_utc_now()

            logger.info(
                "Directory ingest %s done: files=%s chunks=%s size=%sB "
                "proc=%.2fs embed=%.2fs",
                task.task_id,
                files_processed,
                task.total_chunks,
                task.total_size_bytes,
                total_doc_processing,
                total_embedding,
            )

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
            task.completed_at = get_utc_now()
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
        await self._ingest_documents_non_blocking(task, documents, overwrite)

    async def _ingest_documents_non_blocking(
        self, task: IngestionTask, documents: List[Document], overwrite: bool
    ) -> None:
        """Ingest processed documents with yield points to keep API responsive."""
        if not documents:
            logger.warning("No documents to ingest")
            return

        batch_size = self.settings.performance.batch.embedding_batch_size

        # Process in batches with yield points
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            batch_texts = [doc.content for doc in batch]

            logger.debug(
                f"Processing embedding batch {i//batch_size + 1}: "
                f"{len(batch)} documents"
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
            if overwrite and not self.vector_retriever.supports_native_upsert:
                # Only explicitly delete if the backend doesn't support native upsert
                for doc in vector_docs:
                    try:
                        await self.vector_retriever.delete_document(doc.id)
                    except Exception:
                        pass  # Document might not exist

            # Insert into vector database
            # (native upsert if supported, otherwise insert after delete)
            try:
                # Add retry mechanism for transient failures
                max_retries = 3
                retry_count = 0
                while retry_count < max_retries:
                    try:
                        await self.vector_retriever.insert_documents(vector_docs)
                        task.processed_chunks += len(vector_docs)

                        logger.debug(
                            f"Ingested batch {i//batch_size + 1}: "
                            f"{len(vector_docs)} documents "
                            f"({task.processed_chunks}/{task.total_chunks} total)"
                        )
                        break  # Success, exit retry loop

                    except Exception as batch_error:
                        retry_count += 1
                        if retry_count >= max_retries:
                            logger.error(
                                f"Failed to insert document batch after "
                                f"{max_retries} retries: {batch_error}"
                            )
                            raise
                        else:
                            logger.warning(
                                f"Insert batch failed (attempt "
                                f"{retry_count}/{max_retries}): {batch_error}, retry..."
                            )
                            await asyncio.sleep(
                                min(retry_count * 0.5, 2.0)
                            )  # Exponential backoff

            except Exception as e:
                logger.error(f"Failed to insert document batch: {e}")
                raise

            # Yield control and add backpressure management
            await asyncio.sleep(0.01)  # Small delay for backpressure

            # Optional: Check if we should slow down based on system load
            # This could be expanded with more sophisticated backpressure
            if (i + batch_size) % (batch_size * 10) == 0:  # Every 10 batches
                await asyncio.sleep(0.1)  # Longer pause every 10 batches

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
                self._tasks[task_id].completed_at = get_utc_now()

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
            total_size_bytes = 0
            files_by_type: Dict[str, int] = {}
            stage_timings: Dict[str, float] = {}
            errors = []

            for task in self._tasks.values():
                if task.status == TaskStatus.COMPLETED:
                    total_files += task.processed_files
                    total_size_bytes += task.total_size_bytes

                    # Aggregate files by type
                    if task.files_by_type is not None:
                        for ext, count in task.files_by_type.items():
                            files_by_type[ext] = files_by_type.get(ext, 0) + count

                    # Aggregate stage timings
                    if task.processing_stages is not None:
                        for stage, timing in task.processing_stages.items():
                            stage_timings[stage] = (
                                stage_timings.get(stage, 0.0) + timing
                            )

                    if task.started_at and task.completed_at:
                        processing_time = (
                            task.completed_at - task.started_at
                        ).total_seconds()
                        total_processing_time += processing_time
                elif task.status == TaskStatus.FAILED and task.error_message:
                    errors.append(task.error_message)

            total_chunks = collection_stats.get("num_entities", 0)
            chunks_per_doc = total_chunks / total_files if total_files > 0 else 0.0
            avg_processing_time = (
                total_processing_time / total_files if total_files > 0 else 0.0
            )
            avg_file_size = total_size_bytes / total_files if total_files > 0 else 0.0

            return IngestionStats(
                total_documents=total_files,
                total_chunks=total_chunks,
                total_size_bytes=total_size_bytes,
                processing_time_seconds=total_processing_time,
                files_by_type=files_by_type,
                chunks_per_document=chunks_per_doc,
                errors=errors[-10:],  # Last 10 errors
                stage_timings=stage_timings,
                avg_processing_time_per_file=avg_processing_time,
                avg_file_size_bytes=avg_file_size,
            )

        except Exception as e:
            logger.error(f"Failed to get ingestion stats: {e}")
            raise

    async def cleanup_completed_tasks(self, older_than_hours: int = 24) -> int:
        """Clean up completed tasks older than specified hours."""
        cutoff_time = get_utc_now().timestamp() - (older_than_hours * 3600)

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
        """Shutdown the ingestion service gracefully with timeout."""
        try:
            logger.info("Starting ingestion service shutdown...")

            # Cancel all running tasks
            for task_id in list(self._running_tasks.keys()):
                await self.cancel_task(task_id)

            # Wait for tasks to complete with timeout
            if self._running_tasks:
                try:
                    await asyncio.wait_for(
                        asyncio.gather(
                            *self._running_tasks.values(), return_exceptions=True
                        ),
                        timeout=5.0,  # 5 second timeout
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        "Some ingestion tasks did not complete within timeout, "
                        "forcing shutdown"
                    )

            # Shutdown thread pool with timeout
            logger.info("Shutting down ingestion thread pool...")
            try:
                # Try modern shutdown with timeout (Python 3.9+)
                import inspect

                shutdown_sig = inspect.signature(self._executor.shutdown)
                if "timeout" in shutdown_sig.parameters:
                    self._executor.shutdown(wait=True, timeout=2.0)  # type: ignore
                    logger.info("Ingestion thread pool shut down gracefully")
                else:
                    # Fallback for older Python versions
                    self._executor.shutdown(wait=True)
                    logger.info("Ingestion thread pool shut down (no timeout support)")
            except Exception as e:
                logger.warning(f"Graceful shutdown failed: {e}, forcing shutdown")
                try:
                    self._executor.shutdown(wait=False)
                except Exception as force_e:
                    logger.error(
                        f"Failed to force shutdown ingestion thread pool: {force_e}"
                    )
            finally:
                # Mark as None to prevent further use
                self._executor = None  # type: ignore

            # Cleanup embedding service (with its own timeout)
            try:
                await asyncio.wait_for(
                    self.embedding_manager.embedding_service.cleanup(), timeout=3.0
                )
            except asyncio.TimeoutError:
                logger.warning("Embedding service cleanup timed out")
            except Exception as e:
                logger.warning(f"Error cleaning up embedding service: {e}")

            # Disconnect from vector database (with timeout)
            try:
                await asyncio.wait_for(self.vector_retriever.disconnect(), timeout=2.0)
            except asyncio.TimeoutError:
                logger.warning("Vector database disconnect timed out")
            except Exception as e:
                logger.warning(f"Error disconnecting from vector database: {e}")

            logger.info("Ingestion service shutdown completed")

        except Exception as e:
            logger.error(f"Error during ingestion service shutdown: {e}")
            # Don't re-raise to avoid blocking the shutdown process
