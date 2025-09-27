"""
FastAPI application for askme hybrid RAG system.
"""

import logging
import os
import signal
import sys
from contextlib import asynccontextmanager
from types import FrameType
from typing import Any, AsyncGenerator, cast

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from askme.api.routes import evaluation, health, ingest, query
from askme.core.config import get_settings
from askme.core.embeddings import BGEEmbeddingService
from askme.core.logging_config import setup_logging
from askme.generation.generator import (
    BaseGenerator,
    LocalOllamaGenerator,
    OpenAIChatGenerator,
    SimpleTemplateGenerator,
)
from askme.ingest.ingest_service import IngestionService
from askme.rerank.rerank_service import RerankingService
from askme.retriever.base import VectorRetriever
from askme.retriever.milvus_retriever import MilvusRetriever
from askme.retriever.weaviate_retriever import WeaviateRetriever


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan management."""
    logger = logging.getLogger(__name__)

    # Startup
    logger.info("Starting askme API server")
    settings = get_settings()

    # Initialize components
    try:
        # Build services (lazy heavy loads)
        embedding_service = BGEEmbeddingService(settings.embedding)

        # Choose retriever backend
        retriever: VectorRetriever
        if settings.vector_backend.lower() == "weaviate":
            retriever_cfg = {
                "url": settings.database.weaviate.url,
                "api_key": settings.database.weaviate.api_key,
                "class_name": settings.database.weaviate.class_name,
                "dimension": settings.embedding.dimension,
            }
            retriever = WeaviateRetriever(retriever_cfg)
        else:
            retriever_cfg = {
                "host": settings.database.milvus.host,
                "port": settings.database.milvus.port,
                "username": settings.database.milvus.username,
                "password": settings.database.milvus.password,
                "secure": settings.database.milvus.secure,
                "collection_name": settings.database.milvus.collection_name,
                "dimension": settings.embedding.dimension,
            }
            retriever = MilvusRetriever(retriever_cfg)

        # Unified Cohere logic: enable via config or env var, only when API key provided
        enable_cohere_env = os.getenv("ASKME_ENABLE_COHERE", "0") in {
            "1",
            "true",
            "True",
        }
        cohere_api_key = (
            os.getenv("COHERE_API_KEY")
            if (enable_cohere_env or settings.rerank.cohere_enabled)
            else None
        )
        # Force enable cohere_enabled if API key detected, disable otherwise
        try:
            settings.rerank.cohere_enabled = cohere_api_key is not None
        except Exception:
            pass
        reranking_service = RerankingService(settings.rerank, cohere_api_key)

        ingestion_service = IngestionService(retriever, embedding_service, settings)

        # Generator: prefer local Ollama when explicitly enabled, else simple
        generator: BaseGenerator
        enable_ollama = os.getenv("ASKME_ENABLE_OLLAMA", "0") in {"1", "true", "True"}
        if enable_ollama or settings.generation.provider.lower() == "ollama":
            from loguru import logger as loguru_logger

            if not settings.generation.ollama_model:
                default_model = "gpt-oss:20b"
                loguru_logger.warning(
                    "ollama_model not configured, defaulting to {}", default_model
                )
                settings.generation.ollama_model = default_model

            loguru_logger.info(
                "Initializing Ollama generator with config: {}",
                settings.generation.model_dump(),
            )
            generator = LocalOllamaGenerator(settings.generation)
        elif settings.generation.provider.lower() == "openai":
            generator = OpenAIChatGenerator(settings.generation)
        else:
            generator = SimpleTemplateGenerator(settings.generation)

        # Attach to app state
        app.state.embedding_service = embedding_service
        app.state.retriever = retriever
        app.state.reranking_service = reranking_service
        app.state.ingestion_service = ingestion_service
        app.state.generator = generator

        logger.info("askme API server components initialized")

    except Exception as e:
        logger.error(f"Failed to initialize askme components: {e}")
        raise

    # Eager initialize heavy services; allow env var skip for fast startup
    skip_heavy = os.getenv("ASKME_SKIP_HEAVY_INIT", "0") in {"1", "true", "True"}
    if not skip_heavy:
        try:
            if hasattr(app.state, "ingestion_service"):
                await app.state.ingestion_service.initialize()
            if hasattr(app.state, "reranking_service"):
                await app.state.reranking_service.initialize()
            # Embedding model is initialized by ingestion_service.initialize();
            # generator is lightweight; LocalOllama uses HTTP on demand.
        except Exception as e:
            logging.getLogger(__name__).warning(
                f"Service initialization encountered issues: {e}"
            )
    else:
        logging.getLogger(__name__).info(
            "Skipping heavy service initialization (ASKME_SKIP_HEAVY_INIT=1)"
        )

    yield

    # Shutdown
    logger.info("Shutting down askme API server")
    try:
        if hasattr(app.state, "ingestion_service"):
            await app.state.ingestion_service.shutdown()
        if hasattr(app.state, "reranking_service"):
            await app.state.reranking_service.cleanup()
        if hasattr(app.state, "embedding_service"):
            await app.state.embedding_service.cleanup()
        if hasattr(app.state, "generator") and hasattr(app.state.generator, "cleanup"):
            await app.state.generator.cleanup()

        # Force cleanup of ML model resources
        import gc

        # Clear PyTorch cache if available
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        # Force garbage collection
        gc.collect()

    except Exception as e:
        logger.warning(f"Error during shutdown: {e}")


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    settings = get_settings()

    # Setup logging
    setup_logging(settings)

    # Create FastAPI app
    app = FastAPI(
        title="askme",
        description="Hybrid RAG system with BGE-M3, reranking, and evaluation",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # Configure CORS
    if settings.api.cors.allow_origins:
        # Starlette's type annotations are strict for add_middleware kwargs,
        # use cast(Any, app) to match runtime signature and avoid type errors.
        cast(Any, app).add_middleware(
            CORSMiddleware,
            allow_origins=settings.api.cors.allow_origins,
            allow_credentials=True,
            allow_methods=settings.api.cors.allow_methods,
            allow_headers=settings.api.cors.allow_headers,
        )

    # Include routers
    app.include_router(health.router, prefix="/health", tags=["health"])
    app.include_router(ingest.router, prefix="/ingest", tags=["ingestion"])
    app.include_router(query.router, prefix="/query", tags=["query"])
    app.include_router(evaluation.router, prefix="/eval", tags=["evaluation"])

    # Global exception handler
    async def global_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        logger = logging.getLogger(__name__)
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "detail": str(exc)},
        )

    app.add_exception_handler(Exception, global_exception_handler)

    return app


# Create app instance
app = create_app()


def setup_signal_handlers() -> None:
    """Setup signal handlers for graceful shutdown."""

    def signal_handler(signum: int, frame: FrameType | None) -> None:
        logger = logging.getLogger(__name__)
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        # FastAPI's lifespan will handle the cleanup
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


if __name__ == "__main__":
    import uvicorn

    # Setup signal handlers for graceful shutdown
    setup_signal_handlers()

    settings = get_settings()
    uvicorn.run(
        "askme.api.main:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=settings.api.reload,
        workers=settings.api.workers,
        access_log=settings.api.access_log,
    )
