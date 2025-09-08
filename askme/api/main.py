"""
FastAPI application for askme hybrid RAG system.
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from askme.api.routes import evaluation, health, ingest, query
from askme.core.config import get_settings
from askme.core.embeddings import BGEEmbeddingService
from askme.core.logging_config import setup_logging
from askme.ingest.ingest_service import IngestionService
from askme.rerank.rerank_service import RerankingService
from askme.retriever.milvus_retriever import MilvusRetriever


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

        cohere_api_key = os.getenv("COHERE_API_KEY") if settings.enable_cohere else None
        reranking_service = RerankingService(settings.rerank, cohere_api_key)

        ingestion_service = IngestionService(retriever, embedding_service, settings)

        # Attach to app state
        app.state.embedding_service = embedding_service
        app.state.retriever = retriever
        app.state.reranking_service = reranking_service
        app.state.ingestion_service = ingestion_service

        logger.info("askme API server components initialized")

    except Exception as e:
        logger.error(f"Failed to initialize askme components: {e}")
        raise

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
        app.add_middleware(
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
    @app.exception_handler(Exception)
    async def global_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        logger = logging.getLogger(__name__)
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "detail": str(exc)},
        )

    return app


# Create app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "askme.api.main:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=settings.api.reload,
        workers=settings.api.workers,
        access_log=settings.api.access_log,
    )
