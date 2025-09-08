"""
FastAPI application for askme hybrid RAG system.
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from askme.api.routes import ingest, query, evaluation, health
from askme.core.config import get_settings, Settings
from askme.core.logging_config import setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan management."""
    logger = logging.getLogger(__name__)
    
    # Startup
    logger.info("Starting askme API server")
    settings = get_settings()
    
    # Initialize components
    try:
        # Initialize vector database connection
        logger.info(f"Initializing {settings.vector_backend} vector database")
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {settings.embedding.model}")
        
        # Initialize reranker
        if settings.rerank.local_enabled:
            logger.info(f"Loading local reranker: {settings.rerank.local_model}")
        
        logger.info("askme API server started successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize askme components: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down askme API server")


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
    async def global_exception_handler(request, exc):
        logger = logging.getLogger(__name__)
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "detail": str(exc)}
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