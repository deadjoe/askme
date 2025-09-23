"""
Health check endpoints.
"""

from datetime import datetime, timezone
from typing import Any, Dict

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from askme.core.config import Settings, get_settings


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str
    timestamp: datetime
    version: str
    components: Dict[str, Any]


router = APIRouter()


@router.get("/", response_model=HealthResponse)
async def health_check(settings: Settings = Depends(get_settings)) -> HealthResponse:
    """Basic health check endpoint."""

    # Include vector backend and collection/class name for clients
    if settings.vector_backend.lower() == "weaviate":
        collection_name = settings.database.weaviate.class_name
    else:
        collection_name = settings.database.milvus.collection_name

    components = {
        "api": "healthy",
        "vector_backend": settings.vector_backend,
        "embedding_model": settings.embedding.model,
        "collection_name": collection_name,
        "reranker": {
            "local": settings.rerank.local_enabled,
            "cohere": settings.rerank.cohere_enabled,
        },
    }

    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(timezone.utc),
        version="0.1.0",
        components=components,
    )


@router.get("/ready")
async def readiness_check(settings: Settings = Depends(get_settings)) -> Dict[str, Any]:
    """Readiness check for container orchestration."""

    # TODO: Add actual component readiness checks
    checks = {
        "vector_db": "ready",  # Check actual connection
        "embedding_model": "ready",  # Check model loaded
        "reranker": "ready",  # Check reranker loaded
    }

    all_ready = all(status == "ready" for status in checks.values())

    return {
        "ready": all_ready,
        "checks": checks,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/live")
async def liveness_check() -> Dict[str, Any]:
    """Liveness check for container orchestration."""
    return {
        "alive": True,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
