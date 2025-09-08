# Multi-stage build for production-ready askme API
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast Python package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml .
COPY README.md .

# Install dependencies
RUN uv venv /opt/venv && \
    . /opt/venv/bin/activate && \
    uv pip install --no-deps -e .

# Production stage
FROM python:3.11-slim as production

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r askme && useradd -r -g askme -d /app -s /bin/bash askme

# Set working directory and permissions
WORKDIR /app
RUN mkdir -p /app/{data,logs,models,configs} && \
    chown -R askme:askme /app

# Copy virtual environment from builder stage
COPY --from=builder --chown=askme:askme /opt/venv /opt/venv

# Add venv to path
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY --chown=askme:askme . .

# Create directories for runtime
RUN mkdir -p data/documents data/eval logs models/embeddings models/rerank

# Switch to non-root user
USER askme

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Expose port
EXPOSE 8080

# Default command
CMD ["uvicorn", "askme.api.main:app", "--host", "0.0.0.0", "--port", "8080"]