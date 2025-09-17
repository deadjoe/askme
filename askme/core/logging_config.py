"""
Logging configuration setup.
"""

import logging
import sys
import warnings
from pathlib import Path
from types import FrameType
from typing import Optional

from loguru import logger

from askme.core.config import Settings


class InterceptHandler(logging.Handler):
    """Intercept standard logging messages toward Loguru sinks."""

    def emit(self, record: logging.LogRecord) -> None:
        # Get corresponding Loguru level if it exists
        level: str | int
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame: Optional[FrameType] = logging.currentframe()
        depth = 2
        while frame is not None and (
            depth == 0 or frame.f_code.co_filename == logging.__file__
        ):
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def setup_logging(settings: Settings) -> None:
    """Setup logging configuration."""

    # Filter SWIG deprecation warnings from BGE models
    warnings.filterwarnings("ignore", message=".*SwigPy.*has no __module__ attribute.*")

    # Remove default Loguru handler
    logger.remove()

    # Console handler
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )

    if settings.security.privacy.anonymize_logs:
        log_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan> | "
            "<level>{message}</level>"
        )

    logger.add(
        sys.stderr,
        format=log_format,
        level=settings.log_level,
        colorize=True,
    )

    # File handler
    if settings.logging.file.enabled:
        log_file = Path(settings.logging.file.path)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        logger.add(
            log_file,
            format=settings.logging.format,
            level=settings.log_level,
            rotation=settings.logging.file.rotation,
            retention=settings.logging.file.retention,
            compression="gz",
            serialize=settings.logging.structured.enabled,
        )

    # Intercept standard library logging
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

    # Set specific loggers
    logging.getLogger("uvicorn").handlers = [InterceptHandler()]
    logging.getLogger("uvicorn.access").handlers = [InterceptHandler()]
    logging.getLogger("fastapi").handlers = [InterceptHandler()]

    # Suppress noisy loggers
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)

    logger.info("Logging configured successfully")
