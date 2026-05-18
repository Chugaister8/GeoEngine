"""
GeoEngine — Structured Logging
Налаштування structlog для розробки та продакшену.
"""

from __future__ import annotations

import logging
import sys
from contextlib import contextmanager
from typing import Any, Iterator

import structlog


def configure_logging(
    level:      str  = "INFO",
    json_logs:  bool = False,
    service:    str  = "geoengine",
) -> None:
    """
    Налаштувати structlog.

    Args:
        level:     рівень логування (DEBUG/INFO/WARNING/ERROR)
        json_logs: True = JSON для продакшену, False = кольоровий для dev
        service:   назва сервісу для JSON логів
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Shared processors
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if json_logs:
        # Production: JSON output (for Loki, ELK, CloudWatch)
        shared_processors += [
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(),
        ]
    else:
        # Development: colored console output
        shared_processors += [
            structlog.dev.ConsoleRenderer(colors=True),
        ]

    structlog.configure(
        processors=shared_processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(sys.stdout),
        cache_logger_on_first_use=True,
    )

    # Також налаштовуємо stdlib logging для uvicorn/fastapi
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )


def get_logger(name: str = "") -> structlog.BoundLogger:
    """Отримати structlog logger."""
    return structlog.get_logger(name)


@contextmanager
def log_context(**kwargs: Any) -> Iterator[None]:
    """
    Context manager для додавання контексту до всіх логів у блоці.

    Usage:
        with log_context(request_id="abc123", user_id=42):
            log.info("Processing request")
            # → {"event":"Processing request","request_id":"abc123","user_id":42}
    """
    structlog.contextvars.bind_contextvars(**kwargs)
    try:
        yield
    finally:
        structlog.contextvars.unbind_contextvars(*kwargs.keys())
