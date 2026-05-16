"""
GeoEngine — Structured Logging Configuration
Налаштування structlog для всього проекту.

Підтримує два режими:
  development: кольоровий консольний вивід (ConsoleRenderer)
  production:  JSON рядки (JSONRenderer) → для ELK/Grafana Loki
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog


def configure_logging(
    level:       str = "INFO",
    json_output: bool = False,
    service:     str = "geoengine",
) -> None:
    """
    Налаштувати structlog для всього застосунку.

    Викликати один раз при старті (main.py або __main__).

    Args:
        level:       рівень логування (DEBUG/INFO/WARNING/ERROR)
        json_output: True = JSON (prod), False = консоль (dev)
        service:     назва сервісу для поля 'service' у JSON
    """
    # Стандартний logging → structlog bridge
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper(), logging.INFO),
    )

    # Спільні processors для обох режимів
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        _add_service_field(service),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if json_output:
        # Production: JSON рядки
        processors = shared_processors + [
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(),
        ]
    else:
        # Development: кольоровий вивід
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(
                colors=True,
                exception_formatter=structlog.dev.plain_traceback,
            )
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def _add_service_field(service: str):
    """Processor що додає поле 'service' до кожного запису."""
    def processor(logger: Any, method: str, event_dict: dict) -> dict:
        event_dict["service"] = service
        return event_dict
    return processor


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Отримати структурований логер для модуля.

    Usage:
        log = get_logger(__name__)
        log.info("terrain.loaded", tiles=5, elapsed_ms=120)
        log.error("dem.fetch.failed", url=url, error=str(e))
    """
    return structlog.get_logger(name)


# ---- Контекстні менеджери ----

class log_context:
    """
    Контекстний менеджер для тимчасового додавання полів до логів.

    Usage:
        with log_context(request_id="abc123", user="johndoe"):
            log.info("processing")   # автоматично матиме request_id + user
    """

    def __init__(self, **kwargs: Any) -> None:
        self._kwargs = kwargs

    def __enter__(self) -> "log_context":
        structlog.contextvars.bind_contextvars(**self._kwargs)
        return self

    def __exit__(self, *args: Any) -> None:
        structlog.contextvars.unbind_contextvars(*self._kwargs.keys())
