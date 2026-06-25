"""Structured logging (structlog) with JSON output for experiment auditability."""

from __future__ import annotations

import logging
import sys

import structlog


def configure_logging(level: str = "INFO", json_output: bool = False) -> None:
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=getattr(logging, level))
    processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
    ]
    processors.append(
        structlog.processors.JSONRenderer()
        if json_output
        else structlog.dev.ConsoleRenderer()
    )
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, level)),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
