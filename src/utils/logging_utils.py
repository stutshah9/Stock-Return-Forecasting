"""Logging helpers with consistent formatting across scripts."""

from __future__ import annotations

import logging
from typing import Optional


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger with a single stream handler."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def set_global_log_level(level: int, names: Optional[list[str]] = None) -> None:
    """Update the level of the root logger or a list of named loggers."""
    if names is None:
        logging.getLogger().setLevel(level)
        return
    for name in names:
        logging.getLogger(name).setLevel(level)
