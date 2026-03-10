"""
Structured logging setup with coloured console output and a rotating file handler.

Usage::

    from src.utils.logger import get_logger

    logger = get_logger(__name__)
    logger.info("Pipeline started")
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Final

from src.utils.constants import LOGS_DIR

# ---------------------------------------------------------------------------
# Colour codes (ANSI -- works in every modern terminal)
# ---------------------------------------------------------------------------
_RESET: Final[str] = "\033[0m"
_COLOURS: Final[dict[int, str]] = {
    logging.DEBUG: "\033[36m",       # cyan
    logging.INFO: "\033[32m",        # green
    logging.WARNING: "\033[33m",     # yellow
    logging.ERROR: "\033[31m",       # red
    logging.CRITICAL: "\033[1;31m",  # bold red
}

# ---------------------------------------------------------------------------
# Format strings
# ---------------------------------------------------------------------------
_LOG_FMT: Final[str] = "%(asctime)s | %(name)-30s | %(levelname)-8s | %(message)s"
_DATE_FMT: Final[str] = "%Y-%m-%d %H:%M:%S"


class _ColouredFormatter(logging.Formatter):
    """Adds ANSI colour codes around the level name for console output."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        colour = _COLOURS.get(record.levelno, _RESET)
        record.levelname = f"{colour}{record.levelname}{_RESET}"
        return super().format(record)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_logger(name: str, level: str | None = None) -> logging.Logger:
    """Return a configured logger with console and file handlers.

    The log level is determined by (in priority order):

    1. The *level* argument (if provided).
    2. The ``LOG_LEVEL`` environment variable.
    3. Falls back to ``INFO``.

    Parameters
    ----------
    name:
        Logger name, typically ``__name__`` of the calling module.
    level:
        Optional override for the log level.

    Returns
    -------
    logging.Logger
        Fully configured logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers when called multiple times.
    if logger.handlers:
        return logger

    resolved_level = (level or os.environ.get("LOG_LEVEL", "INFO")).upper()
    log_level = getattr(logging, resolved_level, logging.INFO)
    logger.setLevel(log_level)

    # --- Console handler (coloured) ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(
        _ColouredFormatter(fmt=_LOG_FMT, datefmt=_DATE_FMT),
    )
    logger.addHandler(console_handler)

    # --- File handler ---
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOGS_DIR / "app.log"

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(
        logging.Formatter(fmt=_LOG_FMT, datefmt=_DATE_FMT),
    )
    logger.addHandler(file_handler)

    # Prevent log messages from propagating to the root logger.
    logger.propagate = False

    return logger
