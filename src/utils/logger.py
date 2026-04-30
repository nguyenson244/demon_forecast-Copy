"""
logger.py — Centralized Logging Setup
============================================================
Mọi module trong project đều import logger từ đây:

    from src.utils.logger import get_logger
    logger = get_logger(__name__)

Features:
  - Console handler  : INFO+  (colored format)
  - File handler     : DEBUG+ (rotating, max 10 MB × 5 backups)
  - Log file path    : logs/pipeline.log (auto-created)
============================================================
"""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

# ---------------------------------------------------------------------------
# Resolve log directory (project_root/logs/)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]   # src/utils/ → project/
_LOG_DIR = _PROJECT_ROOT / "logs"
_LOG_FILE = _LOG_DIR / "pipeline.log"

_LOG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Formats
# ---------------------------------------------------------------------------
_CONSOLE_FMT = "[%(levelname)s] %(name)s — %(message)s"
_FILE_FMT    = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FMT    = "%Y-%m-%d %H:%M:%S"

# ---------------------------------------------------------------------------
# Root logger flag — configure only once
# ---------------------------------------------------------------------------
_configured = False


def _configure_root() -> None:
    """Configure the root logger (called once per process)."""
    global _configured
    if _configured:
        return

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # ── Console handler ──────────────────────────────────────
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(_CONSOLE_FMT))
    root.addHandler(console)

    # ── Rotating file handler ────────────────────────────────
    file_handler = RotatingFileHandler(
        _LOG_FILE,
        maxBytes=10 * 1024 * 1024,   # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(_FILE_FMT, datefmt=_DATE_FMT))
    root.addHandler(file_handler)

    # Suppress noisy third-party loggers
    for noisy in ("prophet", "cmdstanpy", "optuna", "numexpr"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger with the centralized configuration applied.

    Usage
    -----
    >>> from src.utils.logger import get_logger
    >>> logger = get_logger(__name__)
    >>> logger.info("Pipeline started")
    """
    _configure_root()
    return logging.getLogger(name)
