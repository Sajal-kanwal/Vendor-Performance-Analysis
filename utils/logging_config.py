"""
Centralized Logging Configuration
==================================
Provides a consistent logging setup across all project modules.
Supports rotating file handlers and optional colorized console output.

Usage:
    from utils.logging_config import setup_logger
    logger = setup_logger(__name__)
    logger.info("Pipeline started")
"""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from datetime import datetime
from typing import Optional


_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-25s | %(message)s"
_DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logger(
    name: str,
    log_dir: Optional[str] = None,
    log_file: Optional[str] = None,
    level: str = "INFO",
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5,
    console: bool = True,
) -> logging.Logger:
    """
    Create and configure a logger with rotating file + console handlers.

    Args:
        name:         Logger name (typically ``__name__``).
        log_dir:      Directory for log files. Defaults to ``./logs``.
        log_file:     Explicit log filename. Auto-generated if omitted.
        level:        Log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        max_bytes:    Max size per log file before rotation.
        backup_count: Number of rotated backups to keep.
        console:      Whether to also log to stdout.

    Returns:
        Configured ``logging.Logger`` instance.
    """
    logger = logging.getLogger(name)

    # Avoid duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    formatter = logging.Formatter(_DEFAULT_FORMAT, datefmt=_DEFAULT_DATE_FORMAT)

    # ── File handler (rotating) ──────────────────────────────
    if log_dir is None:
        log_dir = str(Path(__file__).resolve().parent.parent / "logs")
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d")
        log_file = f"{name.replace('.', '_')}_{timestamp}.log"

    file_handler = RotatingFileHandler(
        log_path / log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # ── Console handler ──────────────────────────────────────
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger
