"""
src/utils/logging_utils.py

Structured logging setup using Loguru.
Adds file rotation, coloured console output, and optional W&B integration.
"""

import sys
from pathlib import Path
from loguru import logger


def setup_logging(
    log_dir: str = "logs",
    log_level: str = "INFO",
    run_name: str = "run",
):
    """Configure Loguru for the project."""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Remove default handler
    logger.remove()

    # Console handler — coloured, human-readable
    logger.add(
        sys.stderr,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{line}</cyan> — "
            "<level>{message}</level>"
        ),
        level=log_level,
        colorize=True,
    )

    # File handler — full detail, rotated daily
    logger.add(
        log_path / f"{run_name}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} — {message}",
        level="DEBUG",
        rotation="1 day",
        retention="7 days",
        compression="gz",
    )

    return logger
