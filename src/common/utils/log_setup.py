"""Logging setup helpers for pipeline CLI scripts."""

from __future__ import annotations

import logging
from pathlib import Path

_FMT     = "%(asctime)s  %(levelname)-8s  %(message)s"
_DATEFMT = "%H:%M:%S"


class _FlushFileHandler(logging.FileHandler):
    """FileHandler that flushes after every record (real-time visibility in log file)."""

    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        self.flush()


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger with a console StreamHandler.

    No-op if root logger already has handlers (safe to call multiple times).
    """
    root = logging.getLogger()
    if root.handlers:
        return
    root.setLevel(level)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter(_FMT, datefmt=_DATEFMT))
    root.addHandler(ch)


def add_file_handler(log_file: Path) -> logging.Handler:
    """Add a flushing FileHandler to the root logger; return it for later removal."""
    fh = _FlushFileHandler(log_file, encoding="utf-8")
    fh.setFormatter(logging.Formatter(_FMT, datefmt=_DATEFMT))
    logging.getLogger().addHandler(fh)
    return fh
