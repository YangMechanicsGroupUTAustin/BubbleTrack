"""Logging configuration for BubbleTrack."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(*, level: int = logging.INFO, log_dir: Path | None = None) -> None:
    """Configure root logger with console + optional file handler."""
    root = logging.getLogger("bubbletrack")
    root.setLevel(level)

    # Idempotent: replace handlers from a previous call instead of stacking
    # them (otherwise every log line is printed once per call).
    for handler in [h for h in root.handlers if getattr(h, "_bubbletrack", False)]:
        root.removeHandler(handler)
        handler.close()

    # Console handler
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
    console._bubbletrack = True
    root.addHandler(console)

    # File handler (optional)
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / "bubbletrack.log", encoding="utf-8")
        fh.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
        fh._bubbletrack = True
        root.addHandler(fh)
