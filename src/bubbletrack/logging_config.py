"""Logging configuration for BubbleTrack."""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from pathlib import Path

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_MAX_BYTES = 5_000_000  # the log rolls over to bubbletrack.log.1, .2 (never grows unbounded)
LOG_BACKUPS = 2


class _SharedRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """Rotates like its base, but moves the live file aside before shifting the backups.

    When another running instance holds the file, or a log viewer holds a backup
    (Windows refuses the rename), no record or backup is lost: records keep being
    appended, a half-done rollover is finished first next time, and the rollover is
    tried again once the file has grown by another tenth of ``maxBytes``.
    """

    _retry_at = 0  # file position before which a refused rollover is not retried

    def doRollover(self) -> None:  # noqa: N802 (logging API)
        if self.stream is not None and self.stream.tell() < self._retry_at:
            return
        if self.stream is not None:
            self.stream.close()
            self.stream = None
        aside = f"{self.baseFilename}.rolling"
        try:
            if not os.path.exists(aside):  # else: finish the previous rollover first
                os.replace(self.baseFilename, aside)
            for i in range(self.backupCount - 1, 0, -1):
                older = f"{self.baseFilename}.{i}"
                if os.path.exists(older):
                    os.replace(older, f"{self.baseFilename}.{i + 1}")
            os.replace(aside, f"{self.baseFilename}.1")
        except OSError:
            self.stream = self._open()
            self._retry_at = self.stream.tell() + self.maxBytes // 10
            return
        self._retry_at = 0
        self.stream = self._open()


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
        fh = _SharedRotatingFileHandler(
            log_dir / "bubbletrack.log", maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUPS,
            encoding="utf-8",
        )
        fh.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
        fh._bubbletrack = True
        root.addHandler(fh)
