"""Last-resort handling of uncaught exceptions.

Without a custom ``sys.excepthook``, PyQt6 aborts the whole process when an
exception escapes a slot, and every unsaved result is lost.  ``CrashHandler``
logs the error, writes a recovery session (rate-limited), and asks the UI to
tell the user, while the application keeps running.

The handler has no Qt dependency; the UI supplies ``notify``.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from bubbletrack.model.session import save_session

logger = logging.getLogger(__name__)

# Minimum seconds between two recovery files, so an error storm cannot flood the disk.
RECOVERY_MIN_INTERVAL_S = 30.0

Notify = Callable[[str, str | None], None]


def _summarize(exc_type, exc) -> str:
    """One-line description that never raises, even if ``str(exc)`` does."""
    try:
        detail = str(exc)
    except Exception:  # noqa: BLE001
        detail = "<message unavailable>"
    return f"{exc_type.__name__}: {detail}" if detail else exc_type.__name__


def _has_results(state) -> bool:
    radius = getattr(state, "radius", None)
    return radius is not None and bool(np.any(np.asarray(radius) > 0))


class CrashHandler:
    """Callable with the ``sys.excepthook`` signature.

    Parameters
    ----------
    state_provider : callable
        Returns the current ``AppState`` (or None).
    notify : callable(summary, recovery_path)
        Shows the error to the user.  Called on the thread that raised, so a
        Qt implementation must marshal to the GUI thread itself.
    recovery_dir : Path
        Where recovery ``.brt`` sessions are written.
    clock : callable
        Time source (injectable for tests).
    fallback : callable
        Hook used for KeyboardInterrupt/SystemExit (defaults to the original).
    """

    def __init__(
        self,
        *,
        state_provider: Callable[[], object],
        notify: Notify,
        recovery_dir: Path,
        clock: Callable[[], float] = time.time,
        fallback: Callable[..., None] = sys.__excepthook__,
        min_interval: float = RECOVERY_MIN_INTERVAL_S,
    ) -> None:
        self._state_provider = state_provider
        self._notify = notify
        self._recovery_dir = Path(recovery_dir)
        self._clock = clock
        self._fallback = fallback
        self._min_interval = min_interval
        self._last_save_time: float | None = None
        self._last_recovery_path: str | None = None
        self._lock = threading.RLock()
        self._handling = False
        self._saved_hooks: tuple | None = None

    # -- hook entry points ------------------------------------------------

    def __call__(self, exc_type, exc, tb) -> None:
        if issubclass(exc_type, (KeyboardInterrupt, SystemExit)):
            self._fallback(exc_type, exc, tb)
            return
        with self._lock:
            if self._handling:
                logger.critical("Error while handling a previous error",
                                exc_info=(exc_type, exc, tb))
                return
            self._handling = True
            try:
                self._handle(exc_type, exc, tb)
            finally:
                self._handling = False

    def handle_thread_exception(self, args: threading.ExceptHookArgs) -> None:
        if args.exc_type is None:
            return
        self(args.exc_type, args.exc_value, args.exc_traceback)

    def install(self) -> None:
        self._saved_hooks = (sys.excepthook, threading.excepthook)
        sys.excepthook = self
        threading.excepthook = self.handle_thread_exception

    def uninstall(self) -> None:
        if self._saved_hooks is not None:
            sys.excepthook, threading.excepthook = self._saved_hooks
            self._saved_hooks = None

    # -- internals ----------------------------------------------------------

    def _handle(self, exc_type, exc, tb) -> None:
        try:
            logger.critical("Uncaught exception", exc_info=(exc_type, exc, tb))
        except Exception:  # noqa: BLE001 - e.g. an exception whose __str__ fails
            logger.critical("Uncaught %s (could not be formatted)", exc_type.__name__)
        recovery_path = self._save_recovery()
        try:
            self._notify(_summarize(exc_type, exc), recovery_path)
        except Exception:  # noqa: BLE001 - must never escape an excepthook
            logger.exception("Could not show the error notification")

    def _save_recovery(self) -> str | None:
        try:
            state = self._state_provider()
        except Exception:  # noqa: BLE001
            logger.exception("Could not read application state for recovery")
            return None
        if not _has_results(state):
            return None

        now = self._clock()
        recent = (
            self._last_save_time is not None
            and now - self._last_save_time < self._min_interval
        )
        if recent:
            return self._last_recovery_path

        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(now))
        path = self._recovery_dir / f"recovery_{stamp}_{int(now * 1000) % 1000:03d}.brt"
        try:
            self._recovery_dir.mkdir(parents=True, exist_ok=True)
            save_session(str(path), state)
        except Exception:  # noqa: BLE001
            logger.exception("Could not write recovery session to %s", path)
            return None
        self._last_save_time = now
        self._last_recovery_path = str(path)
        logger.warning("Recovery session written to %s", path)
        return self._last_recovery_path
