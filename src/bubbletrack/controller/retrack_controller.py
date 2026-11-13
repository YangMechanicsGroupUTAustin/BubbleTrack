"""Automatic re-tracking after the user edits results (Auto mode).

About two seconds after the last edit (any ``RESULTS_CHANGED`` that is not a
run finishing, a source loading, an undo or the re-track itself), the
sequence solver runs again in the background, so one fixed frame can repair
its neighbours.  It only runs in Auto mode with "Track across frames after
Fit All" on, when results exist and no fit is running.

A re-track never damages frames that look good: it may only change frames
that, when it starts, have no result or are listed in the Review queue
(``sequence_pipeline.retrack_frames``); manual, drawn-circle and locked
frames are anchors and stay as they are.  Everything it changed is undone
with one Ctrl+Z (shared ``ResultHistory``).

While it runs, editing is locked like a run (``run_state_changed``) except
the Manual tab: a fix there cancels the re-track (its snapshot is stale) and
restarts the countdown.  Stop cancels it.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from functools import partial

import numpy as np
from PyQt6.QtCore import QTimer

from bubbletrack.controller import edge_support
from bubbletrack.controller.base import BaseController
from bubbletrack.controller.display_mixin import display_frame, refresh_chart
from bubbletrack.controller.tracking import TrackingWorker, apply_tracking, review_items
from bubbletrack.event_bus import RESULTS_CHANGED, EventBus
from bubbletrack.model.cache import ImageCache
from bubbletrack.model.result_history import ResultHistory, record_edit, snapshot
from bubbletrack.model.sequence_pipeline import retrack_frames

logger = logging.getLogger(__name__)

AUTO_RETRACK_ENABLED = True
"""Owner decision (pending; default on): re-track automatically after edits.
Set to False to keep tracking a Fit All step only (one-line switch)."""
RETRACK_DEBOUNCE_MS = 2000  # quiet time after the last edit before re-tracking

_OK, _WARN = "#10b981", "#f59e0b"


class RetrackController(BaseController):
    """Debounced background re-tracking after result edits."""

    def __init__(self, bus: EventBus, get_state, set_state, window, get_max_radius,
                 cache: ImageCache | None = None, *, history: ResultHistory,
                 is_run_active: Callable[[], bool],
                 queued_frames: Callable[[], Iterable[int]],
                 worker_factory=TrackingWorker) -> None:
        super().__init__(bus, get_state, set_state, window)
        self._get_max_radius = get_max_radius
        self._cache = cache
        self._history = history
        self._is_run_active = is_run_active
        self._queued_frames = queued_frames
        self.worker_factory = worker_factory
        self._worker = None
        self._retired: list = []  # stopped workers still finishing (keep them alive)
        self._allowed: frozenset[int] = frozenset()
        self._muted = 0
        self._quiet = False
        self._own_emit = False
        self.timer = QTimer()
        self.timer.setSingleShot(True)
        self.timer.setInterval(RETRACK_DEBOUNCE_MS)
        self.timer.timeout.connect(self._on_timer)
        bus.subscribe(RESULTS_CHANGED, self._on_results_changed)
        bus.subscribe("run_state_changed", self._on_run_state)
        bus.subscribe("source_loaded", self._on_source_loaded)

    # -- public API ---------------------------------------------------------- #

    def is_running(self) -> bool:
        return self._worker is not None

    @contextmanager
    def muted(self):
        """Result changes announced inside this block do not trigger a re-track."""
        self._muted += 1
        try:
            yield
        finally:
            self._muted -= 1

    def cancel(self) -> None:
        """Drop a pending countdown and a running re-track (results unchanged)."""
        self.timer.stop()
        if self._worker is not None:
            self._abandon()

    def stop(self) -> None:
        """The Stop button."""
        self.timer.stop()
        if self._worker is not None:
            self._abandon()
            self.w.header.set_status("Re-track stopped; results unchanged", _WARN)

    def shutdown(self) -> None:
        """Window closing: stop everything and wait for the threads."""
        self.cancel()
        for worker in self._retired:
            worker.wait(10_000)
        self._retired.clear()

    # -- triggers ------------------------------------------------------------ #

    def _on_results_changed(self, *_args) -> None:
        if self._muted or self._quiet:
            return
        if self._worker is not None:  # an edit made its snapshot stale
            self._abandon()
        if AUTO_RETRACK_ENABLED:
            self.timer.start()

    def _on_run_state(self, running: bool) -> None:
        if self._own_emit:
            return
        if running:
            self.timer.stop()  # a run tracks on its own
        else:
            self._quiet_turn()  # its closing RESULTS_CHANGED is not an edit

    def _on_source_loaded(self, _n: int) -> None:
        self.cancel()
        self._quiet_turn()

    def _quiet_turn(self) -> None:
        """Ignore result changes until control returns to the event loop."""
        self._quiet = True
        QTimer.singleShot(0, self._end_quiet)

    def _end_quiet(self) -> None:
        self._quiet = False

    def eligible(self) -> bool:
        state = self.state
        return (AUTO_RETRACK_ENABLED and self._worker is None and not self._is_run_active()
                and state.detection_mode == "edge" and state.radius is not None
                and bool(np.any(np.asarray(state.radius) > 0))
                and self.w.left_panel.pretune_tab.edge_box.tracking_enabled())

    # -- running ------------------------------------------------------------- #

    def _on_timer(self) -> None:
        if not self.eligible():
            return
        state = self.state
        allowed = retrack_frames(state, self._queued_frames())
        if not allowed:
            return
        try:
            job = edge_support.build_job(state, self._get_max_radius())
            worker = self.worker_factory(state, job)
        except Exception as exc:
            logger.warning("Cannot start re-tracking: %s", exc)
            return
        worker.solved.connect(partial(self._on_solved, worker))
        worker.failed.connect(partial(self._on_ended, worker, "failed"))
        worker.stopped.connect(partial(self._on_ended, worker, "stopped"))
        self._worker, self._allowed = worker, allowed
        self._lock(True)
        logger.info("Re-tracking after an edit: %d frame(s) may change", len(allowed))
        worker.start()

    def _on_solved(self, worker, outcome) -> None:
        if worker is not self._worker:
            return
        state = self.state
        before = snapshot(state, self._allowed)
        apply_tracking(state, outcome, allowed_frames=self._allowed)
        edit = record_edit(state, "re-track", before)
        self._history.push(edit)
        self._release(worker)
        n = len(edit.frames) if edit is not None else 0
        refresh_chart(state, self.w)
        if state.images:
            display_frame(state, self.w, state.image_no, self._set_state, self._cache)
        text = f"Re-tracked: {n} frame(s) updated" + (" — Ctrl+Z undoes" if n else "")
        self.w.header.set_status(text, _OK)
        logger.info(text)
        with self.muted():
            self.bus.emit("tracking_review", tuple(review_items(outcome.result)))
            if n:
                self._results_changed()

    def _on_ended(self, worker, note: str, *_detail) -> None:
        if worker is not self._worker:
            return
        self._release(worker)
        self.w.header.set_status(f"Re-track {note}; results unchanged", _WARN)

    def _abandon(self) -> None:
        worker = self._worker
        worker.request_stop()
        self._release(worker)

    def _release(self, worker) -> None:
        """Forget *worker* (kept alive until its thread ends) and unlock editing."""
        self._worker = None
        self._retired = [w for w in self._retired if w.isRunning()]
        self._retired.append(worker)
        self._lock(False)

    def _lock(self, running: bool) -> None:
        self._own_emit = True
        try:
            self.bus.emit("run_state_changed", running)
        finally:
            self._own_emit = False
        lp = self.w.left_panel
        lp.automatic_tab.set_running(running)
        if running:
            lp.manual_tab.setEnabled(True)  # fixes there cancel and restart the re-track
            lp.automatic_tab.set_progress_text("Re-tracking after your edit...")
            self.w.header.set_status("Re-tracking the bubble after your edit... "
                                     "(Stop cancels)", _WARN)
