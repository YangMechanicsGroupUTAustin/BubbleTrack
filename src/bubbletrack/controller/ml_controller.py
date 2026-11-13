"""ML assist in the app: the Auto box's state line, teaching mode, Train and apply, Fit All.

ML assist is a helper of Auto (``model.teach``), not a method of its own:

* the Auto box shows "ML assist: Off / On · 3 examples", a switch and
  "Teach…", which opens the teaching mode (``ml_teach``, a dock): confirm two
  frames of different situations where the outline is right, then press
  Train and apply;
* Train and apply trains once and measures the frames that need review (the
  frames listed in Review and the frames without a result; never the user's
  frames nor the ones Auto measured without a flag), in the background like
  Fit All (``AutoController.run_ml_assist``); then the teaching mode closes,
  the Review dock comes forward and says what happened.  It turns ML assist
  on for the recording; while it is on, Fit All in Auto mode uses it;
* after a plain Auto run where many frames fail, a short hint in the Review
  dock (and the status line) suggests it;
* the examples and the switch live in the state, so sessions and the
  autosave keep them.

The teaching mode and Review's Fix never share the image's mouse: opening
the teaching mode ends a fix (``end_fix``), and a fix starting closes the
teaching mode first (``FixMode`` start hook).  Without scikit-learn the
button is disabled with a tooltip.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import replace

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QDockWidget

from bubbletrack.controller import edge_support, review_actions
from bubbletrack.controller import ml_teach_steps as steps
from bubbletrack.controller.display_mixin import current_source, display_frame
from bubbletrack.controller.ml_teach import TeachingMode
from bubbletrack.event_bus import RESULTS_CHANGED
from bubbletrack.model import edge_pipeline
from bubbletrack.model.ml_assist import (
    UNAVAILABLE_TIP, examples_for_run, failures, make_job, ml_available, protected_frames,
    stored_auto_map,
)
from bubbletrack.model.ml_pipeline import MLJob, train_job
from bubbletrack.model.state import update_state
from bubbletrack.ui.ml_hint import MLHint
from bubbletrack.ui.ml_teach_panel import MLTeachPanel
from bubbletrack.ui.wording import plural

logger = logging.getLogger(__name__)

_OK, _WARN, _ERR = "#10b981", "#f59e0b", "#ef4444"
GUI_THREADS = 4  # model threads when training on the GUI thread (validation)
NOT_TAUGHT = "Teach ML assist first: Teach…, then confirm 2 different frames"


def switch_text(on: bool, n_examples: int) -> str:
    """The ML assist switch's text: 'On · 2 examples', 'Off · not taught yet'."""
    head = "On" if on else "Off"
    if n_examples:
        return f"{head} · {plural(n_examples, 'example')}"
    return head if on else f"{head} · not taught yet"


def hint_text(missing) -> str:
    """The Review dock's short hint after a run where many frames failed (two lines)."""
    if missing.few_accepted:
        return f"Only {missing.accepted} of {missing.run} frames measured."
    return f"{round(100 * missing.share)}% of the bubble's frames have no result."


@contextmanager
def _wait_cursor():
    QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
    try:
        yield
    finally:
        QApplication.restoreOverrideCursor()


class MLAssistController:
    """Wires ML assist into the running app (see module doc)."""

    def __init__(self, window, app, panel: MLTeachPanel, dock: QDockWidget) -> None:
        self.w = window
        self._app = app
        self.available = ml_available()
        self.history = app.history
        self.dock = dock
        self.teach = TeachingMode(self, panel, dock)
        self.hint = MLHint()
        self.hint.open_clicked.connect(self.open_teaching)
        app.review.panel.add_banner(self.hint)
        app.review.fix.add_start_hook(self.teach.yield_to_fix)
        self._validated: tuple | None = None  # what the last validation trained on
        self._training: tuple[int, ...] | None = None  # Train and apply's frames while it runs
        box = window.left_panel.measure_panel.edge_box
        box.ml_assist_clicked.connect(self.open_teaching)
        box.ml_switch_toggled.connect(self.set_enabled)
        box.set_ml_available(self.available, UNAVAILABLE_TIP)
        panel.train_clicked.connect(self.apply)
        panel.close_clicked.connect(self.teach.leave)
        panel.stop_clicked.connect(self._stop)
        # After the pre-tune handler stored the new method: teaching is for Auto only.
        window.left_panel.measure_panel.filters_changed.connect(self._on_method_changed)
        app.auto_ctrl.ml_job_provider = self.job_for_fit_all
        bus = app.bus
        bus.subscribe("source_loaded", self._on_source_loaded)
        bus.subscribe("run_state_changed", self._on_run_state)
        bus.subscribe("run_progress", lambda current, total: panel.set_progress(current, total))
        bus.subscribe("run_finished", self._on_run_finished)
        bus.subscribe(RESULTS_CHANGED, lambda *_: self._on_results_changed())
        self.refresh_state_line()

    # -- what the teaching mode uses ------------------------------------------- #

    @property
    def state(self):
        return self._app._get_state()

    def _update(self, **kwargs) -> None:
        self._app._set_state(update_state(self.state, **kwargs))

    def status(self, text: str, colour: str = _OK) -> None:
        self.w.header.set_status(text, colour)

    def go_to(self, idx: int) -> None:
        self._app.file_ctrl.on_frame_changed(idx)

    def redisplay(self) -> None:
        state = self.state
        if state.images:
            display_frame(state, self.w, state.image_no, self._app._set_state, self._app._image_cache)

    def reference_index(self) -> int | None:
        try:
            return edge_support.reference_index(self.state)
        except Exception:  # noqa: BLE001 - only used to leave it out of the suggestions
            return None

    def set_examples(self, examples) -> None:
        """New strokes: kept in the state (sessions); autosaved without a re-track."""
        self._update(ml_examples=examples)
        self._app.session_ctrl.notify_results_changed()

    def accept_frame(self, frame: int) -> str | None:
        """Looks right: Review's Accept on *frame* (its guess, else its result; locked).

        Choosing an example is not a fix: it starts no re-track (one ran
        seconds later after every Looks right and locked the panel), Train and
        apply re-measures the frames around it anyway."""
        with self._app.retrack.muted():
            return review_actions.accept_frame(self._app, frame)

    def scope(self) -> tuple[int, ...]:
        """The frames Train and apply measures (``ml_teach_steps.train_scope``)."""
        return steps.train_scope(self.state, self._app.review.queued_frames())

    def end_fix(self) -> str:
        """End Review's fix, if one runs (nothing is stored); a note for the status."""
        fix = self._app.review.fix
        frame = fix.frame
        if frame is None:
            return ""
        fix.end()
        return f"Fix of frame {frame + 1} cancelled (nothing stored)"

    def _is_busy(self) -> bool:
        """A Fit runs (a quiet re-track does not count: Train and apply replaces it)."""
        return self._app.auto_ctrl.is_busy()

    # -- the Auto box -------------------------------------------------------------- #

    def refresh_state_line(self) -> None:
        """The Auto box's ML assist line: 'On · 2 examples', shown once relevant."""
        state = self.state
        progress = steps.progress(state) if state.radius is not None else None
        taught = progress is not None and progress.ready
        on = bool(state.ml_assist)
        n = 0 if progress is None else len(progress.examples)  # as the teaching panel counts
        fitted = state.radius is not None and bool((state.radius > 0).any())
        self.w.left_panel.measure_panel.edge_box.set_ml_state(
            switch_text(on, n), on, self.available and (taught or on), _OK if on else "",
            relevant=fitted or n > 0 or on)

    def open_teaching(self) -> None:
        if not self.available:
            self.status(UNAVAILABLE_TIP, _WARN)
            return
        if self._is_busy():
            self.status("Wait for the running fit to finish", _WARN)
            return
        if self.state.detection_mode != "edge":
            self.status("ML assist helps Auto: choose the Auto (reference frame) method first", _WARN)
            return
        self.hint.hide()
        review = getattr(self._app.review, "dock", None)
        if review is not None and review.isVisible() and not review.isFloating():
            self.w.tabifyDockWidget(review, self.dock)
        self.teach.enter()

    def set_enabled(self, on: bool) -> None:
        """The switch: Fit All measures with ML assist (on) or plain Auto (off)."""
        if on and not steps.progress(self.state).ready:
            self.refresh_state_line()
            self.status(NOT_TAUGHT, _WARN)
            return
        self._update(ml_assist=bool(on))
        self._app.session_ctrl.notify_results_changed()
        self.refresh_state_line()
        if on:
            self.status("ML assist on: Fit measures with it", _OK)
        else:
            self.status("ML assist off: Fit measures with plain Auto "
                        "(fit again to return to Auto's results)", _OK)

    # -- runs ------------------------------------------------------------------------ #

    def prepare_job(self, examples) -> MLJob:
        """The pool job for *examples*, checked by training once here (raises if it cannot)."""
        state = self.state
        reference = edge_support.reference_raw(state)
        job = make_job(state, examples, lambda i: edge_pipeline.read_raw(current_source(state, i)),
                       reference)
        key = (examples, job.gridx, job.gridy, edge_support.reference_index(state))
        if self._validated != key:
            with _wait_cursor():
                train_job(replace(job, n_jobs=GUI_THREADS))
        self._validated = key
        return job

    def apply(self) -> None:
        """Train and apply: train once, then measure the frames that need review."""
        state = self.state
        if self._is_busy():
            self.status("Wait for the running fit to finish", _WARN)
            return
        progress = steps.progress(state)
        scope = self.scope()
        ready, why = steps.train_state(progress, scope=len(scope))
        if not ready:
            self.status(f"ML assist: {why}", _WARN)
            return
        examples = examples_for_run(state)
        panel = self.teach.panel
        panel.set_train(False, "Training…")
        panel.set_next(f"Training ML assist on {plural(len(examples.frames), 'frame')}…")
        panel.repaint()  # shown before the short wait (no events: no second click)
        try:
            job = self.prepare_job(examples)
        except Exception as exc:  # noqa: BLE001 - shown to the user
            logger.warning("ML assist cannot train: %s", exc)
            self.status(f"ML assist cannot learn from these examples: {exc}", _ERR)
            self.teach.refresh()
            panel.set_next(f"ML assist could not learn from these frames ({exc}): confirm other "
                           "frames, or paint the bubble and the background by hand.", warn=True)
            return
        protected = protected_frames(state)
        before = (state.ml_assist, state.ml_examples)
        self._update(ml_assist=True, ml_examples=examples)
        self._app.retrack.cancel()  # a pending re-track would only redo these frames
        self._training = scope
        if not self._app.auto_ctrl.run_ml_assist(job, protected=protected,
                                                 stored=stored_auto_map(state, scope), frames=scope):
            self._training = None
            self._update(ml_assist=before[0], ml_examples=before[1])
            self.teach.refresh()
        self.refresh_state_line()

    def job_for_fit_all(self) -> tuple[MLJob | None, str]:
        """AutoController's provider: (job, note) for a Fit All while ML assist is on."""
        state = self.state
        if not state.ml_assist:
            return None, ""
        if not self.available:
            return None, "ML assist skipped (scikit-learn is missing)"
        if not steps.progress(state).ready:
            return None, "ML assist skipped (it needs 2 different confirmed frames)"
        try:
            return self.prepare_job(examples_for_run(state)), ""
        except Exception as exc:  # noqa: BLE001 - the run goes on with plain Auto
            logger.warning("ML assist skipped for Fit All: %s", exc)
            return None, f"ML assist skipped ({exc})"

    def _stop(self) -> None:
        self._app.auto_ctrl.on_auto_stop()
        self._app.retrack.stop()

    # -- bus events --------------------------------------------------------------------- #

    def _on_method_changed(self) -> None:
        if self.state.detection_mode != "edge":
            self.teach.leave()

    def _on_run_state(self, running: bool) -> None:
        self.teach.set_running(running)
        if running and self._training is not None:
            self.teach.show_measuring(len(self._training))

    def _on_run_finished(self, info: dict) -> None:
        state = self.state
        trained, self._training = self._training, None
        self.refresh_state_line()
        if trained is not None and info.get("ml"):
            self._after_training(info)
            return
        if info.get("stopped") or not info.get("edge") or state.radius is None:
            return
        if info.get("ml"):
            self._show_review()
            return
        if state.ml_assist or not self.available:
            return
        missing = failures(state, info.get("frames", ()))
        if missing.worth_a_hint():
            self.hint.show_text(hint_text(missing))
            header = self.w.header
            header.set_status(f"{header.status_text()} · ML assist can help", _WARN,
                              header.status_details())

    def _after_training(self, info: dict) -> None:
        """Train and apply is done: close teaching, Review forward, say what happened."""
        state, frames = self.state, list(info.get("frames", ()))
        measured = int(np.sum(np.asarray(state.radius)[frames] > 0)) if frames else 0
        text = steps.summary_text(measured, len(frames), self._app.review.check_count,
                                  stopped=bool(info.get("stopped")))
        details = self.w.header.status_details()
        self.teach.leave()
        self._show_review(first_row=True)
        self.hint.show_note(text)
        self.w.header.set_status(text, _WARN if info.get("stopped") else _OK, details)

    def _show_review(self, *, first_row: bool = False) -> None:
        review = self._app.review
        review.dock.show()
        review.dock.raise_()
        rows = review.panel.segments()
        if first_row and rows and review.panel.select_frame(rows[0].entry, keep_row=False):
            self.go_to(rows[0].entry)

    def _on_results_changed(self) -> None:
        self.teach.on_results_changed()
        self.refresh_state_line()

    def _on_source_loaded(self, _n: int) -> None:
        self.teach.reset()
        self.hint.hide()
        self._validated = None
        self._training = None
        self._update(ml_assist=False, ml_examples=None)
        self.refresh_state_line()

    def shutdown(self) -> None:
        """Window closing: leave the teaching mode."""
        self.teach.leave()


def install_ml_assist(window, app) -> MLAssistController:
    """Add ML assist's teaching dock (hidden until 'Teach…') and wire it.

    Idempotent, like the Review dock.  Needs ``app.review`` (the hint and the
    fix it must never run beside live there).
    """
    existing = getattr(app, "ml", None)
    if isinstance(existing, MLAssistController):
        return existing
    panel = MLTeachPanel()
    dock = QDockWidget("ML assist", window)
    dock.setObjectName("MLAssistDock")
    # Closed with the panel's Close button only: the dock *is* the teaching mode.
    dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable)
    dock.setWidget(panel)
    window.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
    dock.hide()
    return MLAssistController(window, app, panel, dock)
