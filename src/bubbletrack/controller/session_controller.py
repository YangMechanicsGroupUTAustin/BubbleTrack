"""SessionController — save and fully restore analysis sessions (.brt).

Restoring is done in three steps so nothing the user saved is lost:
1. load the image folder (this resets the frame-dependent views),
2. re-apply the saved parameters (ROI, thresholds, frame position, ...),
3. put the saved results back if the frame count still matches.

The full session is also autosaved to ``<CONFIG_DIR>/recovery/autosave.brt``:
after results change (debounced, at most once per minute) and on close.
At the next launch an autosave newer than the last clean exit is moved
aside to ``previous_session.brt`` and offered for restoring (see app.py).
"""

from __future__ import annotations

import dataclasses
import logging
import os
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QFileDialog

from bubbletrack.controller.base import BaseController
from bubbletrack.controller.unsaved_review import (
    CANCEL, SAVE, ask_save_review, describe, recording_name, review_counts, review_signature,
)
from bubbletrack.event_bus import RESULTS_CHANGED, EventBus
from bubbletrack.model import config as config_module
from bubbletrack.model.provenance import FitSource
from bubbletrack.model.session import load_session, save_session
from bubbletrack.model.state import AppState, update_state

logger = logging.getLogger(__name__)

RESULT_KEYS = ("radius", "circle_fit_par", "circle_xy", "region_count", "fit_source",
               "fit_confidence", "fit_metrics", "fit_flags", "fit_locked",
               "fit_suggestion")
_NOT_PARAMS = {"version", "folder_path", "video_path", "images", *RESULT_KEYS}
_STATE_FIELDS = {f.name for f in dataclasses.fields(AppState)}

AUTOSAVE_NAME = "autosave.brt"
PREVIOUS_SESSION_NAME = "previous_session.brt"
AUTOSAVE_MIN_INTERVAL_S = 60.0  # at most one autosave per minute ...
AUTOSAVE_DEBOUNCE_S = 2.0       # ... starting this long after the first change
RECOVERY_MTIME_SLACK_S = 2.0    # file-time resolution (FAT: 2 s) vs the clean-exit stamp
RESULTS_CHANGED_EVENT = RESULTS_CHANGED  # bus event: any controller edited results
# Bus event (no arguments): a session file restored the recording and its parameters.
SESSION_RESTORED = "session_restored"
SESSION_FILTER = "BubbleTrack Session (*.brt);;All Files (*)"

_OK, _WARN, _ERR, _DIM = "#10b981", "#f59e0b", "#ef4444", "#a1a1aa"


def recovery_dir() -> Path:
    """``<CONFIG_DIR>/recovery``, resolved at call time (tests patch CONFIG_DIR)."""
    return Path(config_module.CONFIG_DIR) / "recovery"


def autosave_path() -> Path:
    return recovery_dir() / AUTOSAVE_NAME


def has_results(state: AppState) -> bool:
    radius = state.radius
    return radius is not None and bool(np.any(np.asarray(radius) > 0))


def claim_pending_recovery() -> Path | None:
    """Return the session to offer for restoring after an unclean exit, or None.

    An autosave newer than the last recorded clean exit (or with no clean
    exit on record) means the app did not close normally.  It is moved to
    ``previous_session.brt`` first, so this session's autosaves cannot
    overwrite it whatever the user decides; if it cannot be moved, the
    autosave itself is returned.
    """
    path = autosave_path()
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None  # no autosave
    clean_exit = config_module.get_meta("last_clean_exit")
    if clean_exit is not None and mtime <= clean_exit + RECOVERY_MTIME_SLACK_S:
        return None
    claimed = path.with_name(PREVIOUS_SESSION_NAME)
    try:
        os.replace(path, claimed)
    except OSError as exc:
        logger.warning("Could not move %s aside: %s", path, exc)
        return path
    return claimed


def split_session(data: dict) -> tuple[str, dict, dict]:
    """Split a loaded session dict into (folder, parameters, results)."""
    folder = data.get("folder_path") or ""
    results = {k: data[k] for k in RESULT_KEYS if data.get(k) is not None}
    params = {k: v for k, v in data.items()
              if k in _STATE_FIELDS and k not in _NOT_PARAMS}
    return folder, params, results


def sanitize_params(params: dict, n_frames: int,
                    full_roi: tuple[tuple[int, int], tuple[int, int]]) -> tuple[dict, list[str]]:
    """Drop parameters that no longer fit the loaded images; return warnings."""
    params = dict(params)
    warnings: list[str] = []
    (_, h), (_, w) = full_roi
    gx, gy = params.get("gridx"), params.get("gridy")
    if gx is not None or gy is not None:
        ok = (gx is not None and gy is not None
              and 1 <= gx[0] <= gx[1] <= h and 1 <= gy[0] <= gy[1] <= w)
        if not ok:
            params.pop("gridx", None)
            params.pop("gridy", None)
            warnings.append("saved ROI does not fit the images (ROI reset)")
    if "image_no" in params:
        params["image_no"] = int(min(max(params["image_no"], 0), max(n_frames - 1, 0)))
    return params, warnings


def restore_results(state: AppState, results: dict) -> str | None:
    """Copy saved result arrays into *state*; return a warning if impossible."""
    radius = results.get("radius")
    if radius is None or state.radius is None:
        return None
    n = len(state.radius)
    if len(radius) != n:
        return (f"frame count changed ({len(radius)} saved, {n} now); "
                "results were not restored")
    state.radius[:] = radius
    centers = results.get("circle_fit_par")
    if centers is not None and np.shape(centers) == (n, 2):
        state.circle_fit_par[:] = centers
    counts = results.get("region_count")
    if counts is not None and len(counts) == n and state.region_count is not None:
        state.region_count[:] = counts
    edges = results.get("circle_xy")
    if edges is not None and len(edges) == n:
        for i, xy in enumerate(edges):
            state.circle_xy[i] = xy
    metrics = results.get("fit_metrics")
    if metrics is not None and state.fit_metrics is not None:
        for key, values in metrics.items():
            if key in state.fit_metrics and len(values) == n:
                state.fit_metrics[key][:] = values
    flags = results.get("fit_flags")
    if flags is not None and len(flags) == n and state.fit_flags is not None:
        state.fit_flags[:] = list(flags)
    locked = results.get("fit_locked")
    if locked is not None and len(locked) == n and state.fit_locked is not None:
        state.fit_locked[:] = locked
    guesses = results.get("fit_suggestion")  # Auto's rejected guesses (display only)
    if guesses is not None and len(guesses) == n and state.fit_suggestion is not None:
        state.fit_suggestion[:] = list(guesses)
    confidence = results.get("fit_confidence")
    if confidence is not None and len(confidence) == n and state.fit_confidence is not None:
        state.fit_confidence[:] = confidence
    sources = results.get("fit_source")
    if state.fit_source is not None:
        if sources is not None and len(sources) == n:
            state.fit_source[:] = sources
        else:  # sessions saved before provenance existed
            state.fit_source[:] = np.where(radius > 0, FitSource.AUTO, FitSource.NONE)
    return None


class SessionController(BaseController):
    """Save/load sessions and keep the autosaved recovery session up to date."""

    def __init__(
        self,
        bus: EventBus,
        get_state,
        set_state,
        window,
        file_ctrl,
        *,
        is_busy: Callable[[], bool],
        locate_folder: Callable[[str], str],
        after_restore: Callable[[], None],
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__(bus, get_state, set_state, window)
        self._file_ctrl = file_ctrl
        self._is_busy = is_busy
        self._locate_folder = locate_folder
        self._after_restore = after_restore

        # The review last saved in (or restored from) a session file; another
        # recording replaces the current one only after asking (confirm_leave).
        self._saved_review: tuple = ()
        self.choose_save_path: Callable[[], str] | None = None  # None: a file dialog
        if file_ctrl is not None:  # None in tests of the autosave alone
            file_ctrl.confirm_leave = self.confirm_leave
        self.bus.subscribe("source_loaded", self._on_source_loaded)

        self._clock = clock
        self._autosave_dirty = False
        self._autosave_target: Path | None = None
        self._last_autosave: float | None = None
        self._autosave_timer = QTimer()
        self._autosave_timer.setSingleShot(True)
        self._autosave_timer.timeout.connect(self.autosave_if_due)
        self.bus.subscribe("run_state_changed", self._on_run_state_changed)
        self.bus.subscribe(RESULTS_CHANGED_EVENT, lambda *_: self.notify_results_changed())

    def save(self, path: str) -> bool:
        try:
            save_session(path, self.state)
        except Exception as exc:
            logger.error("Session save failed: %s", exc)
            self.w.header.set_status(f"Save failed: {exc}", _WARN)
            return False
        self._saved_review = review_signature(self.state)
        self.w.header.set_status(f"Session saved: {os.path.basename(path)}", _OK)
        return True

    def save_as(self) -> bool:
        """Ask where to save the session, then save it; False if cancelled or failed."""
        path = self.choose_save_path() if self.choose_save_path else self._ask_save_path()
        return bool(path) and self.save(path)

    def _ask_save_path(self) -> str:
        default_dir = self.state.folder_path or os.path.expanduser("~")
        path, _ = QFileDialog.getSaveFileName(self.w, "Save Session", default_dir, SESSION_FILTER)
        return path

    # -- leaving a recording -------------------------------------------------- #

    def has_unsaved_review(self) -> bool:
        """The user's decisions differ from the last saved or loaded session."""
        current = review_signature(self.state)
        return bool(current) and current != self._saved_review

    def confirm_leave(self) -> bool:
        """Before another recording replaces this one: True to go on.

        With unsaved decisions: "Save the review of <name> first?"
        [Save session…] [Discard] [Cancel].
        """
        if not self.has_unsaved_review():
            return True
        detail = describe(review_counts(self.state))
        answer = ask_save_review(self.w, recording_name(self.state), detail)
        if answer == SAVE:
            return self.save_as()
        return answer != CANCEL

    def _on_source_loaded(self, _n: int) -> None:
        self._saved_review = ()  # a new recording: nothing decided yet

    def load(self, path: str) -> bool:
        if self._is_busy():
            self.w.header.set_status("Stop the running fit before loading a session", _WARN)
            return False
        if not self.confirm_leave():
            self.w.header.set_status("Session not loaded", _DIM)
            return False
        try:
            data = load_session(path)
        except (ValueError, OSError, UnicodeDecodeError) as exc:
            logger.error("Session load failed: %s", exc)
            self.w.header.set_status(f"Load failed: {exc}", _ERR)
            return False

        folder, params, results = split_session(data)
        video = data.get("video_path") or ""
        if video:
            return self._restore(path, params, results, video=video)
        if folder and not os.path.isdir(folder):
            located = self._locate_folder(folder)
            if not located:
                self.w.header.set_status(f"Image folder not found: {folder}", _ERR)
                return False
            folder = located
        return self._restore(path, params, results, folder=folder)

    def _restore(self, path: str, params: dict, results: dict, *,
                 folder: str = "", video: str = "") -> bool:
        if video:
            if not os.path.isfile(video):
                self.w.header.set_status(f"Video not found: {video}", _ERR)
                return False
            self._file_ctrl.on_video_selected(video, ask=False)
            if self.state.video_path != video:
                return False  # opening failed; FileController reported why
        elif folder:
            self._file_ctrl.on_folder_selected(folder, ask=False)
            if self.state.folder_path != folder:
                return False  # loading failed; FileController reported why

        full_roi = (self.state.gridx, self.state.gridy)
        params, warnings = sanitize_params(params, len(self.state.images), full_roi)
        try:
            self.state = update_state(self.state, **params)
        except TypeError as exc:
            logger.error("Session restore failed: %s", exc)
            self.w.header.set_status(f"Restore failed: {exc}", _ERR)
            return False

        warning = restore_results(self.state, results)
        if warning:
            warnings.append(warning)
        self._saved_review = review_signature(self.state)
        self._after_restore()
        self.bus.emit(SESSION_RESTORED)

        name = os.path.basename(path)
        if warnings:
            self.w.header.set_status(f"Session loaded: {name} — " + "; ".join(warnings), _WARN)
        else:
            self.w.header.set_status(f"Session loaded: {name}", _OK)
        return True

    # -- autosave ----------------------------------------------------------

    def notify_results_changed(self) -> None:
        """Schedule an autosave: results changed (fit, edit, refit, review...).

        Cheap to call often: saves are debounced and happen at most once per
        ``AUTOSAVE_MIN_INTERVAL_S``, never while a run is writing results.
        Sub-controllers can emit ``RESULTS_CHANGED_EVENT`` on the bus instead.
        """
        self._autosave_dirty = True
        self._autosave_target = autosave_path()
        if not self._autosave_timer.isActive():
            self._arm(max(AUTOSAVE_DEBOUNCE_S, self._wait_for_interval()))

    def autosave_if_due(self) -> str | None:
        """Timer slot: write the autosave if one is pending and allowed now.

        Returns the written path, or None.  While a run is active nothing is
        written; the end of the run notifies again.
        """
        self._autosave_timer.stop()  # when called directly, not by the timer
        if not self._autosave_dirty or self._is_busy():
            return None
        wait = self._wait_for_interval()
        if wait > 0:
            self._arm(wait)
            return None
        if not has_results(self.state):
            self._autosave_dirty = False
            return None
        self._last_autosave = self._clock()
        path = self._write_recovery(self._autosave_target or autosave_path())
        if path is not None:
            self._autosave_dirty = False
        return path

    def stop_autosave(self) -> None:
        """Cancel a pending autosave (the window is closing)."""
        self._autosave_timer.stop()

    def save_last_session(self) -> str | None:
        """Write the autosave now (on close) if there are results.

        Same file as the periodic autosave, so there is one recovery
        session; a pending debounced save is cancelled.
        """
        self.stop_autosave()
        self._autosave_dirty = False
        if not has_results(self.state):
            return None
        return self._write_recovery(autosave_path())

    def _on_run_state_changed(self, running: bool) -> None:
        if not running:
            self.notify_results_changed()

    def _wait_for_interval(self) -> float:
        if self._last_autosave is None:
            return 0.0
        return self._last_autosave + AUTOSAVE_MIN_INTERVAL_S - self._clock()

    def _arm(self, seconds: float) -> None:
        self._autosave_timer.start(max(0, int(round(seconds * 1000))))

    def _write_recovery(self, path: Path) -> str | None:
        """Save the session to *path* atomically; None (logged) on failure.

        Written to a temporary file and renamed, so a crash mid-write never
        destroys the previous good autosave.
        """
        tmp = path.with_name(path.name + ".tmp")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            save_session(str(tmp), self.state)
            os.replace(tmp, path)
        except Exception as exc:
            logger.warning("Could not write %s: %s", path, exc)
            return None
        logger.info("Session autosaved to %s", path)
        return str(path)
