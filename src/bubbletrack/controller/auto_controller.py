"""AutoController — Fit: measures every frame (or a chart range) in the background."""

from __future__ import annotations

import logging
import time
from functools import partial
from pathlib import Path

import numpy as np
from PyQt6.QtWidgets import QMessageBox

from bubbletrack.controller import edge_support
from bubbletrack.controller.base import BaseController
from bubbletrack.controller.display_mixin import (
    current_source, display_frame, refresh_chart,
)
from bubbletrack.controller.tracking import TrackingWorker, apply_tracking, review_items
from bubbletrack.controller.worker import BatchWorker
from bubbletrack.event_bus import EventBus
from bubbletrack.model.cache import ImageCache
from bubbletrack.model.constants import AUTO_DISPLAY_THROTTLE_MS, QUALITY_WARN_THRESHOLD
from bubbletrack.model.conventions import frame_to_display
from bubbletrack.model.export import export_r_data
from bubbletrack.model.frame_source import normalize_source
from bubbletrack.model.provenance import FitSource, is_user_confirmed
from bubbletrack.ui.status_bar import MEASURE_MODE

logger = logging.getLogger(__name__)

AUTOSAVE_INTERVAL = 50  # frames between incremental auto-saves
AUTOSAVE_NAME = ".bubbletrack_autosave.mat"


class AutoController(BaseController):
    """Manages automatic batch fitting with a background worker thread.

    While a run is active the controller emits ``run_state_changed`` on the
    event bus so the rest of the UI can lock anything that would mutate the
    result arrays the worker is writing into.
    """

    def __init__(self, bus: EventBus, get_state, set_state, window,
                 get_max_radius, cache: ImageCache | None = None) -> None:
        super().__init__(bus, get_state, set_state, window)
        self._worker: BatchWorker | None = None
        self._tracker: TrackingWorker | None = None
        self._running = False
        self._auto_last_display: float = 0.0
        self._auto_start_time: float = 0.0
        self._get_max_radius = get_max_radius
        self._cache = cache
        # Per-run bookkeeping
        self._protected: frozenset[int] = frozenset()
        self._stop_requested = False
        self._frame_errors = 0
        self._frames_done = 0
        self._autosave_failed = False
        self._edge_job = None  # the run's EdgeJob, reused by tracking
        self._run_notes: list[str] = []  # e.g. what tracking changed
        self._ml_run = False  # the run measures with ML assist
        self._run_frames: tuple[int, ...] = ()  # frames of the current / last run
        # ML assist for Fit All: () -> (MLJob | None, note); set by MLAssistController.
        self.ml_job_provider = None

    # -- run state -------------------------------------------------------- #

    def is_busy(self) -> bool:
        """True while a fit is running."""
        return self._running

    def stop_and_wait(self, timeout_ms: int = 10_000) -> None:
        """Stop any running worker and wait for its thread to exit."""
        for worker in (self._worker, self._tracker):
            if worker is not None and worker.isRunning():
                worker.request_stop()
                worker.wait(timeout_ms)

    def _set_run_state(self, running: bool) -> None:
        self.bus.emit("run_state_changed", running)

    # -- public handlers -------------------------------------------------- #

    def on_auto_fit(self) -> None:
        """The Fit button: measure every frame."""
        if not self.state.images:
            self.w.header.set_status("Load images or a video first", "#f59e0b")
            return
        if self.is_busy():
            self.w.header.set_status("A fit is already running", "#f59e0b")
            return
        self.fit_range(0, len(self.state.images) - 1)

    def fit_range(self, start_idx: int, end_idx: int) -> None:
        """Fit frames start_idx..end_idx (0-based, inclusive) in the background.

        The Fit button runs it on every frame, the chart's range menu on a
        selection: locked frames are kept, manual fits are asked about.
        """
        if not self.state.images:
            return
        if self.is_busy():
            self.w.header.set_status("A fit is already running", "#f59e0b")
            return
        start_idx = max(0, start_idx)
        end_idx = min(end_idx, len(self.state.images) - 1)
        if start_idx > end_idx:
            return

        protected = self._choose_protected_frames(start_idx, end_idx)
        if protected is None:  # user cancelled
            return

        edge_job = None
        if self.state.detection_mode == "edge":
            try:
                edge_job = edge_support.build_job(self.state, self._get_max_radius())
            except Exception as exc:
                self.w.header.set_status(f"Cannot prepare Auto fit: {exc}", "#ef4444")
                return
        ml_job, ml_note = None, ""
        if edge_job is not None and self.ml_job_provider is not None:
            ml_job, ml_note = self.ml_job_provider()  # ML assist is on: Fit uses it
        self._launch(start_idx, end_idx, protected, edge_job, ml_job=ml_job,
                     notes=[ml_note] if ml_note else [])

    def run_ml_assist(self, ml_job, *, protected: frozenset[int], stored: dict,
                      frames=None) -> bool:
        """ML assist's Apply: re-measure *frames* (default: all) but *protected*.

        *stored* ({frame: StoredAuto}) are the plain-Auto results to reuse.
        Runs like Fit (progress, Stop, tracking); False if it cannot start.
        """
        if not self.state.images or self.state.detection_mode != "edge":
            return False
        if self.is_busy():
            self.w.header.set_status("A fit is already running", "#f59e0b")
            return False
        try:
            edge_job = edge_support.build_job(self.state, self._get_max_radius())
        except Exception as exc:
            self.w.header.set_status(f"Cannot prepare ML assist: {exc}", "#ef4444")
            return False
        n = len(self.state.images)
        todo = sorted({int(i) for i in (range(n) if frames is None else frames)
                       if 0 <= int(i) < n} - set(protected))
        if not todo:
            self.w.header.set_status("Nothing to measure: every frame is confirmed", "#f59e0b")
            return False
        self._launch(todo[0], todo[-1], frozenset(protected), edge_job,
                     ml_job=ml_job, stored=stored, frames=todo)
        return True

    def _launch(self, start_idx: int, end_idx: int, protected: frozenset[int], edge_job, *,
                ml_job=None, stored: dict | None = None, notes=(), frames=None) -> None:
        """Start the background run of frames start_idx..end_idx, or of *frames*."""
        worker = BatchWorker(
            images=self.state.images,
            start=start_idx,
            end=end_idx,
            sensitivity=self.state.img_thr,
            gridx=self.state.gridx,
            gridy=self.state.gridy,
            removing_factor_slider=self.state.removing_factor,
            bubble_cross_edges=self.state.bubble_cross_edges,
            removing_obj_radius=self.state.removing_obj_radius,
            gaussian_sigma=self.state.gaussian_sigma,
            clahe_clip=self.state.clahe_clip,
            closing_radius=self.state.closing_radius,
            opening_radius=self.state.opening_radius,
            detection_mode=self.state.detection_mode,
            gradient_boundary=self.state.gradient_boundary,
            area_max_regions=self.state.area_max_regions,
            max_radius=self._get_max_radius(),
            video_path=(self.state.video_path
                        if self.state.video_reader is not None else None),
            edge_job=edge_job,
            ml_job=ml_job,
            ml_stored=stored,
            frames=frames,
        )
        # Every slot is bound to *this* worker: results from an older run
        # (e.g. one still draining after Stop) can never reach new arrays.
        worker.progress.connect(partial(self._on_worker_progress, worker))
        worker.frame_done.connect(partial(self._on_worker_frame_done, worker))
        worker.frame_error.connect(partial(self._on_worker_frame_error, worker))
        worker.frame_meta.connect(partial(self._on_worker_frame_meta, worker))
        worker.finished.connect(partial(self._on_worker_finished, worker))
        worker.error.connect(
            lambda msg: self.w.header.set_status(msg, "#f59e0b"),
        )

        self._worker = worker
        self._tracker = None
        self._edge_job = edge_job
        self._run_notes = list(notes)
        self._ml_run = ml_job is not None and edge_job is not None
        self._run_frames = tuple(range(start_idx, end_idx + 1) if frames is None else frames)
        self._protected = protected
        self._stop_requested = False
        self._frame_errors = 0
        self._frames_done = 0
        self._autosave_failed = False
        self._running = True
        self._set_run_state(True)

        self.w.left_panel.fit_controls.set_running(True)
        self.w.header.set_status("Processing...", "#f59e0b")
        self.w.status_bar.update_mode("Fitting")
        self._auto_last_display = 0.0  # force first frame to display
        self._auto_start_time = time.monotonic()
        worker.start()

    def on_auto_stop(self) -> None:
        if self._worker:
            self._stop_requested = True
            self._worker.request_stop()
            if self._tracker is not None:
                self._tracker.request_stop()
            self.w.header.set_status("Stopping...", "#f59e0b")

    def on_auto_clear(self) -> None:
        if self.is_busy():
            return
        if self.state.images:
            self.state = self.state.with_results_initialized(len(self.state.images))
        self.w.left_panel.fit_controls.reset_progress()
        self.w.radius_chart.clear()
        display_frame(
            self.state, self.w, self.state.image_no, self._set_state,
            self._cache,
        )
        self.w.header.set_status("Cleared", "#10b981")
        self._results_changed()

    # -- manual-fix protection -------------------------------------------- #

    def _choose_protected_frames(self, start_idx: int, end_idx: int) -> frozenset[int] | None:
        """Frames a fit must not overwrite, or None if the user cancelled."""
        source = self.state.fit_source
        if source is None:
            return frozenset()
        in_range = np.arange(start_idx, end_idx + 1)
        locked_arr = self.state.fit_locked
        # Frames accepted in review are always kept; manual fits are asked about.
        locked = frozenset(int(i) for i in in_range
                           if locked_arr is not None and locked_arr[i])
        manual = [int(i) for i in in_range  # manual fits and drawn circles
                  if is_user_confirmed(source[i]) and i not in locked]
        if not manual:
            return locked
        policy = self._ask_manual_policy(len(manual))
        if policy == "cancel":
            return None
        if policy == "overwrite":
            return locked
        return locked | frozenset(manual)

    def _ask_manual_policy(self, n_manual: int) -> str:
        """Ask whether a fit may overwrite manual fixes: keep/overwrite/cancel."""
        box = QMessageBox(self.w)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle("Manual fixes in range")
        box.setText(
            f"{n_manual} frame(s) in this range were fixed by hand "
            "(manual fits or drawn circles).\n"
            "Fit can keep them or replace them with automatic fits."
        )
        keep = box.addButton("Keep manual fits", QMessageBox.ButtonRole.AcceptRole)
        overwrite = box.addButton("Overwrite", QMessageBox.ButtonRole.DestructiveRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(keep)
        box.exec()
        clicked = box.clickedButton()
        if clicked is keep:
            return "keep"
        if clicked is overwrite:
            return "overwrite"
        return "cancel"

    # -- internal signal handlers ----------------------------------------- #
    # Thread safety: BatchWorker runs on a QThread and communicates results
    # via queued signals, so every state mutation below runs on the main thread.

    def _on_worker_progress(self, worker, current: int, total: int) -> None:
        if worker is self._worker:
            self._on_auto_progress(current, total)

    def _on_worker_frame_done(self, worker, *result) -> None:
        if worker is not self._worker:
            logger.debug("Ignoring result from a stale worker")
            return
        self._on_auto_frame_done(*result)

    def _on_worker_frame_meta(self, worker, idx: int, meta: dict) -> None:
        if worker is not self._worker or idx in self._protected:
            return
        if self.state.fit_confidence is not None:
            self.state.fit_confidence[idx] = meta.get("confidence", np.nan)
        edge_support.store_meta(self.state, idx, meta)
        source = meta.get("source")  # ML assist: AUTO or ML for a stored result
        if source is not None and self.state.fit_source is not None and self.state.radius[idx] > 0:
            self.state.fit_source[idx] = source

    def _on_worker_frame_error(self, worker, idx: int, message: str) -> None:
        if worker is self._worker:
            self._frame_errors += 1

    def _on_worker_finished(self, worker) -> None:
        if worker is not self._worker:
            return
        if (not self._stop_requested and self._edge_job is not None
                and self.w.left_panel.measure_panel.edge_box.tracking_enabled()):
            self._start_tracking()
        else:
            self._finish_run(stopped=self._stop_requested)

    # -- sequence tracking (last step of an Edge-mode run) ----------------- #

    def _start_tracking(self) -> None:
        tracker = TrackingWorker(self.state, self._edge_job)
        tracker.solved.connect(partial(self._on_tracking_solved, tracker))
        tracker.failed.connect(partial(self._on_tracking_ended, tracker, "tracking failed: {}"))
        tracker.stopped.connect(partial(self._on_tracking_ended, tracker, "tracking stopped", ""))
        self._tracker = tracker
        self.w.left_panel.fit_controls.set_progress_text("Tracking the bubble across frames...")
        self.w.header.set_status("Tracking the bubble across frames...", "#f59e0b")
        tracker.start()

    def _on_tracking_solved(self, tracker, outcome) -> None:
        if tracker is not self._tracker:
            return
        changed = apply_tracking(self.state, outcome)
        logger.info("Tracking re-measured %d frame(s): %s", len(changed), changed[:20])
        if changed:
            self._run_notes.append(f"tracking re-measured {len(changed)} frame(s)")
        self._finish_run(stopped=False, review=review_items(outcome.result))

    def _on_tracking_ended(self, tracker, note: str, detail: str = "") -> None:
        """Tracking failed or was stopped: keep the single-frame results."""
        if tracker is not self._tracker:
            return
        self._run_notes.append(note.format(detail))
        self._finish_run(stopped=False)

    def _on_auto_progress(self, current: int, total: int) -> None:
        # Calculate ETA based on elapsed time
        if current > 0:
            elapsed = time.monotonic() - self._auto_start_time
            eta_sec = elapsed / current * (total - current)
            minutes, secs = divmod(int(eta_sec), 60)
            eta_text = f"{minutes}m {secs}s remaining"
        else:
            eta_text = "Calculating..."
        self.w.left_panel.fit_controls.set_progress(current, total, eta_text)
        self.bus.emit("run_progress", current, total)

    def _on_auto_frame_done(self, idx: int, radius: float, edge_xy, binary_roi,
                            rc: float, cc: float, count: int) -> None:
        self._frames_done += 1
        if idx in self._protected:
            return  # keep the user's manual fit

        state = self.state
        edge_support.clear_suggestion(state, idx)  # an Edge run's frame_meta brings the new guess
        state.radius[idx] = radius
        if radius > 0 and edge_xy is not None and edge_xy.shape[0] > 0:
            state.circle_fit_par[idx] = [rc, cc]
            state.circle_xy[idx] = edge_xy
            source = FitSource.AUTO
        else:
            # Do not leave a previous run's centre/edges behind a failed frame.
            state.circle_fit_par[idx] = [np.nan, np.nan]
            state.circle_xy[idx] = None
            source = FitSource.NONE
        if state.region_count is not None:
            state.region_count[idx] = count
        if state.fit_source is not None:
            state.fit_source[idx] = source

        if (idx + 1) % AUTOSAVE_INTERVAL == 0:
            self._write_autosave()

        self._update_live_view(idx, radius, edge_xy, binary_roi, rc, cc, count)

    def _update_live_view(self, idx, radius, edge_xy, binary_roi, rc, cc, count) -> None:
        # Throttle display updates so the UI thread can process user events
        now = time.monotonic()
        if now - self._auto_last_display < AUTO_DISPLAY_THROTTLE_MS / 1000.0:
            return
        self._auto_last_display = now

        # Update frame scrubber WITHOUT triggering on_frame_changed
        self.w.frame_scrubber.blockSignals(True)
        self.w.frame_scrubber.set_value(idx)
        self.w.frame_scrubber.blockSignals(False)
        self._update(image_no=idx)

        self.w.status_bar.update_frame(
            frame_to_display(idx), self.state.total_frames,
        )

        # Load and display original image for current frame
        try:
            cur_img, _, _, _ = normalize_source(
                current_source(self.state, idx), self.state.img_thr,
                self.state.gridx, self.state.gridy,
                gaussian_sigma=self.state.gaussian_sigma,
                clahe_clip=self.state.clahe_clip,
            )
            self.w.original_panel.set_image(cur_img)
            self.w.original_panel.draw_roi_rect(
                self.state.gridx, self.state.gridy,
            )
            # Draw fitted circle and edge points on original image
            if radius > 0 and edge_xy is not None and edge_xy.shape[0] > 0:
                if count == 0:
                    self.w.original_panel.draw_circle(rc, cc, radius, "#6366f1")
                self.w.original_panel.draw_points(edge_xy, "#ef4444", 2.0)
        except Exception:
            logger.debug("Live view update failed for frame %d", idx, exc_info=True)

        # Show binary result from worker (no disk I/O needed). Apply
        # invert_mask as a pure visual swap — same convention as the
        # display_frame / preview path so the binary panel behaves
        # consistently across the auto-fit, manual fit, and idle views.
        if binary_roi is not None:
            bin_display = ~binary_roi if self.state.invert_mask else binary_roi
            self.w.binary_panel.set_image(bin_display)

        # Update chart periodically (skip quality scoring for speed;
        # quality coloring is applied once when the run finishes)
        if radius > 0 and self.state.radius is not None:
            frames = np.arange(1, len(self.state.radius) + 1)
            self.w.radius_chart.plot_all(frames, self.state.radius)

    def _autosave_path(self) -> Path | None:
        """Next to the images, or next to the video (one file per video)."""
        if self.state.video_reader is not None and self.state.video_path:
            video = Path(self.state.video_path)
            return video.with_name(f".{video.stem}{AUTOSAVE_NAME}")
        if self.state.folder_path:
            return Path(self.state.folder_path) / AUTOSAVE_NAME
        return None

    def _write_autosave(self) -> bool:
        """Write the incremental autosave; never raises."""
        path = self._autosave_path()
        if path is None:
            return False
        try:
            export_r_data(
                str(path),
                self.state.radius,
                self.state.circle_fit_par,
                self.state.circle_xy,
                self.state.region_count,
            )
            logger.info("Auto-saved to %s", path)
            return True
        except Exception as exc:
            self._autosave_failed = True
            logger.warning("Auto-save to %s failed: %s", path, exc)
            return False

    def _on_auto_finished(self) -> None:
        """Kept for callers that finish a run without a worker reference."""
        self._finish_run(stopped=self._stop_requested)

    def _finish_run(self, stopped: bool, review=()) -> None:
        self._running = False
        self._tracker = None
        self._set_run_state(False)
        self.w.left_panel.fit_controls.set_running(False)
        self.w.status_bar.update_mode(MEASURE_MODE)
        self._write_autosave()  # final state, not just the last multiple of 50

        # Final full display update
        if self.state.images:
            display_frame(
                self.state, self.w, self.state.image_no, self._set_state,
                self._cache,
            )
        scores = refresh_chart(self.state, self.w)

        notes = list(self._run_notes)
        if self._ml_run and self.state.radius is not None:
            from bubbletrack.model.ml_assist import run_counts, run_summary
            notes.append(run_summary(run_counts(self.state, self._run_frames)))
        if self._frame_errors:
            notes.append(f"{self._frame_errors} frame(s) could not be read")
        if self._autosave_failed:
            notes.append("autosave failed (is the folder read-only?)")
        poor_count = 0
        if scores is not None and self.state.radius is not None:
            valid = self.state.radius > 0
            poor_count = int(np.sum(valid & (scores < QUALITY_WARN_THRESHOLD)))
            logger.info("Fit quality: %d / %d valid fits below threshold",
                        poor_count, int(valid.sum()))
        if poor_count and not stopped:
            notes.append(f"{poor_count} unreliable fits detected")

        head = (f"Stopped after {self._frames_done} frame(s)" if stopped else "Done")
        text = head + (" — " + "; ".join(notes) if notes else "")
        warn = stopped or bool(notes)
        self.w.header.set_status(text, "#f59e0b" if warn else "#10b981")
        # The tracking review list (empty when tracking did not run) and the
        # frames it re-measured.
        self.bus.emit("tracking_review", tuple(review))
        self._results_changed()
        self.bus.emit("run_finished", {"stopped": stopped, "ml": self._ml_run,
                                       "edge": self._edge_job is not None,
                                       "frames": self._run_frames})
