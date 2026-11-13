"""AutoController — batch automatic fitting across frame ranges."""

from __future__ import annotations

import logging
import time
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
from PyQt6.QtWidgets import QFileDialog, QMessageBox

from bubbletrack.ui.batch_config_dialog import BatchConfigDialog

from bubbletrack.controller import edge_support
from bubbletrack.controller.base import BaseController
from bubbletrack.controller.batch_folder_worker import BatchFolderWorker
from bubbletrack.controller.display_mixin import (
    current_source, display_frame, refresh_chart,
)
from bubbletrack.controller.tracking import TrackingWorker, apply_tracking, review_items
from bubbletrack.controller.worker import BatchWorker
from bubbletrack.event_bus import EventBus
from bubbletrack.model.cache import ImageCache
from bubbletrack.model.constants import AUTO_DISPLAY_THROTTLE_MS, QUALITY_WARN_THRESHOLD
from bubbletrack.model.conventions import display_to_frame, frame_to_display
from bubbletrack.model.export import export_r_data
from bubbletrack.model.frame_source import normalize_source
from bubbletrack.model.provenance import FitSource, is_user_confirmed

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
        self._batch_worker: BatchFolderWorker | None = None
        self._running = False
        self._batch_running = False
        self._auto_last_display: float = 0.0
        self._auto_start_time: float = 0.0
        self._get_max_radius = get_max_radius
        self._cache = cache
        self._batch_results: list = []  # Accumulates FolderResult during batch
        self._batch_params: tuple | None = None  # (gridx, gridy, sensitivity) at start
        self._batch_store = None  # last BatchResultStore, for "Load into Main View"
        # Per-run bookkeeping
        self._protected: frozenset[int] = frozenset()
        self._stop_requested = False
        self._frame_errors = 0
        self._frames_done = 0
        self._autosave_failed = False
        self._edge_job = None  # the run's EdgeJob, reused by tracking
        self._run_notes: list[str] = []  # e.g. what tracking changed

    # -- run state -------------------------------------------------------- #

    def is_busy(self) -> bool:
        """True while a single-folder or multi-folder fit is running."""
        return self._running or self._batch_running

    def stop_and_wait(self, timeout_ms: int = 10_000) -> None:
        """Stop any running worker and wait for its thread to exit."""
        for worker in (self._worker, self._tracker, self._batch_worker):
            if worker is not None and worker.isRunning():
                worker.request_stop()
                worker.wait(timeout_ms)

    def _set_run_state(self, running: bool) -> None:
        self.bus.emit("run_state_changed", running)

    # -- public handlers -------------------------------------------------- #

    def on_auto_fit(self) -> None:
        if not self.state.images:
            return
        if self.is_busy():
            self.w.header.set_status("A fit is already running", "#f59e0b")
            return

        start, end = self.w.left_panel.automatic_tab.get_range()
        if start > end:
            self.w.header.set_status(
                f"Start frame ({start}) must not be after end frame ({end})",
                "#ef4444",
            )
            return
        self.fit_range(display_to_frame(start), display_to_frame(end))

    def fit_range(self, start_idx: int, end_idx: int) -> None:
        """Fit frames start_idx..end_idx (0-based, inclusive) in the background.

        Same run as Fit All (locked frames kept, manual fits asked about),
        without changing the From/To fields of the Automatic tab.
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
        self._run_notes = []
        self._protected = protected
        self._stop_requested = False
        self._frame_errors = 0
        self._frames_done = 0
        self._autosave_failed = False
        self._running = True
        self._set_run_state(True)

        self.w.left_panel.automatic_tab.set_running(True)
        self.w.header.set_status("Processing...", "#f59e0b")
        self.w.status_bar.update_mode("Automatic")
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
        if self._batch_worker:
            self._batch_worker.request_stop()
            self.w.header.set_status("Stopping batch...", "#f59e0b")

    def on_auto_clear(self) -> None:
        if self.is_busy():
            return
        if self.state.images:
            self.state = self.state.with_results_initialized(len(self.state.images))
        self.w.left_panel.automatic_tab.reset_progress()
        self.w.radius_chart.clear()
        display_frame(
            self.state, self.w, self.state.image_no, self._set_state,
            self._cache,
        )
        self.w.header.set_status("Cleared", "#10b981")
        self._results_changed()

    # -- manual-fix protection -------------------------------------------- #

    def _choose_protected_frames(self, start_idx: int, end_idx: int) -> frozenset[int] | None:
        """Frames Fit All must not overwrite, or None if the user cancelled."""
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
        """Ask whether Fit All may overwrite manual fixes: keep/overwrite/cancel."""
        box = QMessageBox(self.w)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle("Manual fixes in range")
        box.setText(
            f"{n_manual} frame(s) in this range were fixed by hand "
            "(manual fits or drawn circles).\n"
            "Fit All can keep them or replace them with automatic fits."
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

    def _on_worker_frame_error(self, worker, idx: int, message: str) -> None:
        if worker is self._worker:
            self._frame_errors += 1

    def _on_worker_finished(self, worker) -> None:
        if worker is not self._worker:
            return
        if (not self._stop_requested and self._edge_job is not None
                and self.w.left_panel.pretune_tab.edge_box.tracking_enabled()):
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
        self.w.left_panel.automatic_tab.set_progress_text("Tracking the bubble across frames...")
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
        self.w.left_panel.automatic_tab.set_progress(current, total, eta_text)

    def _on_auto_frame_done(self, idx: int, radius: float, edge_xy, binary_roi,
                            rc: float, cc: float, count: int) -> None:
        self._frames_done += 1
        if idx in self._protected:
            return  # keep the user's manual fit

        state = self.state
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
        self.w.left_panel.automatic_tab.set_running(False)
        self._write_autosave()  # final state, not just the last multiple of 50

        # Final full display update
        if self.state.images:
            display_frame(
                self.state, self.w, self.state.image_no, self._set_state,
                self._cache,
            )
        scores = refresh_chart(self.state, self.w)

        notes = list(self._run_notes)
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

    # ------------------------------------------------------------------ #
    #  Batch processing of many sources (image folders and videos)
    # ------------------------------------------------------------------ #

    def on_batch_folders(self) -> None:
        """Ask for a parent directory and process every folder and video in it."""
        if self.is_busy():
            self.w.header.set_status("A fit is already running", "#f59e0b")
            return
        if not self.state.images:
            QMessageBox.information(
                self.w, "Load a source first",
                "Load one experiment folder or video and tune the ROI and "
                "parameters on it first.\n\nBatch processing applies exactly "
                "those settings to every folder and video.",
            )
            return

        root = QFileDialog.getExistingDirectory(
            self.w, "Select Parent Directory Containing Experiments",
        )
        if not root:
            return

        from bubbletrack.model.batch_experiments import find_batch_sources
        sources = find_batch_sources(root)
        if not sources:
            QMessageBox.information(
                self.w, "No Experiments Found",
                f"No image folders or videos found under:\n{root}\n\n"
                "Image folders are subfolders holding TIFF, PNG, JPG or BMP "
                "files. Videos (AVI, MP4, MOV, MKV) may sit in the folder "
                "itself or in its subfolders.",
            )
            return

        edge_params = None
        if self.state.detection_mode == "edge":
            from bubbletrack.model import edge_pipeline
            try:
                edge_params = edge_pipeline.params_from_state(self.state)
            except Exception as exc:
                self.w.header.set_status(f"Cannot prepare Auto batch: {exc}", "#ef4444")
                return

        pp = self.w.left_panel.post_processing
        dlg = BatchConfigDialog(
            sources,
            detection_mode=self.state.detection_mode,
            default_fps=pp.get_fps(),
            default_um2px=pp.get_scale(),
            default_fit_length=pp.get_fit_length(),
            parent=self.w,
        )
        if dlg.exec() != BatchConfigDialog.DialogCode.Accepted:
            return
        cfg = dlg.get_config()
        if cfg.sources and not self.is_busy():
            self._start_batch_worker(cfg, edge_params)

    def _start_batch_worker(self, cfg, edge_params) -> None:
        """Start a BatchFolderWorker with the current parameters."""
        state = self.state
        at = self.w.left_panel.automatic_tab
        at.set_batch_running(True)
        at.set_batch_status(f"Starting batch: {len(cfg.sources)} source(s)...")
        self.w.header.set_status("Batch processing...", "#f59e0b")

        # Record the parameters actually used, not those current at the end.
        self._batch_params = (state.gridx, state.gridy, state.img_thr, state.detection_mode)
        worker = BatchFolderWorker(
            sources=cfg.sources,
            sensitivity=state.img_thr,
            gridx=state.gridx,
            gridy=state.gridy,
            removing_factor_slider=state.removing_factor,
            bubble_cross_edges=state.bubble_cross_edges,
            removing_obj_radius=state.removing_obj_radius,
            gaussian_sigma=state.gaussian_sigma,
            clahe_clip=state.clahe_clip,
            closing_radius=state.closing_radius,
            opening_radius=state.opening_radius,
            detection_mode=state.detection_mode,
            gradient_boundary=state.gradient_boundary,
            area_max_regions=state.area_max_regions,
            max_radius=self._get_max_radius(),
            export_physical=cfg.export_physical,
            fps=cfg.fps,
            um2px=cfg.um2px,
            rmax_fit_length=cfg.rmax_fit_length,
            edge_params=edge_params,
        )
        self._batch_results = []
        self._batch_error = ""
        worker.folder_started.connect(self._on_batch_folder_started)
        worker.frame_progress.connect(self._on_batch_frame_progress)
        worker.folder_done.connect(self._on_batch_folder_done)
        worker.result_ready.connect(self._on_batch_folder_result)
        worker.error.connect(self._on_batch_error)
        worker.all_done.connect(self._on_batch_all_done)
        self._batch_worker = worker
        self._batch_running = True
        self._set_run_state(True)
        worker.start()

    def _on_batch_folder_started(self, idx: int, total: int, name: str) -> None:
        at = self.w.left_panel.automatic_tab
        at.set_batch_status(f"Source {idx}/{total}: {name}")
        self.w.header.set_status(f"Batch: {name} ({idx}/{total})", "#f59e0b")

    def _on_batch_frame_progress(self, done: int, total: int) -> None:
        self.w.left_panel.automatic_tab.set_batch_progress(done, total)

    def _on_batch_folder_done(self, name: str, success: bool, msg: str) -> None:
        if success:
            logger.info("Batch source %s: ✓ — %s", name, msg)
        else:
            logger.warning("Batch source %s: ✗ — %s", name, msg)

    def _on_batch_folder_result(self, result) -> None:
        """One FolderResult per source (failed sources included)."""
        self._batch_results.append(result)

    def _on_batch_error(self, message: str) -> None:
        self._batch_error = message
        logger.error("Batch run failed: %s", message)

    def _on_batch_all_done(self, folders_ok: int, total_fitted: int) -> None:
        results = tuple(self._batch_results)
        error = getattr(self, "_batch_error", "")
        n_failed = sum(1 for r in results if not r.success)
        summary = f"{folders_ok}/{len(results)} source(s), {total_fitted} frames fitted"
        if n_failed:
            summary += f", {n_failed} failed"

        at = self.w.left_panel.automatic_tab
        at.set_batch_running(False)
        at.set_batch_status(f"Complete: {summary}")
        if error:
            self.w.header.set_status(error, "#ef4444")
        else:
            self.w.header.set_status(f"Batch done: {summary}",
                                     "#f59e0b" if n_failed else "#10b981")
        self._batch_worker = None
        self._batch_running = False
        self._set_run_state(False)

        if results or error:
            from bubbletrack.model.batch_result import BatchResultStore
            gridx, gridy, sensitivity, mode = self._batch_params or (
                self.state.gridx, self.state.gridy, self.state.img_thr,
                self.state.detection_mode,
            )
            store = BatchResultStore(
                results=results,
                timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
                gridx=gridx,
                gridy=gridy,
                sensitivity=sensitivity,
                detection_mode=mode,
                error=error,
            )
            self._batch_store = store
            self._show_batch_browser(store)
        self._batch_results = []

    def _show_batch_browser(self, store) -> None:
        from bubbletrack.ui.batch_results_dialog import BatchResultsBrowserDialog
        # Store reference to prevent garbage collection before dialog is shown
        self._batch_browser_dlg = BatchResultsBrowserDialog(store, parent=self.w)
        self._batch_browser_dlg.load_folder_requested.connect(self._on_load_batch_folder)
        self._batch_browser_dlg.show()  # Non-modal

    def _on_load_batch_folder(self, result) -> None:
        """Load a batch source (folder or video) and restore its results.

        Parameters
        ----------
        result : FolderResult
            The batch result to restore.
        """
        if self.is_busy():
            self.w.header.set_status("Stop the running fit first", "#f59e0b")
            return
        if not result.has_results:
            self.w.header.set_status(
                f"{result.folder_name} has no results to load", "#f59e0b")
            return

        # 1. Open the source the normal way (blank results, frame 0 displayed)
        is_video = result.source_kind == "video"
        self.bus.emit("load_video" if is_video else "load_folder", result.folder_path)
        state = self.state
        if is_video:
            loaded = state.video_reader is not None and state.video_path == result.folder_path
        else:
            loaded = state.video_reader is None and state.folder_path == result.folder_path
        if state.radius is None or not loaded:
            logger.warning("Batch source %s was not loaded", result.folder_path)
            return

        # 2. Restore the ROI the batch actually used
        store = self._batch_store
        if store is not None:
            self._update(gridx=tuple(store.gridx), gridy=tuple(store.gridy))
            self.w.original_panel.set_roi_text(self.state.gridx, self.state.gridy)

        # 3. Restore batch results (radius, centres, edges, confidence, flags, metrics)
        from bubbletrack.model.batch_result import restore_into_state
        n_copy = restore_into_state(self.state, result)
        n_state = len(self.state.radius)

        # 4. Re-display current frame and chart
        display_frame(
            self.state, self.w, self.state.image_no, self._set_state,
            self._cache,
        )
        if result.circle_xy is not None:
            refresh_chart(self.state, self.w)
        else:
            frames = np.arange(1, n_state + 1)
            scores = np.zeros(n_state)
            scores[:n_copy] = result.quality_scores[:n_copy]
            self.w.radius_chart.plot_all(frames, self.state.radius, scores)

        n_fitted = int(np.sum(self.state.radius[:n_copy] > 0))
        self.w.header.set_status(
            f"Loaded {result.folder_name}: {n_fitted}/{n_state} fitted", "#10b981",
        )
        self._results_changed()
