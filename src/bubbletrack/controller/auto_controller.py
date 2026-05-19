"""AutoController — batch automatic fitting across frame ranges."""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QFileDialog, QMessageBox

from bubbletrack.ui.batch_config_dialog import BatchConfigDialog

from bubbletrack.controller.base import BaseController
from bubbletrack.controller.batch_folder_worker import BatchFolderWorker
from bubbletrack.controller.display_mixin import display_frame, refresh_chart
from bubbletrack.controller.worker import BatchWorker
from bubbletrack.event_bus import EventBus
from bubbletrack.model.cache import ImageCache
from bubbletrack.model.circle_fit import circle_fit_taubin
from bubbletrack.model.constants import AUTO_DISPLAY_THROTTLE_MS, QUALITY_WARN_THRESHOLD
from bubbletrack.model.conventions import frame_to_display
from bubbletrack.model.export import export_r_data
from bubbletrack.model.image_io import load_and_normalize
from bubbletrack.model.state import update_state

logger = logging.getLogger(__name__)

AUTOSAVE_INTERVAL = 50  # frames between incremental auto-saves


class AutoController(BaseController):
    """Manages automatic batch fitting with a background worker thread."""

    def __init__(self, bus: EventBus, get_state, set_state, window,
                 get_max_radius, cache: ImageCache | None = None) -> None:
        super().__init__(bus, get_state, set_state, window)
        self._worker: BatchWorker | None = None
        self._batch_worker: BatchFolderWorker | None = None
        self._auto_last_display: float = 0.0
        self._auto_start_time: float = 0.0
        self._get_max_radius = get_max_radius
        self._cache = cache
        self._batch_results: list = []  # Accumulates FolderResult during batch

    # -- public handlers -------------------------------------------------- #

    def on_auto_fit(self) -> None:
        if not self.state.images:
            return

        from bubbletrack.model.conventions import display_to_frame

        start, end = self.w.left_panel.automatic_tab.get_range()
        start_idx = display_to_frame(start)
        end_idx = display_to_frame(end)

        self._worker = BatchWorker(
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
            max_radius=self._get_max_radius(),
        )
        self._worker.progress.connect(self._on_auto_progress)
        self._worker.frame_done.connect(self._on_auto_frame_done)
        self._worker.finished.connect(self._on_auto_finished)
        self._worker.error.connect(
            lambda msg: self.w.header.set_status(msg, "#f59e0b"),
        )

        self.w.left_panel.automatic_tab.set_running(True)
        self.w.header.set_status("Processing...", "#f59e0b")
        self.w.status_bar.update_mode("Automatic")
        self._auto_last_display = 0.0  # force first frame to display
        self._auto_start_time = time.monotonic()
        self._worker.start()

    def on_auto_stop(self) -> None:
        if self._worker:
            self._worker.request_stop()
            self.w.header.set_status("Stopping...", "#f59e0b")
        if self._batch_worker:
            self._batch_worker.request_stop()
            self.w.header.set_status("Stopping batch...", "#f59e0b")

    def on_auto_clear(self) -> None:
        if self.state.images:
            self.state = self.state.with_results_initialized(len(self.state.images))
        self.w.left_panel.automatic_tab.reset_progress()
        self.w.radius_chart.clear()
        display_frame(
            self.state, self.w, self.state.image_no, self._set_state,
            self._cache,
        )
        self.w.header.set_status("Cleared", "#10b981")

    # -- internal signal handlers ----------------------------------------- #
    # Thread safety: BatchWorker runs on a QThread and communicates results
    # via frame_done signal. State mutations happen here on the main thread
    # (guaranteed by Qt's queued signal/slot connection for cross-thread signals).

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

    def _on_auto_frame_done(self, idx: int, radius: float, edge_xy, binary_roi) -> None:
        # Always update state (fast, no UI)
        self.state.radius[idx] = radius
        if edge_xy is not None and edge_xy.shape[0] > 0:
            rc, cc, _ = circle_fit_taubin(edge_xy)
            self.state.circle_fit_par[idx] = [rc, cc]
            self.state.circle_xy[idx] = edge_xy

        # Incremental auto-save every AUTOSAVE_INTERVAL frames
        if (idx + 1) % AUTOSAVE_INTERVAL == 0 and self.state.folder_path:
            autosave_path = Path(self.state.folder_path) / ".bubbletrack_autosave.mat"
            try:
                export_r_data(
                    str(autosave_path),
                    self.state.radius,
                    self.state.circle_fit_par,
                    self.state.circle_xy,
                )
                logger.info("Auto-saved at frame %d to %s", idx, autosave_path)
            except Exception as exc:
                logger.warning("Auto-save failed at frame %d: %s", idx, exc)

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
            cur_img, _, _, _ = load_and_normalize(
                self.state.images[idx], self.state.img_thr,
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
                rc, cc, _ = circle_fit_taubin(edge_xy)
                self.w.original_panel.draw_circle(rc, cc, radius, "#6366f1")
                self.w.original_panel.draw_points(edge_xy, "#ef4444", 2.0)
        except Exception:
            pass

        # Show binary result from worker (no disk I/O needed). Apply
        # invert_mask as a pure visual swap — same convention as the
        # display_frame / preview path so the binary panel behaves
        # consistently across the auto-fit, manual fit, and idle views.
        if binary_roi is not None:
            bin_display = ~binary_roi if self.state.invert_mask else binary_roi
            self.w.binary_panel.set_image(bin_display)

        # Update chart periodically (skip quality scoring for speed;
        # quality coloring is applied once in _on_auto_finished)
        if radius > 0 and self.state.radius is not None:
            frames = np.arange(1, len(self.state.radius) + 1)
            self.w.radius_chart.plot_all(frames, self.state.radius)

    def _on_auto_finished(self) -> None:
        self.w.left_panel.automatic_tab.set_running(False)
        self.w.header.set_status("Done", "#10b981")
        # Final full display update
        display_frame(
            self.state, self.w, self.state.image_no, self._set_state,
            self._cache,
        )
        # Refresh chart with quality-based coloring
        scores = refresh_chart(self.state, self.w)
        if scores is not None:
            valid = self.state.radius > 0
            poor_count = int(np.sum(valid & (scores < QUALITY_WARN_THRESHOLD)))
            if poor_count > 0:
                self.w.header.set_status(
                    f"Done — {poor_count} unreliable fits detected", "#f59e0b",
                )
            logger.info("Fit quality: %d / %d valid fits below threshold",
                        poor_count, int(valid.sum()))

    # ------------------------------------------------------------------ #
    #  Batch multi-folder processing
    # ------------------------------------------------------------------ #

    def on_batch_folders(self) -> None:
        """Prompt user for a parent directory and process all experiment folders."""
        root = QFileDialog.getExistingDirectory(
            self.w, "Select Parent Directory Containing Experiment Folders",
        )
        if not root:
            return

        # Scan for experiment subfolders
        from bubbletrack.model.batch_experiments import find_experiment_folders
        folders = find_experiment_folders(root)
        if not folders:
            QMessageBox.information(
                self.w, "No Experiments Found",
                f"No image folders found under:\n{root}\n\n"
                "Each subfolder must contain image files (TIFF, PNG, JPG, BMP).",
            )
            return

        # Read default physical params from PostProcessing panel
        pp = self.w.left_panel.post_processing
        dlg = BatchConfigDialog(
            folders,
            default_fps=pp.get_fps(),
            default_um2px=pp.get_scale(),
            default_fit_length=pp.get_fit_length(),
            parent=self.w,
        )
        if dlg.exec() != BatchConfigDialog.DialogCode.Accepted:
            return

        cfg = dlg.get_config()

        at = self.w.left_panel.automatic_tab
        at.set_batch_running(True)
        at.set_batch_status(f"Starting batch: {len(cfg.folders)} folders...")
        self.w.header.set_status("Batch processing...", "#f59e0b")

        self._batch_worker = BatchFolderWorker(
            folders=cfg.folders,
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
            max_radius=self._get_max_radius(),
            export_physical=cfg.export_physical,
            fps=cfg.fps,
            um2px=cfg.um2px,
            rmax_fit_length=cfg.rmax_fit_length,
        )
        self._batch_results = []
        self._batch_worker.folder_started.connect(self._on_batch_folder_started)
        self._batch_worker.frame_progress.connect(self._on_batch_frame_progress)
        self._batch_worker.folder_done.connect(self._on_batch_folder_done)
        self._batch_worker.folder_result_ready.connect(self._on_batch_folder_result)
        self._batch_worker.all_done.connect(self._on_batch_all_done)
        self._batch_worker.start()

    def _on_batch_folder_started(self, idx: int, total: int, name: str) -> None:
        at = self.w.left_panel.automatic_tab
        at.set_batch_status(f"Folder {idx}/{total}: {name}")
        self.w.header.set_status(f"Batch: {name} ({idx}/{total})", "#f59e0b")

    def _on_batch_frame_progress(self, done: int, total: int) -> None:
        self.w.left_panel.automatic_tab.set_batch_progress(done, total)

    def _on_batch_folder_done(self, name: str, success: bool, msg: str) -> None:
        status = "✓" if success else "✗"
        logger.info("Batch folder %s: %s — %s", name, status, msg)

    def _on_batch_folder_result(
        self, folder_path: str, n_frames: int, n_fitted: int,
        radius: np.ndarray, quality_scores: np.ndarray,
    ) -> None:
        from bubbletrack.model.batch_result import FolderResult
        self._batch_results.append(FolderResult(
            folder_path=folder_path,
            folder_name=Path(folder_path).name,
            success=True,
            message=f"{n_fitted}/{n_frames} fitted",
            n_frames=n_frames,
            n_fitted=n_fitted,
            radius=radius,
            quality_scores=quality_scores,
        ))

    def _on_batch_all_done(self, folders_ok: int, total_fitted: int) -> None:
        at = self.w.left_panel.automatic_tab
        at.set_batch_running(False)
        at.set_batch_status(
            f"Complete: {folders_ok} folder(s), {total_fitted} frames fitted"
        )
        self.w.header.set_status(
            f"Batch done: {folders_ok} folders, {total_fitted} fitted", "#10b981",
        )
        self._batch_worker = None

        if self._batch_results:
            from bubbletrack.model.batch_result import BatchResultStore
            store = BatchResultStore(
                results=tuple(self._batch_results),
                timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
                gridx=self.state.gridx,
                gridy=self.state.gridy,
                sensitivity=self.state.img_thr,
            )
            self._show_batch_browser(store)
        self._batch_results = []

    def _show_batch_browser(self, store) -> None:
        from bubbletrack.ui.batch_results_dialog import BatchResultsBrowserDialog
        # Store reference to prevent garbage collection before dialog is shown
        self._batch_browser_dlg = BatchResultsBrowserDialog(store, parent=self.w)
        self._batch_browser_dlg.load_folder_requested.connect(self._on_load_batch_folder)
        self._batch_browser_dlg.show()  # Non-modal

    def _on_load_batch_folder(self, result) -> None:
        """Load a batch folder into the main view and restore its results.

        Parameters
        ----------
        result : FolderResult
            The batch result to restore (contains radius + quality_scores).
        """
        # 1. Load the folder (scans images, initialises blank results, displays frame 0)
        self.bus.emit("load_folder", result.folder_path)

        # 2. Restore batch-fitted radius into state
        if self.state.radius is None:
            return
        n_state = len(self.state.radius)
        n_batch = len(result.radius)
        n_copy = min(n_state, n_batch)
        self.state.radius[:n_copy] = result.radius[:n_copy]

        # 3. Re-display current frame (to show fitted/unfitted binary correctly)
        display_frame(
            self.state, self.w, self.state.image_no, self._set_state,
            self._cache,
        )

        # 4. Plot chart with stored quality scores (circle_xy not available
        #    so refresh_chart cannot recompute quality — use stored scores)
        frames = np.arange(1, n_state + 1)
        if n_state == n_batch:
            scores = result.quality_scores
        else:
            # Frame count changed since batch; pad/truncate scores
            scores = np.zeros(n_state)
            scores[:n_copy] = result.quality_scores[:n_copy]
        self.w.radius_chart.plot_all(frames, self.state.radius, scores)

        n_fitted = int(np.sum(self.state.radius[:n_copy] > 0))
        self.w.header.set_status(
            f"Loaded {result.folder_name}: {n_fitted}/{n_state} fitted", "#10b981",
        )
