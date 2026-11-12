"""AppController — thin signal router that owns state and sub-controllers."""

from __future__ import annotations

import logging
import os

import numpy as np
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QFileDialog, QMessageBox

from bubbletrack.controller import edge_support
from bubbletrack.controller.auto_controller import AutoController
from bubbletrack.controller.display_mixin import (
    current_source, display_frame, refresh_chart,
)
from bubbletrack.controller.export_controller import ExportController
from bubbletrack.controller.file_controller import FileController
from bubbletrack.controller.manual_controller import ManualController
from bubbletrack.controller.pretune_controller import PretuneController
from bubbletrack.controller.session_controller import SessionController
from bubbletrack.event_bus import EventBus
from bubbletrack.model.cache import ImageCache
from bubbletrack.model.circle_fit import circle_fit_taubin
from bubbletrack.model.constants import DISPLAY_DEBOUNCE_MS, PREVIEW_DEBOUNCE_MS
from bubbletrack.model.config import load_config
from bubbletrack.model.detection import (
    detect_bubble_gradient, detect_bubble_with_mode, measure_detection,
)
from bubbletrack.model.frame_source import normalize_source
from bubbletrack.model.provenance import FitSource
from bubbletrack.model.removing_factor import compute_removing_factor
from bubbletrack.model.state import AppState, update_state
from bubbletrack.ui.shortcuts import setup_shortcuts

logger = logging.getLogger(__name__)


class AppController:
    """Thin orchestrator: owns state, creates sub-controllers, wires Qt signals.

    All domain logic lives in the sub-controllers.  This class only:
    1. Holds the canonical ``AppState`` instance.
    2. Creates and configures sub-controllers.
    3. Connects Qt widget signals to the correct handler.
    """

    def __init__(self, window) -> None:
        self.w = window
        # Load persisted config and apply to initial state
        saved = load_config()
        self._state = update_state(AppState(), **saved) if saved else AppState()
        self._max_radius: float = float("inf")

        # LRU image cache (200 MB limit)
        self._image_cache = ImageCache()

        # Event bus for decoupled inter-controller communication
        self.bus = EventBus()

        # Debounce timers for slider-driven updates
        self._display_timer = QTimer()
        self._display_timer.setSingleShot(True)
        self._display_timer.setInterval(DISPLAY_DEBOUNCE_MS)
        self._display_timer.timeout.connect(
            lambda: display_frame(
                self._state, self.w, self._state.image_no, self._set_state,
                self._image_cache,
            )
        )

        self._preview_timer = QTimer()
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(PREVIEW_DEBOUNCE_MS)

        # Create sub-controllers (shared state via get/set accessors)
        gs, ss = self._get_state, self._set_state

        self.file_ctrl = FileController(
            self.bus, gs, ss, self.w, self._set_max_radius,
            self._image_cache,
            is_busy=lambda: self.auto_ctrl.is_busy(),
        )
        self.pretune_ctrl = PretuneController(
            self.bus, gs, ss, self.w,
            self._display_timer, self._preview_timer, self._get_max_radius,
            self._image_cache,
        )
        self.manual_ctrl = ManualController(
            self.bus, gs, ss, self.w, self._get_max_radius,
            self._image_cache,
        )
        self.auto_ctrl = AutoController(
            self.bus, gs, ss, self.w, self._get_max_radius,
            self._image_cache,
        )
        self.export_ctrl = ExportController(self.bus, gs, ss, self.w)
        self.session_ctrl = SessionController(
            self.bus, gs, ss, self.w, self.file_ctrl,
            is_busy=lambda: self.auto_ctrl.is_busy(),
            locate_folder=lambda original: self._locate_moved_folder(original),
            after_restore=lambda: self._after_session_restore(),
        )

        # Wire preview timer to pretune handler
        self._preview_timer.timeout.connect(self.pretune_ctrl.preview_detection)

        # Clear manual points on tab change
        self.bus.subscribe("tab_changed", lambda idx: self.manual_ctrl.clear_points())

        # Batch browser "Load into Main View" → reuse the normal loading logic
        self.bus.subscribe("load_folder", self.file_ctrl.on_folder_selected)
        self.bus.subscribe("load_video", self.file_ctrl.on_video_selected)

        # Lock everything that mutates results while a worker writes into them
        self.bus.subscribe("run_state_changed", self._apply_run_lock)
        self.w.set_close_guard(self._confirm_close)

        self._connect_signals()
        self._sync_ui_from_state()

        # Review queue dock (suspicious frames, one-key triage)
        from bubbletrack.controller.review_controller import install_review
        self.review = install_review(self.w, self)

        # Keyboard shortcuts
        self._shortcuts = setup_shortcuts(self.w, {
            "frame_prev": lambda: self.file_ctrl.on_frame_changed(
                max(0, self._state.image_no - 1),
            ),
            "frame_next": lambda: self.file_ctrl.on_frame_changed(
                min(self._state.total_frames - 1, self._state.image_no + 1),
            ),
            "zoom_in": self.w.original_panel._zoom_in,
            "zoom_out": self.w.original_panel._zoom_out,
            "zoom_reset": self.w.original_panel._zoom_reset,
            "undo": self.manual_ctrl.undo_last_point,
            "play_pause": self.w.frame_scrubber.toggle_play,
        })

    # ------------------------------------------------------------------ #
    #  State accessors (shared with sub-controllers)
    # ------------------------------------------------------------------ #

    def _get_state(self) -> AppState:
        return self._state

    def _set_state(self, new: AppState) -> None:
        self._state = new

    def _get_max_radius(self) -> float:
        return self._max_radius

    def _set_max_radius(self, value: float) -> None:
        self._max_radius = value

    # ------------------------------------------------------------------ #
    #  Signal wiring
    # ------------------------------------------------------------------ #

    def _connect_signals(self) -> None:
        lp = self.w.left_panel

        # Image source
        lp.image_source.folder_selected.connect(self.file_ctrl.on_folder_selected)
        lp.image_source.video_selected.connect(self.file_ctrl.on_video_selected)

        # Frame scrubber
        self.w.frame_scrubber.value_changed.connect(self.file_ctrl.on_frame_changed)

        # Pre-tune
        pt = lp.pretune_tab
        pt.threshold_changed.connect(self.pretune_ctrl.on_threshold_changed)
        pt.invert_mask_changed.connect(self.pretune_ctrl.on_invert_mask_changed)
        pt.removing_factor_changed.connect(self.pretune_ctrl.on_removing_factor_changed)
        pt.edges_changed.connect(self.pretune_ctrl.on_edges_changed)
        pt.fit_clicked.connect(self.pretune_ctrl.on_pretune_fit)
        pt.autotune_clicked.connect(self.pretune_ctrl.on_autotune)
        pt.frame_selected.connect(self.file_ctrl.on_tab_frame_selected)
        pt.filters_changed.connect(self.pretune_ctrl.on_filters_changed)

        # Manual
        mt = lp.manual_tab
        mt.select_clicked.connect(self.manual_ctrl.on_manual_select)
        mt.done_clicked.connect(self.manual_ctrl.on_manual_done)
        mt.clear_clicked.connect(self.manual_ctrl.on_manual_clear)
        mt.frame_selected.connect(self.file_ctrl.on_tab_frame_selected)
        self.w.original_panel.point_clicked.connect(self.manual_ctrl.on_point_clicked)
        self.w.original_panel.point_clicked.connect(self.pretune_ctrl.on_point_clicked)
        eb = pt.edge_box
        eb.use_current_reference_clicked.connect(self.pretune_ctrl.on_use_current_as_reference)
        eb.auto_reference_clicked.connect(self.pretune_ctrl.on_auto_reference)
        eb.pick_seed_clicked.connect(self.pretune_ctrl.on_pick_seed)
        eb.clear_seed_clicked.connect(self.pretune_ctrl.on_clear_seed)
        self.bus.subscribe("source_loaded", lambda _n: self.pretune_ctrl.refresh_edge_labels())

        # Tab switching
        lp.tab_bar.tab_changed.connect(self.file_ctrl.on_tab_changed)

        # Automatic
        at = lp.automatic_tab
        at.fit_clicked.connect(self.auto_ctrl.on_auto_fit)
        at.stop_clicked.connect(self.auto_ctrl.on_auto_stop)
        at.clear_clicked.connect(self.auto_ctrl.on_auto_clear)
        at.batch_folders_clicked.connect(self.auto_ctrl.on_batch_folders)

        # Post processing
        pp = lp.post_processing
        pp.export_r_data_clicked.connect(self.export_ctrl.on_export_r_data)
        pp.export_rof_t_clicked.connect(self.export_ctrl.on_export_rof_t)
        pp.generate_report_clicked.connect(self.export_ctrl.on_generate_report)
        pp.physical_params_changed.connect(self._on_physical_params_changed)

        # ROI: button click → enter selection mode, drag complete → apply ROI
        self.w.original_panel.select_roi_clicked.connect(self.pretune_ctrl.on_select_roi)
        self.w.original_panel.roi_selected.connect(self.pretune_ctrl.on_roi_selected)

        # R-t chart click-to-jump
        self.w.radius_chart.point_clicked.connect(self.file_ctrl.on_frame_changed)

        # R-t chart right-click: delete / refit
        self.w.radius_chart.delete_requested.connect(self._on_delete_point)
        self.w.radius_chart.refit_requested.connect(self._on_refit_point)
        self.w.radius_chart.neighbor_refit_requested.connect(
            self._on_refit_point_with_neighbors,
        )

        # Compare mode change — refresh display immediately
        self.w.compare_mode_changed.connect(self._on_compare_mode_changed)

        # Session save/load (menubar File menu)
        self.w.save_session_requested.connect(self._on_save_session)
        self.w.load_session_requested.connect(self._on_load_session)

    def _sync_ui_from_state(self) -> None:
        """Reflect loaded/persisted state in the visible controls."""
        pt = self.w.left_panel.pretune_tab
        pt.set_threshold(self._state.img_thr)
        pt.set_removing_factor(self._state.removing_factor)
        pt.set_invert_mask(self._state.invert_mask)
        pt.set_edge_flags(self._state.bubble_cross_edges)
        pt.set_filter_params({
            "gaussian_sigma": self._state.gaussian_sigma,
            "clahe_clip": self._state.clahe_clip,
            "closing_radius": self._state.closing_radius,
            "opening_radius": self._state.opening_radius,
            "detection_mode": self._state.detection_mode,
            "area_max_regions": self._state.area_max_regions,
            "gradient_boundary": self._state.gradient_boundary,
            "temporal_refine": self._state.temporal_refine,
            "edge_definition": self._state.edge_definition,
            "edge_polarity": self._state.edge_polarity,
        })
        self.pretune_ctrl.refresh_edge_labels()

        self.w.left_panel.post_processing.set_physical_params(
            self._state.fps,
            self._state.um2px,
            self._state.rmax_fit_length,
        )

    def _apply_run_lock(self, running: bool) -> None:
        """Disable every control that loads data or edits results during a run.

        Frame navigation stays available so the user can watch progress.
        """
        enabled = not running
        lp = self.w.left_panel
        for widget in (lp.image_source, lp.pretune_tab, lp.manual_tab,
                       lp.post_processing):
            widget.setEnabled(enabled)
        lp.automatic_tab.set_editing_enabled(enabled)
        self.w.save_session_action.setEnabled(enabled)
        self.w.load_session_action.setEnabled(enabled)
        self.w.radius_chart.set_editing_enabled(enabled)

    def _on_physical_params_changed(self) -> None:
        """Keep fps / scale / fit length in state so sessions, config and
        reports record the values the user actually exports with."""
        pp = self.w.left_panel.post_processing
        self._state = update_state(
            self._state,
            fps=pp.get_fps(),
            um2px=pp.get_scale(),
            rmax_fit_length=pp.get_fit_length(),
        )

    # ------------------------------------------------------------------ #
    #  Compare mode refresh
    # ------------------------------------------------------------------ #

    def _on_compare_mode_changed(self, _mode_value: str) -> None:
        """Re-render the current frame when compare mode changes."""
        if self._state.images:
            display_frame(
                self._state, self.w, self._state.image_no,
                self._set_state, self._image_cache,
            )

    # ------------------------------------------------------------------ #
    #  Result editing (delete / refit single frame)
    # ------------------------------------------------------------------ #

    def _on_delete_point(self, idx: int) -> None:
        """Delete the fitted result at frame *idx* (0-based)."""
        if self._state.radius is None or idx < 0 or idx >= len(self._state.radius):
            return

        self._state.radius[idx] = -1.0
        self._state.circle_fit_par[idx] = [np.nan, np.nan]
        if self._state.region_count is not None:
            self._state.region_count[idx] = 0
        if self._state.circle_xy is not None:
            self._state.circle_xy[idx] = None
        if self._state.fit_source is not None:
            self._state.fit_source[idx] = FitSource.NONE
        edge_support.clear_scoring(self._state, idx)

        self._refresh_chart()

        # Refresh display if we are viewing the deleted frame
        if self._state.image_no == idx:
            display_frame(
                self._state, self.w, idx, self._set_state, self._image_cache,
            )

        self.w.header.set_status(f"Deleted result for frame {idx + 1}", "#10b981")
        logger.info("Deleted result for frame %d", idx + 1)

    def _on_refit_point(self, idx: int) -> None:
        """Re-run detection + circle fitting for a single frame *idx* (0-based)."""
        if not self._state.images or idx < 0 or idx >= len(self._state.images):
            return
        if self._state.radius is None:
            return
        if self._state.detection_mode == "edge":
            self._edge_refit(idx, FitSource.REFIT)
            return

        try:
            _, _, cur_img_roi, binary_roi = normalize_source(
                current_source(self._state, idx),
                self._state.img_thr,
                self._state.gridx,
                self._state.gridy,
                gaussian_sigma=self._state.gaussian_sigma,
                clahe_clip=self._state.clahe_clip,
            )

            rf = compute_removing_factor(
                self._state.removing_factor,
                self._state.gridx,
                self._state.gridy,
            )

            processed, edge_xy = detect_bubble_with_mode(
                binary_roi,
                cur_img_roi,
                self._state.detection_mode,
                self._state.img_thr,
                list(self._state.bubble_cross_edges),
                rf,
                self._state.gridx,
                self._state.gridy,
                self._state.removing_obj_radius,
                opening_radius=self._state.opening_radius,
                closing_radius=self._state.closing_radius,
                gaussian_sigma=self._state.gaussian_sigma,
                gradient_boundary=self._state.gradient_boundary,
                area_max_regions=self._state.area_max_regions,
            )

            fit = None
            if edge_xy.shape[0] >= 3:
                r, rc, cc, count = measure_detection(
                    processed, edge_xy, self._state.detection_mode,
                    self._state.gridx, self._state.gridy,
                )
                # Reject fits that exceed max radius
                if np.isfinite(r) and r > 0 and r <= self._max_radius:
                    fit = (r, rc, cc, count)

            # A failed refit must not destroy the existing (possibly manual) result.
            previous = float(self._state.radius[idx])
            if fit is not None:
                r, rc, cc, count = fit
                self._state.radius[idx] = r
                self._state.circle_fit_par[idx] = [rc, cc]
                if self._state.region_count is not None:
                    self._state.region_count[idx] = count
                if self._state.circle_xy is not None:
                    self._state.circle_xy[idx] = edge_xy
                if self._state.fit_source is not None:
                    self._state.fit_source[idx] = FitSource.REFIT
                self._refresh_chart()
                if self._state.image_no == idx:
                    display_frame(
                        self._state, self.w, idx, self._set_state, self._image_cache,
                    )
                self.w.header.set_status(
                    f"Refit frame {idx + 1}: r = {r:.1f} px", "#10b981",
                )
            elif previous > 0:
                self.w.header.set_status(
                    f"Refit frame {idx + 1}: fit failed, kept previous result "
                    f"(r = {previous:.1f} px)", "#f59e0b",
                )
            else:
                self.w.header.set_status(
                    f"Refit frame {idx + 1}: fit failed", "#f59e0b",
                )
            logger.info("Refit frame %d: %s", idx + 1,
                        f"radius = {fit[0]:.3f}" if fit else "failed")

        except Exception as exc:
            logger.error("Refit frame %d failed: %s", idx + 1, exc)
            self.w.header.set_status(f"Refit failed: {exc}", "#ef4444")

    def _on_refit_point_with_neighbors(self, idx: int) -> None:
        """Refit one frame using neighbouring fitted frames as a circle prior."""
        if not self._state.images or idx < 0 or idx >= len(self._state.images):
            return
        if self._state.radius is None or self._state.circle_fit_par is None:
            return

        prior = self._neighbor_circle_prior(idx)
        if prior is None:
            self.w.header.set_status(
                f"No neighbouring fits for frame {idx + 1}", "#f59e0b",
            )
            return
        if self._state.detection_mode == "edge":
            self._edge_refit(idx, FitSource.NEIGHBOR, prior=prior)
            return

        try:
            _, _, cur_img_roi, _ = normalize_source(
                current_source(self._state, idx),
                self._state.img_thr,
                self._state.gridx,
                self._state.gridy,
                gaussian_sigma=self._state.gaussian_sigma,
                clahe_clip=self._state.clahe_clip,
            )
            processed, edge_xy = detect_bubble_gradient(
                cur_img_roi,
                self._state.img_thr,
                self._state.gridx,
                self._state.gridy,
                boundary_bias=self._state.gradient_boundary,
                prior_circle=prior,
                gaussian_sigma=(
                    self._state.gaussian_sigma
                    if self._state.gaussian_sigma > 0 else 1.2
                ),
            )

            if edge_xy.shape[0] < 3:
                self.w.header.set_status(
                    f"Neighbor refit frame {idx + 1}: no boundary", "#f59e0b",
                )
                return

            rc, cc, r = circle_fit_taubin(edge_xy)
            if not np.isfinite(r) or r <= 0 or r > self._max_radius:
                self.w.header.set_status(
                    f"Neighbor refit frame {idx + 1}: invalid radius", "#f59e0b",
                )
                return

            self._state.radius[idx] = r
            self._state.circle_fit_par[idx] = [rc, cc]
            if self._state.region_count is not None:
                self._state.region_count[idx] = 0
            if self._state.circle_xy is not None:
                self._state.circle_xy[idx] = edge_xy
            if self._state.fit_source is not None:
                self._state.fit_source[idx] = FitSource.NEIGHBOR

            self._refresh_chart()
            if self._state.image_no == idx:
                display_frame(
                    self._state, self.w, idx, self._set_state, self._image_cache,
                )
            self.w.header.set_status(
                f"Neighbor refit frame {idx + 1}: r = {r:.1f} px", "#10b981",
            )
            logger.info("Neighbor refit frame %d: radius = %.3f", idx + 1, r)

        except Exception as exc:
            logger.error("Neighbor refit frame %d failed: %s", idx + 1, exc)
            self.w.header.set_status(f"Neighbor refit failed: {exc}", "#ef4444")

    def _edge_refit(self, idx: int, source: FitSource, *,
                    prior: tuple[float, float, float] | None = None) -> None:
        """Refit one frame with the Edge engine; a rejected fit keeps the old result."""
        try:
            fit = edge_support.fit_frame(self._state, idx, self._max_radius, prior=prior)
        except Exception as exc:
            logger.error("Edge refit frame %d failed: %s", idx + 1, exc)
            self.w.header.set_status(f"Refit failed: {exc}", "#ef4444")
            return
        if fit.radius > 0:
            edge_support.store_fit(self._state, idx, fit, source)
            self._refresh_chart()
            if self._state.image_no == idx:
                display_frame(self._state, self.w, idx, self._set_state, self._image_cache)
        elif self._state.fit_confidence is not None:
            self._state.fit_confidence[idx] = fit.confidence
        text, colour = edge_support.status_text(idx, fit)
        if fit.radius <= 0 and self._state.radius[idx] > 0:
            text += f"; kept previous result (r = {self._state.radius[idx]:.1f} px)"
        self.w.header.set_status(text, colour)

    def _neighbor_circle_prior(
        self,
        idx: int,
    ) -> tuple[float, float, float] | None:
        """Predict a circle for *idx* from nearest valid neighbouring frames."""
        radius = self._state.radius
        centers = self._state.circle_fit_par
        n = len(radius)

        def valid(i: int) -> bool:
            return (
                0 <= i < n
                and radius[i] > 0
                and np.isfinite(radius[i])
                and np.isfinite(centers[i]).all()
            )

        prev_idx = next((i for i in range(idx - 1, -1, -1) if valid(i)), None)
        next_idx = next((i for i in range(idx + 1, n) if valid(i)), None)

        if prev_idx is None and next_idx is None:
            return None
        if prev_idx is None:
            return (
                float(centers[next_idx, 0]),
                float(centers[next_idx, 1]),
                float(radius[next_idx]),
            )
        if next_idx is None:
            return (
                float(centers[prev_idx, 0]),
                float(centers[prev_idx, 1]),
                float(radius[prev_idx]),
            )

        alpha = (idx - prev_idx) / (next_idx - prev_idx)
        rc = (1 - alpha) * centers[prev_idx, 0] + alpha * centers[next_idx, 0]
        cc = (1 - alpha) * centers[prev_idx, 1] + alpha * centers[next_idx, 1]
        r = (1 - alpha) * radius[prev_idx] + alpha * radius[next_idx]
        return float(rc), float(cc), float(r)

    def _refresh_chart(self) -> None:
        """Refresh the R-t chart with quality-based coloring."""
        refresh_chart(self._state, self.w)

    # ------------------------------------------------------------------ #
    #  Session save/load
    # ------------------------------------------------------------------ #

    def save_session_file(self, path: str) -> bool:
        return self.session_ctrl.save(path)

    def load_session_file(self, path: str) -> bool:
        return self.session_ctrl.load(path)

    def _on_save_session(self) -> None:
        """Prompt the user for a path and save the current session."""
        default_dir = self._state.folder_path or os.path.expanduser("~")
        path, _ = QFileDialog.getSaveFileName(
            self.w, "Save Session", default_dir,
            "BubbleTrack Session (*.brt);;All Files (*)",
        )
        if path:
            self.save_session_file(path)

    def _on_load_session(self) -> None:
        """Prompt the user for a .brt file and restore the session."""
        default_dir = self._state.folder_path or os.path.expanduser("~")
        path, _ = QFileDialog.getOpenFileName(
            self.w, "Load Session", default_dir,
            "BubbleTrack Session (*.brt);;All Files (*)",
        )
        if path:
            self.load_session_file(path)

    def _locate_moved_folder(self, original: str) -> str:
        """Ask the user where a session's image folder went; '' if cancelled."""
        QMessageBox.information(
            self.w, "Image folder not found",
            f"The session's image folder no longer exists:\n{original}\n\n"
            "Please select its new location.",
        )
        return QFileDialog.getExistingDirectory(
            self.w, "Locate the image folder", os.path.dirname(original),
        )

    def _after_session_restore(self) -> None:
        """Bring every view in line with a freshly restored state."""
        self._sync_ui_from_state()
        self.w.original_panel.set_roi_text(self._state.gridx, self._state.gridy)
        if self._state.images:
            self.file_ctrl.on_frame_changed(self._state.image_no)
        self._refresh_chart()

    # ------------------------------------------------------------------ #
    #  Closing the window
    # ------------------------------------------------------------------ #

    def _ask_yes_no(self, title: str, text: str) -> bool:
        answer = QMessageBox.question(
            self.w, title, text,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return answer == QMessageBox.StandardButton.Yes

    def _confirm_close(self) -> bool:
        """Return False to keep the window open.

        A running fit needs confirmation.  Results are always kept in
        ~/.bubbletrack/recovery/last_session.brt so closing never loses them.
        """
        if self.auto_ctrl.is_busy():
            if not self._ask_yes_no(
                "Fit in progress",
                "A fit is still running. Stop it and quit?",
            ):
                return False
            self.auto_ctrl.stop_and_wait()
        self.session_ctrl.save_last_session()
        return True
