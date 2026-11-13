"""PretuneController — threshold, ROI, edge flags, single-frame fitting."""

from __future__ import annotations

import logging

import numpy as np

from bubbletrack.controller.base import BaseController
from bubbletrack.controller.display_mixin import (
    EQUAL_AREA_CIRCLE,
    FITTED_CIRCLE,
    current_source,
    display_frame,
    redraw_original,
    refresh_chart,
    show_engine_view,
)
from bubbletrack.event_bus import EventBus
from bubbletrack.model.cache import ImageCache
from bubbletrack.model.conventions import clamp_roi
from bubbletrack.model.detection import detect_bubble_with_mode, measure_detection
from bubbletrack.controller import edge_support, suggestion_view
from bubbletrack.controller.edge_setup import EdgeSetup
from bubbletrack.model.frame_source import normalize_source
from bubbletrack.model.provenance import FitSource
from bubbletrack.model.quality import compute_fit_quality
from bubbletrack.model.removing_factor import compute_removing_factor

logger = logging.getLogger(__name__)


class PretuneController(BaseController):
    """Manages pre-tune parameters and single-frame fitting."""

    def __init__(self, bus: EventBus, get_state, set_state, window,
                 display_timer, preview_timer, get_max_radius,
                 cache: ImageCache | None = None) -> None:
        super().__init__(bus, get_state, set_state, window)
        self._display_timer = display_timer
        self._preview_timer = preview_timer
        self._get_max_radius = get_max_radius
        self._cache = cache

        # Preview detection cache: skip load_and_normalize when only RF changes
        self._last_binary_roi = None
        self._last_binary_params: tuple | None = None

        # Auto-tune worker (lazy import to avoid circular dependency)
        self._autotune_worker = None
        self.edge_setup = EdgeSetup(bus, get_state, set_state, window)

    # -- public handlers -------------------------------------------------- #

    def on_threshold_changed(self, v: float) -> None:
        self._update(img_thr=v)
        self._invalidate_binary_cache()
        if self._cache is not None:
            self._cache.invalidate()
        self._display_timer.start()  # debounced

    def on_invert_mask_changed(self, enabled: bool) -> None:
        """Toggle the visual binary polarity swap.

        ``invert_mask`` is a pure post-detection display flag — it does not
        change the cached binary, the detected edge points, or the fitted
        radius.  We only need to re-render the binary panel.
        """
        self._update(invert_mask=enabled)
        self._display_timer.start()  # debounced

    def on_removing_factor_changed(self, v: int) -> None:
        self._update(removing_factor=v)
        self._preview_timer.start()  # debounced

    def on_filters_changed(self) -> None:
        p = self.w.left_panel.measure_panel.get_filter_params()
        method_changed = p["detection_mode"] != self.state.detection_mode
        self._update(
            gaussian_sigma=p["gaussian_sigma"],
            clahe_clip=p["clahe_clip"],
            closing_radius=p["closing_radius"],
            opening_radius=p["opening_radius"],
            detection_mode=p["detection_mode"],
            area_max_regions=p["area_max_regions"],
            gradient_boundary=p["gradient_boundary"],
            temporal_refine=p["temporal_refine"],
            edge_definition=p["edge_definition"],
            edge_polarity=p["edge_polarity"],
        )
        self.refresh_edge_labels()
        self._invalidate_binary_cache()
        if self._cache is not None:
            self._cache.invalidate()
        if method_changed:
            self._display_timer.start()  # the new method's view, titles and overlays
        else:
            self._preview_timer.start()  # debounced

    def _invalidate_binary_cache(self) -> None:
        """Clear the preview binary ROI cache."""
        self._last_binary_roi = None
        self._last_binary_params = None

    def on_edges_changed(self) -> None:
        self._update(
            bubble_cross_edges=tuple(self.w.left_panel.measure_panel.get_edge_flags()),
        )
        self.preview_detection()

    def preview_detection(self) -> None:
        """Run full detect_bubble pipeline for preview.

        Mirrors ``display_frame``'s rendering path so that toggling between
        threshold and removing-factor sliders converges to the same binary
        for the same ``(state, frame)`` — see tests/test_display_consistency.

        When only removing_factor changes, the cached binary_roi is reused
        to skip the expensive load_and_normalize step.
        """
        if not self.state.images or self.state.detection_mode == "edge":
            return  # Edge mode has no threshold preview
        idx = self.state.image_no
        rf = compute_removing_factor(
            self.state.removing_factor, self.state.gridx, self.state.gridy,
        )
        try:
            # Check if binary ROI can be reused (only RF or edges changed).
            current_params = (
                self.state.images[idx],
                self.state.img_thr,
                self.state.gridx,
                self.state.gridy,
                self.state.gaussian_sigma,
                self.state.clahe_clip,
                self.state.detection_mode,
                self.state.gradient_boundary,
            )
            if (current_params == self._last_binary_params
                    and self._last_binary_roi is not None):
                cur_img_roi, binary_roi = self._last_binary_roi
            else:
                _, _, cur_img_roi, binary_roi = normalize_source(
                    current_source(self.state, idx), self.state.img_thr,
                    self.state.gridx, self.state.gridy,
                    gaussian_sigma=self.state.gaussian_sigma,
                    clahe_clip=self.state.clahe_clip,
                )
                self._last_binary_roi = (cur_img_roi, binary_roi)
                self._last_binary_params = current_params

            processed, _ = detect_bubble_with_mode(
                binary_roi, cur_img_roi, self.state.detection_mode,
                self.state.img_thr, list(self.state.bubble_cross_edges), rf,
                self.state.gridx, self.state.gridy,
                self.state.removing_obj_radius,
                opening_radius=self.state.opening_radius,
                closing_radius=self.state.closing_radius,
                gaussian_sigma=self.state.gaussian_sigma,
                gradient_boundary=self.state.gradient_boundary,
                area_max_regions=self.state.area_max_regions,
            )
            # Apply invert_mask as the last visual step (matches display_frame).
            bin_display = ~processed if self.state.invert_mask else processed
            self.w.binary_panel.set_image(bin_display)
        except Exception:
            pass

    def on_select_roi(self) -> None:
        self.w.original_panel.set_mode("roi")
        self.w.header.set_status("Drag on image to select ROI", "#f59e0b")

    def on_roi_selected(self, r0: int, r1: int, c0: int, c1: int) -> None:
        # Clamp to actual image dimensions
        if self.state.cur_img is not None:
            h, w = self.state.cur_img.shape[:2]
            (r0, r1), (c0, c1) = clamp_roi((r0, r1), (c0, c1), h, w)
        self._update(gridx=(r0, r1), gridy=(c0, c1))
        self._invalidate_binary_cache()
        if self._cache is not None:
            self._cache.invalidate()
        self.w.original_panel.set_roi_text((r0, r1), (c0, c1))
        self.w.header.set_status("ROI selected", "#10b981")
        display_frame(
            self.state, self.w, self.state.image_no, self._set_state,
            self._cache,
        )

    # -- Edge engine: reference frame and bubble position (see edge_setup) - #

    def refresh_edge_labels(self) -> None:
        self.edge_setup.refresh_labels()

    def on_use_current_as_reference(self) -> None:
        self.edge_setup.use_current()

    def on_auto_reference(self) -> None:
        self.edge_setup.use_auto()

    def on_pick_seed(self) -> None:
        self.edge_setup.pick()

    def on_point_clicked(self, x: float, y: float) -> None:
        self.edge_setup.on_point_clicked(x, y)

    def on_clear_seed(self) -> None:
        self.edge_setup.clear()

    def _fit_current_edge(self, idx: int) -> None:
        try:
            fit = edge_support.fit_frame(self.state, idx, self._get_max_radius())
        except Exception as exc:
            logger.exception("Edge fit failed for frame %d", idx)
            self.w.header.set_status(f"Error: {exc}", "#ef4444")
            return
        edge_support.store_fit(self.state, idx, fit, FitSource.REFIT)
        show_engine_view(self.state, self.w, idx)
        redraw_original(self.state, self.w)
        if fit.radius > 0:
            self.w.original_panel.draw_circle(fit.rc, fit.cc, fit.radius, "#6366f1",
                                              legend=EQUAL_AREA_CIRCLE)
            self.w.original_panel.draw_points(fit.edge_xy, "#ef4444", 2.0)
        else:  # rejected: its guess, if any, dashed
            suggestion_view.draw_guess(self.state, self.w.original_panel, idx)
        refresh_chart(self.state, self.w)
        self.w.left_panel.measure_panel.hide_quality()
        self.w.header.set_status(*edge_support.status_text(idx, fit))
        self._results_changed()

    def on_pretune_fit(self) -> None:
        """Detect + fit circle for the current single frame."""
        if not self.state.images:
            return

        self.w.header.set_status("Fitting...", "#f59e0b")
        idx = self.state.image_no
        if self.state.detection_mode == "edge":
            self._fit_current_edge(idx)
            return
        rf = compute_removing_factor(
            self.state.removing_factor, self.state.gridx, self.state.gridy,
        )

        try:
            _, _, cur_img_roi, binary_roi = normalize_source(
                current_source(self.state, idx), self.state.img_thr,
                self.state.gridx, self.state.gridy,
                gaussian_sigma=self.state.gaussian_sigma,
                clahe_clip=self.state.clahe_clip,
            )
            processed, edge_xy = detect_bubble_with_mode(
                binary_roi, cur_img_roi, self.state.detection_mode,
                self.state.img_thr, list(self.state.bubble_cross_edges), rf,
                self.state.gridx, self.state.gridy,
                self.state.removing_obj_radius,
                opening_radius=self.state.opening_radius,
                closing_radius=self.state.closing_radius,
                gaussian_sigma=self.state.gaussian_sigma,
                gradient_boundary=self.state.gradient_boundary,
                area_max_regions=self.state.area_max_regions,
            )

            # Apply invert_mask as the last visual step (matches display_frame).
            bin_display = ~processed if self.state.invert_mask else processed
            self.w.binary_panel.set_image(bin_display)

            if edge_xy.shape[0] >= 3:
                radius, rc, cc, count = measure_detection(
                    processed, edge_xy, self.state.detection_mode,
                    self.state.gridx, self.state.gridy,
                )
                if np.isnan(radius) or radius > self._get_max_radius():
                    logger.error("Fit failed for frame %d", idx)
                    self.w.header.set_status(
                        f"Radius outlier ({radius:.0f} px), skipped", "#f59e0b",
                    )
                    self.w.left_panel.measure_panel.hide_quality()
                else:
                    logger.info("Fit frame %d: radius=%.2f", idx, radius)
                    self.state.radius[idx] = radius
                    self.state.circle_fit_par[idx] = [rc, cc]
                    self.state.circle_xy[idx] = edge_xy
                    if self.state.region_count is not None:
                        self.state.region_count[idx] = count
                    if self.state.fit_source is not None:
                        self.state.fit_source[idx] = FitSource.REFIT

                    # Clear old overlays before drawing new results
                    redraw_original(self.state, self.w)
                    if self.state.detection_mode != "area":
                        self.w.original_panel.draw_circle(rc, cc, radius, "#6366f1",
                                                          legend=FITTED_CIRCLE)
                    self.w.original_panel.draw_points(edge_xy, "#ef4444", 2.0)
                    refresh_chart(self.state, self.w)
                    self._results_changed()
                    label = (f"Area-equivalent R = {radius:.1f} px; regions = {count}"
                             if self.state.detection_mode == "area"
                             else f"R = {radius:.1f} px")
                    self.w.header.set_status(label, "#10b981")

                    # Show quality metrics
                    roi_h = self.state.gridx[1] - self.state.gridx[0] + 1
                    roi_w = self.state.gridy[1] - self.state.gridy[0] + 1
                    if self.state.detection_mode == "area":
                        self.w.left_panel.measure_panel.show_area_result(
                            int(processed.sum()), count,
                        )
                    else:
                        q = compute_fit_quality(
                            edge_xy, rc, cc, radius, min(roi_h, roi_w),
                        )
                        self.w.left_panel.measure_panel.show_quality(
                            q.n_edge_points, q.rms_residual, int(q.score * 100),
                        )
            else:
                self.w.header.set_status("Too few edge points", "#ef4444")
                self.w.left_panel.measure_panel.hide_quality()
        except Exception as exc:
            self.w.header.set_status(f"Error: {exc}", "#ef4444")
            self.w.left_panel.measure_panel.hide_quality()

    # -- Auto-tune -------------------------------------------------------- #

    def on_autotune(self) -> None:
        """Launch auto-tune grid search in background thread."""
        from bubbletrack.controller.autotune_worker import AutoTuneWorker

        if not self.state.images:
            return
        if self.state.detection_mode != "intensity":
            self.w.header.set_status(
                "Auto Tune is for intensity-mask mode; tune Gradient manually",
                "#f59e0b",
            )
            return
        # Cancel any still-running worker before starting a new one
        if self._autotune_worker is not None:
            if self._autotune_worker.isRunning():
                self._autotune_worker.cancel()
                self._autotune_worker.wait()
            self._autotune_worker = None

        idx = self.state.image_no
        self.w.header.set_status("Auto-tuning...", "#f59e0b")
        self.w.left_panel.measure_panel.set_autotune_enabled(False)

        self._autotune_worker = AutoTuneWorker(
            current_source(self.state, idx),
            self.state.gridx,
            self.state.gridy,
            self.state.bubble_cross_edges,
            gaussian_sigma=self.state.gaussian_sigma,
            clahe_clip=self.state.clahe_clip,
            opening_radius=self.state.opening_radius,
            closing_radius=self.state.closing_radius,
            max_radius=self._get_max_radius(),
        )
        self._autotune_worker.progress.connect(self._on_autotune_progress)
        self._autotune_worker.result_ready.connect(self._on_autotune_finished)
        self._autotune_worker.start()

    def _on_autotune_progress(self, current: int, total: int) -> None:
        pct = int(100 * current / total) if total > 0 else 0
        self.w.header.set_status(f"Auto-tuning... {pct}%", "#f59e0b")

    def _on_autotune_finished(self, result) -> None:
        self.w.left_panel.measure_panel.set_autotune_enabled(True)
        # Wait for the thread to fully exit before dropping the reference,
        # otherwise GC may destroy the QThread while it is still running.
        if self._autotune_worker is not None:
            self._autotune_worker.wait()
        self._autotune_worker = None

        if result is None:
            self.w.header.set_status("Auto-tune: no valid fit found", "#ef4444")
            return

        # Apply best parameters to state + UI
        pt = self.w.left_panel.measure_panel
        pt.set_threshold(result.threshold)
        pt.set_removing_factor(result.removing_factor)

        self._update(
            img_thr=result.threshold,
            removing_factor=result.removing_factor,
        )
        self._invalidate_binary_cache()
        if self._cache is not None:
            self._cache.invalidate()

        # Show quality metrics
        q = result.quality
        pt.show_quality(q.n_edge_points, q.rms_residual, int(q.score * 100))

        # Run full fit (includes display_frame, redraw, and chart refresh)
        self.on_pretune_fit()

        # Overwrite per-frame status with autotune summary
        self.w.header.set_status(
            f"Auto-tuned: thr={result.threshold:.2f}, RF={result.removing_factor}"
            f" \u2014 confidence {int(q.score * 100)}%"
            f" ({result.candidates_evaluated} tested)",
            "#10b981",
        )
