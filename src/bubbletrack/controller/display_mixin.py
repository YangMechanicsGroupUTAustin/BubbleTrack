"""Shared display helpers used by multiple sub-controllers.

These are standalone functions that accept (state, window) to avoid
diamond inheritance issues.  Sub-controllers call them by passing
``self.state`` and ``self.w``.
"""

from __future__ import annotations

import numpy as np

from bubbletrack.model.cache import ImageCache
from bubbletrack.model.circle_fit import circle_fit_taubin
from bubbletrack.model.conventions import frame_to_display, roi_to_slice
from bubbletrack.model.detection import detect_bubble_with_mode, measure_detection
from bubbletrack.model.image_io import load_and_normalize, normalize_frame
from bubbletrack.model.quality import compute_all_quality_scores
from bubbletrack.model.removing_factor import compute_removing_factor
from bubbletrack.model.state import AppState, update_state
from bubbletrack.ui.image_compare import CompareMode, create_overlay, create_wipe


def _build_cache_key(filepath: str, state: AppState) -> str:
    """Build a cache key from filepath and processing parameters.

    ``invert_mask`` is intentionally NOT part of the key: it is a pure
    visual swap applied at display time (after detect_bubble), so the
    cached binary is identical regardless of the toggle state.
    """
    return (
        f"{filepath}:{state.img_thr}:{state.gridx}:{state.gridy}"
        f":{state.gaussian_sigma}:{state.clahe_clip}:{state.detection_mode}"
        f":{state.gradient_boundary}:{state.area_max_regions}"
    )


def display_frame(
    state: AppState,
    window,
    idx: int,
    set_state,
    cache: ImageCache | None = None,
) -> None:
    """Load, process and display frame *idx* on both image panels.

    Parameters
    ----------
    state : AppState
        Current application state.
    window : MainWindow
        Reference to the Qt main window (for UI updates).
    idx : int
        0-based frame index to display.
    set_state : callable
        Callback to persist the updated state (with cur_img etc.).
    cache : ImageCache | None
        Optional LRU cache for loaded images.
    """
    if not state.images:
        return
    try:
        # Check cache before loading from disk / video
        cache_key = _build_cache_key(state.images[idx], state)
        cached = cache.get(cache_key) if cache is not None else None

        if cached is not None:
            cur_img, _, cur_img_roi, binary_roi = cached
        elif state.video_reader is not None:
            # Video mode: extract frame then normalise
            raw = state.video_reader.read_frame(idx)
            cur_img, _, cur_img_roi, binary_roi = normalize_frame(
                raw,
                state.img_thr,
                state.gridx,
                state.gridy,
                gaussian_sigma=state.gaussian_sigma,
                clahe_clip=state.clahe_clip,
            )
            if cache is not None:
                cache.put(
                    cache_key,
                    (cur_img, None, cur_img_roi, binary_roi),
                )
        else:
            cur_img, _, cur_img_roi, binary_roi = load_and_normalize(
                state.images[idx],
                state.img_thr,
                state.gridx,
                state.gridy,
                gaussian_sigma=state.gaussian_sigma,
                clahe_clip=state.clahe_clip,
            )
            if cache is not None:
                cache.put(
                    cache_key,
                    (cur_img, None, cur_img_roi, binary_roi),
                )
        state = update_state(
            state, cur_img=cur_img, cur_img_binary_roi=binary_roi,
        )
        set_state(state)

        # Compute bin_display by running the full detect_bubble pipeline
        # for ALL frames (fitted or not). Historical bug: the unfitted
        # branch returned ~binary_roi without applying Removing Factor or
        # morphology, so the binary panel was path-dependent — dragging
        # threshold off and back gave a different visual than dragging RF
        # off and back, even though both ended at the same (a, b) state.
        rf = compute_removing_factor(
            state.removing_factor,
            state.gridx, state.gridy,
        )
        processed, edge_xy = detect_bubble_with_mode(
            binary_roi, cur_img_roi, state.detection_mode, state.img_thr,
            list(state.bubble_cross_edges), rf,
            state.gridx, state.gridy,
            state.removing_obj_radius,
            opening_radius=state.opening_radius,
            closing_radius=state.closing_radius,
            gaussian_sigma=state.gaussian_sigma,
            gradient_boundary=state.gradient_boundary,
            area_max_regions=state.area_max_regions,
        )
        # Apply invert_mask as the final visual step — pure bit flip on the
        # detect_bubble output, so the binary panel mirrors black<->white
        # without affecting the detected edge points or radius.
        bin_display = ~processed if state.invert_mask else processed

        # Lazy recovery: when radius is known but circle params are
        # missing (e.g. loaded from batch results), re-fit from the
        # edge points we just detected.  Cost: one circle_fit_taubin
        # per displayed frame, only when needed.
        if (state.radius is not None and state.radius[idx] > 0
                and state.circle_fit_par is not None
                and not np.isfinite(state.circle_fit_par[idx]).all()
                and edge_xy.shape[0] >= 3):
            _, rc, cc, _ = measure_detection(
                processed, edge_xy, state.detection_mode, state.gridx, state.gridy,
            )
            state.circle_fit_par[idx] = [rc, cc]
            if state.circle_xy is not None:
                state.circle_xy[idx] = edge_xy

        # Render according to compare mode
        mode = window.compare_mode

        if mode in (CompareMode.OVERLAY, CompareMode.WIPE):
            # bin_display is ROI-sized; embed into full-image canvas
            # so the composite matches cur_img dimensions.
            h, w = cur_img.shape[:2]
            full_bin = np.zeros((h, w), dtype=bin_display.dtype)
            rs, cs = roi_to_slice(state.gridx, state.gridy)
            # Clamp slices to actual image bounds to avoid index errors
            r_end = min(rs.stop, h)
            c_end = min(cs.stop, w)
            roi_h = r_end - rs.start
            roi_w = c_end - cs.start
            full_bin[rs.start:r_end, cs.start:c_end] = (
                bin_display[:roi_h, :roi_w]
            )
            if mode is CompareMode.OVERLAY:
                composite = create_overlay(cur_img, full_bin)
            else:
                composite = create_wipe(cur_img, full_bin)
            window.original_panel.set_image_rgb(composite)
        else:
            # Side-by-side (default)
            window.original_panel.set_image(cur_img)
            window.binary_panel.set_image(bin_display)

        # Draw ROI rectangle on original panel
        window.original_panel.draw_roi_rect(state.gridx, state.gridy)

        # Draw existing circle fit if available
        if (state.radius is not None
                and state.radius[idx] > 0
                and state.circle_fit_par is not None):
            rc = state.circle_fit_par[idx, 0]
            cc = state.circle_fit_par[idx, 1]
            r = state.radius[idx]
            is_area = (state.region_count is not None
                       and state.region_count[idx] > 0)
            if not is_area:
                window.original_panel.draw_circle(rc, cc, r, "#6366f1")
            if state.circle_xy and state.circle_xy[idx] is not None:
                window.original_panel.draw_points(
                    state.circle_xy[idx], "#ef4444", 2.0,
                )

        window.status_bar.update_frame(
            frame_to_display(idx), state.total_frames,
        )
        window.status_bar.update_roi(state.gridx, state.gridy)
        if (state.region_count is not None and state.region_count[idx] > 0):
            window.header.set_status(
                f"Area-equivalent R = {state.radius[idx]:.1f} px; "
                f"regions = {state.region_count[idx]}", "#10b981",
            )
    except Exception as exc:
        window.header.set_status(f"Error: {exc}", "#ef4444")


def refresh_chart(state: AppState, window) -> np.ndarray | None:
    """Redraw R-t chart with per-frame quality coloring.

    Returns
    -------
    np.ndarray or None
        Quality scores with full-frame indexing (length == len(state.radius)).
        Frames with radius <= 0 have score 0.0.  None if no radius data.
    """
    if state.radius is not None:
        frames = np.arange(1, len(state.radius) + 1)
        scores = compute_all_quality_scores(
            state.radius, state.circle_fit_par, state.circle_xy,
            state.gridx, state.gridy,
        )
        if state.region_count is not None:
            area_mask = (state.region_count > 0) & (state.radius > 0)
            scores[area_mask] = np.nan
        window.radius_chart.plot_all(frames, state.radius, scores)
        return scores
    return None


def redraw_original(state: AppState, window) -> None:
    """Clear overlays on original panel and re-draw ROI rect."""
    window.original_panel.clear_overlays()
    window.original_panel.draw_roi_rect(state.gridx, state.gridy)
