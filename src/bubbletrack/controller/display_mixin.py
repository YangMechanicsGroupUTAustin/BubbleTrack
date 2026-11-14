"""Shared display helpers used by multiple sub-controllers.

These are standalone functions that accept (state, window) to avoid
diamond inheritance issues.  Sub-controllers call them by passing
``self.state`` and ``self.w``.
"""

from __future__ import annotations

import logging

import numpy as np

from bubbletrack.controller import suggestion_view
from bubbletrack.model.cache import ImageCache
from bubbletrack.model.circle_fit import circle_fit_taubin
from bubbletrack.model.conventions import frame_to_display, roi_to_slice
from bubbletrack.model.detection import detect_bubble_with_mode, measure_detection
from bubbletrack.model.export import locate_rmax
from bubbletrack.model.image_io import load_and_normalize, normalize_frame
from bubbletrack.model.quality import compute_all_quality_scores
from bubbletrack.model.provenance import AREA_RADIUS_FLAG
from bubbletrack.model.removing_factor import compute_removing_factor
from bubbletrack.model.state import AppState, update_state
from bubbletrack.ui.chart_legend import CONFIDENCE, FIT_QUALITY
from bubbletrack.ui.image_compare import CompareMode, create_overlay, create_wipe
from bubbletrack.ui.overlay_legend import (
    CIRCLE_COLOUR, OUTLINE_COLOUR, PAINT_BUBBLE_COLOUR, LegendEntry,
)

logger = logging.getLogger(__name__)


def current_source(state: AppState, idx: int):
    """Where frame *idx* comes from: its file path, or the decoded video frame.

    Pass the result to ``frame_source.normalize_source``.  Video frames are
    identified by placeholder ids in ``state.images`` that are not paths.
    """
    if state.video_reader is not None:
        return state.video_reader.read_frame(idx)
    return state.images[idx]


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
        view = None
        if state.detection_mode == "edge":
            # Auto mode has no threshold mask: show what the engine measures
            # (frame / reference); its half-way level stands in for a mask in
            # the overlay and wipe views.
            view = engine_view_or_none(state, idx)
            processed = (view < 0.5) if view is not None else np.zeros_like(binary_roi, dtype=bool)
            edge_xy = np.empty((0, 2))
        else:
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
            if mode is CompareMode.OVERLAY:
                window.original_panel.add_legend_entry(MASK_TINT_KEY)
        else:
            # Image only (default) or side by side: the raw frame on the left;
            # the mask panel is filled either way (it is hidden in IMAGE mode)
            # so switching modes never shows a stale mask.
            window.original_panel.set_image(cur_img)
            if view is not None:
                show_engine_view(state, window, idx, view)
            else:
                window.binary_panel.set_title(MASK_TITLE)
                window.binary_panel.set_image(bin_display)

        window.original_panel.draw_roi_rect(state.gridx, state.gridy)
        draw_fit_overlay(state, window.original_panel, idx)
        window.radius_chart.set_current_frame(idx)
        window.radius_chart.set_score_name(score_name(state))  # also before any fit

        window.status_bar.update_frame(
            frame_to_display(idx), state.total_frames,
        )
        window.status_bar.update_roi(state.gridx, state.gridy)
        run_display_hooks(window, state, idx)
        if (state.region_count is not None and state.region_count[idx] > 0):
            window.header.set_status(
                f"Area-equivalent R = {state.radius[idx]:.1f} px; "
                f"regions = {state.region_count[idx]}", "#10b981",
            )
        suggestion_view.announce_guess(state, window, idx)
    except Exception as exc:
        window.header.set_status(f"Error: {exc}", "#ef4444")


def run_display_hooks(window, state: AppState, idx: int) -> None:
    """Call ``window.frame_display_hooks`` (e.g. ML assist's teaching overlays)
    after frame *idx* was drawn; a failing hook is logged, never fatal."""
    for hook in tuple(getattr(window, "frame_display_hooks", ())):
        try:
            hook(state, idx)
        except Exception:
            logger.exception("Display hook failed for frame %d", idx + 1)


MASK_TITLE = "Binary Mask"
ENGINE_VIEW_TITLE = "Auto view"  # short: it shares the row with the image at 1280 px
ENGINE_VIEW_TIP = ("What Auto measures: this frame \u00f7 the reference frame (no bubble).\n"
                   "An even grey background means the reference matches this frame;\n"
                   "blotches or texture mean it does not (moved sample, changed lighting).")
_VIEW_WHITE = 1.25  # view value shown as white: background (1.0) is light grey
MASK_TINT_KEY = LegendEntry("mask", PAINT_BUBBLE_COLOUR, "fill")  # the overlay view's green tint
EQUAL_AREA_CIRCLE = "equal-area circle"  # Auto, "Bubble may be non-round": R = sqrt(area / pi)
FITTED_CIRCLE = "fitted circle"  # Auto's default, classic modes and clicked points: a circle fit


def engine_view_or_none(state: AppState, idx: int) -> np.ndarray | None:
    """Frame *idx*'s ROI as the Auto engine sees it, or None if it cannot be read."""
    from bubbletrack.controller import edge_support  # edge_support imports this module

    try:
        return edge_support.view_frame(state, idx)
    except Exception as exc:
        logger.warning("Auto view of frame %d unavailable: %s", idx + 1, exc)
        return None


def show_engine_view(state: AppState, window, idx: int, view: np.ndarray | None = None) -> None:
    """Right panel in Auto mode: the engine's view with the measured outline.

    A flat, even background means the reference frame matches this frame;
    blotches or texture mean it does not (moved sample, changed lighting).
    """
    panel = window.binary_panel
    panel.set_title(ENGINE_VIEW_TITLE, ENGINE_VIEW_TIP)
    if view is None:
        view = engine_view_or_none(state, idx)
    if view is None:
        return
    panel.set_image(np.clip(view / _VIEW_WHITE, 0.0, 1.0).astype(np.float32))
    xy = state.circle_xy[idx] if state.circle_xy is not None else None
    origin = np.array([state.gridx[0] - 1, state.gridy[0] - 1], dtype=float)
    if xy is not None and state.radius is not None and state.radius[idx] > 0:
        panel.draw_fit_outline(np.asarray(xy, dtype=float) - origin, OUTLINE_COLOUR)
    else:  # no result: Auto's rejected guess, if any, dashed
        suggestion_view.draw_guess(state, panel, idx, offset=origin)


def circle_legend(state: AppState, idx: int) -> str:
    """What frame *idx*'s drawn circle is: the equal-area circle of an outline measured with
    "Bubble may be non-round" (flag ``area_radius``), else a fitted circle."""
    flags = state.fit_flags[idx] if state.fit_flags is not None and idx < len(state.fit_flags) else ()
    return EQUAL_AREA_CIRCLE if AREA_RADIUS_FLAG in (flags or ()) else FITTED_CIRCLE


def draw_fit_overlay(state: AppState, panel, idx: int) -> None:
    """Draw frame *idx*'s fitted circle and its boundary on an image panel.

    The boundary is a closed polyline when ``circle_xy`` holds an ordered
    contour (Edge engine), otherwise the detected edge points.  Area-mode
    results have no meaningful circle, so only their boundary is drawn.
    A frame without a result shows Auto's rejected guess, if any, dashed.
    """
    if state.radius is None or idx >= len(state.radius) or state.radius[idx] <= 0:
        suggestion_view.draw_guess(state, panel, idx)
        return
    is_area = state.region_count is not None and state.region_count[idx] > 0
    if not is_area and state.circle_fit_par is not None \
            and np.isfinite(state.circle_fit_par[idx]).all():
        rc, cc = state.circle_fit_par[idx]
        panel.draw_circle(rc, cc, state.radius[idx], CIRCLE_COLOUR,
                          legend=circle_legend(state, idx))
    if state.circle_xy is not None and state.circle_xy[idx] is not None:
        panel.draw_fit_outline(state.circle_xy[idx], OUTLINE_COLOUR)


def score_name(state: AppState) -> str:
    """What the chart's marker colours mean in *state*'s method, results or not:
    Auto's confidence, or the classic methods' geometric fit quality."""
    return CONFIDENCE if state.detection_mode == "edge" else FIT_QUALITY


def chart_scores(state: AppState) -> tuple[np.ndarray, str] | None:
    """Per-frame marker scores for the R(t) chart and the ramp's name.

    Edge mode uses the engine's own confidence (the validated measure; the
    geometric quality score rates wrong fits highly); other modes use the
    geometric fit quality, NaN for area-mode frames.  Frames with
    radius <= 0 score 0.  None without results.
    """
    if state.radius is None:
        return None
    if state.detection_mode == "edge" and state.fit_confidence is not None:
        return np.where(state.radius > 0, state.fit_confidence, 0.0), CONFIDENCE
    scores = compute_all_quality_scores(
        state.radius, state.circle_fit_par, state.circle_xy,
        state.gridx, state.gridy,
    )
    if state.region_count is not None:
        scores[(state.region_count > 0) & (state.radius > 0)] = np.nan
    return scores, FIT_QUALITY


def rmax_window(state: AppState) -> tuple[float, int, int] | None:
    """(Rmax frame, fit-window start, end), 1-based, as the RofT export uses."""
    if state.radius is None:
        return None
    try:
        estimate = locate_rmax(state.radius, state.rmax_fit_length)
    except ValueError:  # no valid frame yet
        return None
    return estimate.frame, estimate.window[0], estimate.window[1]


def refresh_chart(state: AppState, window) -> np.ndarray | None:
    """Redraw the R(t) chart: markers by source/score/flags, Rmax window.

    Returns
    -------
    np.ndarray or None
        Quality scores with full-frame indexing (length == len(state.radius)).
        Frames with radius <= 0 have score 0.0.  None if no radius data.
    """
    scored = chart_scores(state)
    if scored is None:
        return None
    scores, name = scored
    chart = window.radius_chart
    chart.set_score_name(name)
    chart.plot_all(
        np.arange(1, len(state.radius) + 1), state.radius, scores,
        sources=state.fit_source, flags=state.fit_flags, locked=state.fit_locked,
    )
    rmax = rmax_window(state)
    if rmax is None:
        chart.set_rmax_info(None)
    else:
        chart.set_rmax_info(*rmax)
    return scores


def redraw_original(state: AppState, window) -> None:
    """Clear overlays on original panel and re-draw ROI rect."""
    window.original_panel.clear_overlays()
    window.original_panel.draw_roi_rect(state.gridx, state.gridy)
