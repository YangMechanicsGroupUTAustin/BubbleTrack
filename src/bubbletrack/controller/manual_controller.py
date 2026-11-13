"""ManualController — "Click edge points": a circle through points the user clicks.

The fallback tool of the Review dock's Fix (``FixMode``); the point count is
shown by the view attached with ``set_view`` (the fix strip).
"""

from __future__ import annotations

import logging

import numpy as np

from bubbletrack.controller.base import BaseController
from bubbletrack.controller.display_mixin import (
    display_frame,
    redraw_original,
    refresh_chart,
)
from bubbletrack.event_bus import EventBus
from bubbletrack.model.cache import ImageCache
from bubbletrack.model.circle_fit import circle_fit_taubin
from bubbletrack.model.constants import MIN_POINT_DISTANCE_PX
from bubbletrack.model.provenance import FitSource
from bubbletrack.model.result_history import ResultHistory, record_edit, snapshot
from bubbletrack.model.state import clear_frame_result
from bubbletrack.model.suggestion import set_suggestion
from bubbletrack.model.undo import UndoStack
from bubbletrack.ui.circle_tool import TOOL_POINTS
from bubbletrack.ui.fix_strip import FixView, NoFixView

logger = logging.getLogger(__name__)


def is_duplicate_point(
    new_pt: np.ndarray,
    existing: list[tuple[float, float]],
    threshold: float = MIN_POINT_DISTANCE_PX,
) -> bool:
    """Return True if *new_pt* is within *threshold* px of any existing point.

    Parameters
    ----------
    new_pt : (2,) array
        New point as ``[row, col]``.
    existing : list of (row, col) tuples
        Already-accepted manual points.
    threshold : float
        Minimum Euclidean distance in pixels.
    """
    for pt in existing:
        if np.linalg.norm(np.array(pt) - new_pt) < threshold:
            return True
    return False


class ManualController(BaseController):
    """Manages manual edge-point selection and fitting."""

    def __init__(self, bus: EventBus, get_state, set_state, window,
                 get_max_radius, cache: ImageCache | None = None, *,
                 history: ResultHistory | None = None) -> None:
        super().__init__(bus, get_state, set_state, window)
        self._manual_points: list[tuple[float, float]] = []
        self._get_max_radius = get_max_radius
        self._undo_stack = UndoStack()  # clicked points of the running selection
        # Committed fits and clears join the result undo history (Ctrl+Z),
        # shared with the circle tool, so Ctrl+Z always undoes the newest fix.
        self._history = history if history is not None else ResultHistory()
        self._cache = cache
        self._selecting = False
        self._view: FixView = NoFixView()
        bus.subscribe("frame_changed", self._on_frame_changed)

    def set_view(self, view: FixView) -> None:
        """Show the point count in *view* (the Review dock's fix strip)."""
        self._view = view

    @property
    def selecting(self) -> bool:
        """True while clicks on the image add edge points."""
        return self._selecting

    # -- public handlers -------------------------------------------------- #

    def on_manual_select(self, *, announce: bool = True) -> None:
        """Start a point selection on the frame on screen (any earlier points go)."""
        self._manual_points.clear()
        self._undo_stack.clear()
        self._selecting = True
        self._view.set_point_count(0)
        self.w.original_panel.set_mode("point")
        if announce:
            self.w.header.set_status("Click on bubble edge", "#f59e0b")
        self.w.status_bar.update_mode("Manual")

    def _on_frame_changed(self, _idx: int) -> None:
        """Points belong to the frame they were clicked on: cancel on navigation."""
        if not self._selecting and not self._manual_points:
            return
        self._end_selection()
        self.w.header.set_status(
            "Manual selection cancelled: the frame changed", "#f59e0b",
        )

    def on_tool_changed(self, tool: str) -> None:
        """Leaving the "Click edge points" tool cancels a running selection."""
        if tool != TOOL_POINTS:
            self.cancel_selection()

    def cancel_selection(self) -> None:
        """Drop a running selection and its points (nothing is stored)."""
        if self._selecting or self._manual_points:
            self._end_selection()

    def _end_selection(self) -> None:
        self._selecting = False
        self._manual_points.clear()
        self._undo_stack.clear()
        self._view.set_point_count(0)
        self.w.original_panel.set_mode("normal")

    def on_point_clicked(self, x: float, y: float) -> None:
        """x = col, y = row in scene coordinates."""
        if not self._selecting:
            return  # clicks while not selecting belong to someone else (e.g. seed pick)
        new_pt = np.array([y, x])  # row, col
        if is_duplicate_point(new_pt, self._manual_points):
            logger.info("Ignoring duplicate point near (%.1f, %.1f)", x, y)
            return
        # Snapshot current points before adding the new one
        self._undo_stack.push(list(self._manual_points))
        self._manual_points.append((y, x))  # store as (row, col)
        n = len(self._manual_points)
        self._view.set_point_count(n)
        # Draw the point
        pt = np.array([[y, x]])
        self.w.original_panel.draw_points(pt, "#ef4444", 4.0)

    def on_manual_done(self) -> bool:
        """Fit a circle through the clicked points; True if a result was stored.

        With fewer than 3 points the selection goes on; otherwise it ends
        (a failed or out-of-range fit stores nothing).
        """
        if not self.state.images or self.state.radius is None:
            self._end_selection()
            self.w.header.set_status("Load images before fitting", "#ef4444")
            return False
        if len(self._manual_points) < 3:
            self.w.header.set_status("Need at least 3 points", "#ef4444")
            return False

        xy = np.array(self._manual_points)
        rc, cc, radius = circle_fit_taubin(xy)

        idx = self.state.image_no
        stored = False
        if np.isnan(radius):
            self.w.header.set_status("Fitting failed", "#ef4444")
        elif radius > self._get_max_radius():
            self.w.header.set_status(
                f"Radius outlier ({radius:.0f} px), skipped", "#f59e0b",
            )
        else:
            stored = True
            before = snapshot(self.state, [idx])
            self.state.radius[idx] = radius
            self.state.circle_fit_par[idx] = [rc, cc]
            self.state.circle_xy[idx] = xy
            if self.state.region_count is not None:
                self.state.region_count[idx] = 0
            if self.state.fit_source is not None:
                self.state.fit_source[idx] = FitSource.MANUAL
            set_suggestion(self.state, idx, None)  # the user's fit replaces Auto's guess
            self._history.push(record_edit(self.state, f"manual fit on frame {idx + 1}", before))

            # Clear old overlays before drawing new results
            redraw_original(self.state, self.w)
            self.w.original_panel.draw_circle(rc, cc, radius, "#6366f1")
            self.w.original_panel.draw_points(xy, "#ef4444", 2.0)
            refresh_chart(self.state, self.w)
            self.w.header.set_status(f"R = {radius:.1f} px", "#10b981")
            self._results_changed()

        # The fit is committed: Ctrl+Z must not resurrect its points.
        self._end_selection()
        return stored

    def undo_last_point(self) -> bool:
        """Undo the last point click, restoring the previous points list.

        Returns False when there was no click to undo.
        """
        snapshot = self._undo_stack.undo()
        if snapshot is None:
            return False
        self._manual_points = snapshot
        n = len(self._manual_points)
        self._view.set_point_count(n)
        # Redraw all current points
        self.w.original_panel.clear_overlays()
        if self._manual_points:
            pts = np.array(self._manual_points)
            self.w.original_panel.draw_points(pts, "#ef4444", 4.0)
        logger.info("Undo: %d points remaining", n)
        return True

    def on_manual_clear(self) -> None:
        self._end_selection()

        # Mark the current frame as unprocessed, using the same sentinels as
        # everywhere else (-1 / NaN): a 0 radius exported as a 0 um point.
        idx = self.state.image_no
        before = snapshot(self.state, [idx]) if self.state.radius is not None else None
        clear_frame_result(self.state, idx)
        if before is not None:
            self._history.push(record_edit(self.state, f"clear of frame {idx + 1}", before))

        display_frame(self.state, self.w, idx, self._set_state, self._cache)

        # Refresh R-t chart
        if self.state.radius is not None:
            frames = np.arange(1, len(self.state.radius) + 1)
            self.w.radius_chart.plot_all(frames, self.state.radius)

        self.w.header.set_status("Ready", "#10b981")
        self._results_changed()

    def clear_points(self) -> None:
        """Clear the manual points list (called on tab change)."""
        self._selecting = False
        self._manual_points.clear()
        self._undo_stack.clear()
        self._view.set_point_count(0)
