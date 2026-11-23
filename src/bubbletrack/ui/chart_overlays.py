"""Annotation layers of the R(t) chart: cursor, selection, and the span of
the selected Review row.

All positions are on the chart's x axis, which is the 1-based frame number.
Rmax (t = 0) and its fit window are kept as data but not drawn: the owner
asked for neither a line nor a shaded band there.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg

CURSOR_PEN = pg.mkPen(228, 228, 231, 150, width=1)
SELECTION_BRUSH = pg.mkBrush(99, 102, 241, 55)
SELECTION_PEN = pg.mkPen(129, 140, 248, 200, width=1)
REVIEW_SPAN_BRUSH = pg.mkBrush(228, 228, 231, 26)  # the selected Review row, unframed


def frames_in_span(x0: float, x1: float, total: int) -> tuple[int, int] | None:
    """Whole 1-based frames covered by the x span [x0, x1], clamped to [1, total].

    A frame counts when its centre lies inside the span.  None when the span
    covers no frame.
    """
    lo, hi = sorted((float(x0), float(x1)))
    start = max(1, int(np.ceil(lo)))
    end = int(np.floor(hi))
    if total > 0:
        end = min(end, total)
    return (start, end) if start <= end else None


class ChartOverlays:
    """Owns the non-data items drawn on a :class:`pyqtgraph.PlotItem`."""

    def __init__(self, plot_item: pg.PlotItem) -> None:
        self._plot = plot_item

        self.cursor = pg.InfiniteLine(angle=90, movable=False, pen=CURSOR_PEN)
        self.cursor.setZValue(-5)
        self.cursor.hide()
        plot_item.addItem(self.cursor, ignoreBounds=True)

        # t = 0 (1-based frame) and its fit window's x span: kept, not drawn
        self._rmax: float | None = None
        self._rmax_window: tuple[float, float] | None = None

        self.selection = pg.LinearRegionItem(
            values=(0, 1), movable=False, brush=SELECTION_BRUSH, pen=SELECTION_PEN,
        )
        self.selection.setZValue(-10)
        self.selection.hide()
        plot_item.addItem(self.selection, ignoreBounds=True)

        # Shading only: unlike the selection, nothing acts on it.
        self.review_span = pg.LinearRegionItem(
            values=(0, 1), movable=False, brush=REVIEW_SPAN_BRUSH, pen=pg.mkPen(None),
        )
        self.review_span.setZValue(-15)
        self.review_span.hide()
        plot_item.addItem(self.review_span, ignoreBounds=True)

    # -- cursor ---------------------------------------------------------- #

    def set_cursor(self, frame_1based: float | None) -> None:
        if frame_1based is None:
            self.cursor.hide()
            return
        self.cursor.setValue(frame_1based)
        self.cursor.show()

    # -- Rmax (kept, not drawn) -------------------------------------------- #

    def set_rmax(self, frame_1based: float | None, lo: int | None = None,
                 hi: int | None = None) -> None:
        """Keep t = 0 at Rmax and the inclusive fit window [lo, hi]; nothing is drawn."""
        if frame_1based is None or not np.isfinite(frame_1based):
            self._rmax, self._rmax_window = None, None
            return
        self._rmax = float(frame_1based)
        self._rmax_window = (None if lo is None or hi is None
                             else (min(lo, hi) - 0.5, max(lo, hi) + 0.5))

    def rmax_info(self) -> tuple[float, tuple[float, float] | None] | None:
        """(t=0 position, fit window's x span or None) while Rmax is known; None otherwise."""
        if self._rmax is None:
            return None
        return self._rmax, self._rmax_window

    # -- range selection --------------------------------------------------- #

    def show_selection(self, x0: float, x1: float) -> None:
        self.selection.setRegion(tuple(sorted((x0, x1))))
        self.selection.show()

    def hide_selection(self) -> None:
        self.selection.hide()

    # -- Review row ------------------------------------------------------- #

    def set_review_span(self, span: tuple[float, float] | None) -> None:
        """Shade the x span (x0, x1); None hides it."""
        if span is None:
            self.review_span.hide()
            return
        self.review_span.setRegion(tuple(sorted(span)))
        self.review_span.show()

    def clear(self) -> None:
        self.set_cursor(None)
        self.set_rmax(None)
        self.hide_selection()
        self.set_review_span(None)
