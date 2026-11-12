"""Annotation layers of the R(t) chart: cursor, Rmax window, cycles, selection,
and the span of the selected Review row.

All positions are on the chart's x axis, which is the 1-based frame number.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt

CURSOR_PEN = pg.mkPen(228, 228, 231, 150, width=1)
RMAX_BAND_BRUSH = pg.mkBrush(165, 180, 252, 28)
RMAX_LINE_PEN = pg.mkPen("#a5b4fc", width=1.5, style=Qt.PenStyle.DashLine)
CYCLE_PEN = pg.mkPen("#71717a", width=1.2, style=Qt.PenStyle.DashLine)
SELECTION_BRUSH = pg.mkBrush(99, 102, 241, 55)
SELECTION_PEN = pg.mkPen(129, 140, 248, 200, width=1)
REVIEW_SPAN_BRUSH = pg.mkBrush(228, 228, 231, 26)  # the selected Review row, unframed
# Label positions run along the line from the bottom (0) to the top (1) of the view.
# Rmax sits high on the curve, so its label goes low; collapses are low, so theirs go high.
TOP_LABEL = {"position": 0.97, "color": "#a1a1aa", "anchors": [(0, 0), (0, 0)]}
BOTTOM_LABEL = {"position": 0.03, "color": "#c7d2fe", "anchors": [(0, 1), (0, 1)]}


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

        self.rmax_band = pg.LinearRegionItem(
            values=(0, 1), movable=False, brush=RMAX_BAND_BRUSH,
            pen=pg.mkPen(None),
        )
        self.rmax_band.setZValue(-20)
        self.rmax_band.hide()
        plot_item.addItem(self.rmax_band, ignoreBounds=True)

        self.rmax_line = pg.InfiniteLine(
            angle=90, movable=False, pen=RMAX_LINE_PEN,
            label="t = 0 (Rmax)", labelOpts=BOTTOM_LABEL,
        )
        self.rmax_line.setZValue(-4)
        self.rmax_line.hide()
        plot_item.addItem(self.rmax_line, ignoreBounds=True)

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

        self._cycle_lines: list[pg.InfiniteLine] = []

    # -- cursor ---------------------------------------------------------- #

    def set_cursor(self, frame_1based: float | None) -> None:
        if frame_1based is None:
            self.cursor.hide()
            return
        self.cursor.setValue(frame_1based)
        self.cursor.show()

    # -- Rmax fit window -------------------------------------------------- #

    def set_rmax(self, frame_1based: float | None, lo: int | None = None,
                 hi: int | None = None) -> None:
        if frame_1based is None or not np.isfinite(frame_1based):
            self.rmax_band.hide()
            self.rmax_line.hide()
            return
        self.rmax_line.setValue(float(frame_1based))
        self.rmax_line.show()
        if lo is None or hi is None:
            self.rmax_band.hide()
            return
        self.rmax_band.setRegion((min(lo, hi) - 0.5, max(lo, hi) + 0.5))
        self.rmax_band.show()

    def rmax_info(self) -> tuple[float, tuple[float, float] | None] | None:
        """(t=0 position, band region or None) while shown; None otherwise."""
        if not self.rmax_line.isVisible():
            return None
        band = tuple(self.rmax_band.getRegion()) if self.rmax_band.isVisible() else None
        return float(self.rmax_line.value()), band

    # -- collapse cycles -------------------------------------------------- #

    def set_cycles(self, frames_1based: Iterable[float]) -> None:
        for line in self._cycle_lines:
            self._plot.removeItem(line)
        self._cycle_lines = []
        for k, x in enumerate(frames_1based, start=1):
            line = pg.InfiniteLine(
                pos=float(x), angle=90, movable=False, pen=CYCLE_PEN,
                label=f"collapse {k}", labelOpts=TOP_LABEL,
            )
            line.setZValue(-6)
            self._plot.addItem(line, ignoreBounds=True)
            self._cycle_lines.append(line)

    def cycle_positions(self) -> list[float]:
        return [float(line.value()) for line in self._cycle_lines]

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
        self.set_cycles(())
        self.hide_selection()
        self.set_review_span(None)
