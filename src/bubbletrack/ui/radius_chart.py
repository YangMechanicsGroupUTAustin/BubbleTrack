"""Interactive R(t) chart (pyqtgraph): marker SHAPE = computed or decided by
the user, COLOUR = score (red -> green); current-frame cursor,
Rmax window + t = 0, collapse lines, Shift+drag range selection with a menu.
Frame arguments are 0-based unless named ``_1based`` (the x axis is 1-based).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from contextlib import contextmanager

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QHBoxLayout, QLabel, QMenu, QPushButton, QSizePolicy, QVBoxLayout, QWidget,
)

from bubbletrack.ui import chart_style as cs
from bubbletrack.ui.chart_interaction import (
    HOVER_TOLERANCE, ChartViewBox, build_point_menu, build_range_menu, hover_text,
    nearest_point,
)
from bubbletrack.ui.chart_overlays import ChartOverlays, frames_in_span


def _chart_button(text: str, tip: str) -> QPushButton:
    btn = QPushButton(text)
    btn.setObjectName("segmentBtn")
    btn.setProperty("segment", "single")
    btn.setFixedHeight(28)
    btn.setToolTip(tip)
    return btn


class RadiusChart(QWidget):
    """Scatter plot of bubble radius vs frame number.

    Signals (frame indices are 0-based):
        point_clicked(int), delete_requested(int), refit_requested(int),
        neighbor_refit_requested(int): single-frame actions, as before.
        range_selected(int, int): a frame range was brushed (inclusive).
        delete_range_requested / refit_range_requested /
        lock_range_requested / unlock_range_requested /
        no_bubble_range_requested(int, int): chosen
        from the selection's menu.
        data_changed(): emitted after :meth:`plot_all` (results changed).
    """

    point_clicked = pyqtSignal(int)
    delete_requested = pyqtSignal(int)
    refit_requested = pyqtSignal(int)
    neighbor_refit_requested = pyqtSignal(int)
    range_selected = pyqtSignal(int, int)
    delete_range_requested = pyqtSignal(int, int)
    refit_range_requested = pyqtSignal(int, int)
    lock_range_requested = pyqtSignal(int, int)
    unlock_range_requested = pyqtSignal(int, int)
    no_bubble_range_requested = pyqtSignal(int, int)
    data_changed = pyqtSignal()  # plot_all() drew new results

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setMinimumHeight(150)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self._plot = pg.PlotWidget(viewBox=ChartViewBox(self))
        self._plot.setBackground(cs.SURFACE_COLOUR)
        self._plot.setLabel("left", "Radius (px)")
        self._plot.setLabel("bottom", "Frame")
        self._plot.showGrid(x=True, y=True, alpha=0.15)
        for axis_name in ("left", "bottom"):
            axis = self._plot.getAxis(axis_name)
            axis.setPen(pg.mkPen(255, 255, 255, 15))
            axis.setTextPen(pg.mkPen("#71717a"))
        item = self._plot.plotItem
        self._overlays = ChartOverlays(item)

        self._raw_series = pg.ScatterPlotItem(
            symbol="o", size=5, pen=pg.mkPen(None), brush=pg.mkBrush(161, 161, 170, 90),
        )
        self._solved_series = pg.PlotDataItem(pen=pg.mkPen("#e4e4e7", width=2), connect="finite")
        self._scatter = pg.ScatterPlotItem(pen=pg.mkPen(cs.SURFACE_COLOUR, width=1))
        self._highlight = pg.ScatterPlotItem(
            pen=pg.mkPen("#818cf8", width=2), brush=pg.mkBrush(99, 102, 241, 50),
            symbol="o", size=22,
        )
        for layer in (self._raw_series, self._solved_series, self._scatter, self._highlight):
            item.addItem(layer)

        self._hover_label = pg.TextItem(color="#e4e4e7", anchor=(0, 1), fill=pg.mkBrush(12, 13, 18, 220))
        self._hover_label.setFont(QFont("Segoe UI", 9))
        self._hover_label.setZValue(50)
        self._hover_label.hide()
        item.addItem(self._hover_label, ignoreBounds=True)
        self._plot.scene().sigMouseMoved.connect(self._on_mouse_moved)
        self._plot.scene().sigMouseClicked.connect(self._on_mouse_clicked)
        layout.addWidget(self._plot, 1)
        layout.addLayout(self._build_legend())

        self._total_frames = 0
        self._frame_labels: tuple[str, ...] = ()
        self._editing_enabled = True
        self._selection: tuple[int, int] | None = None
        self._review_span: tuple[int, int] | None = None
        self._current: int | None = None  # 1-based frame under the cursor
        self._set_points(np.array([]), np.array([]))

    def _build_legend(self) -> QHBoxLayout:
        """Bottom row: shape key, colour ramp, then the chart's buttons."""
        row = QHBoxLayout()
        row.setContentsMargins(8, 0, 0, 0)
        row.setSpacing(10)
        self._legend_shapes = QLabel(cs.shape_legend_html())
        self._score_name = QLabel("fit quality: poor")
        bar = QLabel()
        bar.setFixedSize(90, 9)
        bar.setStyleSheet(f"background: {cs.ramp_gradient_css()}; border-radius: 4px;")
        good = QLabel("good")
        for lbl in (self._legend_shapes, self._score_name, good):
            lbl.setStyleSheet("background: transparent; color:#a1a1aa; font-size:11px;")
        # The keys may be clipped on narrow windows rather than widen them
        # (the shape key goes first); only the buttons set a minimum width.
        self._legend_shapes.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        keys = QWidget()
        keys.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        keys_row = QHBoxLayout(keys)
        keys_row.setContentsMargins(0, 0, 0, 0)
        keys_row.setSpacing(10)
        keys_row.addWidget(self._legend_shapes, 1)
        keys_row.addWidget(self._score_name)
        keys_row.addWidget(bar)
        keys_row.addWidget(good)
        row.addWidget(keys, 1)
        row.addSpacing(12)
        self._select_btn = _chart_button(
            "Select range",
            "Drag across the chart to select frames (Shift+drag works any time).\n"
            "Right-click the selection to delete, refit or lock those frames.",
        )
        self._select_btn.setCheckable(True)
        self._reset_btn = _chart_button(
            "Reset Zoom", "Show all frames (double-click the chart does the same)",
        )
        self._reset_btn.clicked.connect(self._reset_zoom)
        row.addWidget(self._select_btn)
        row.addWidget(self._reset_btn)
        return row

    def plot_all(self, frames: np.ndarray, radii: np.ndarray,
                 quality_scores: np.ndarray | None = None, *,
                 sources: np.ndarray | None = None,
                 flags: Sequence | np.ndarray | None = None,
                 locked: np.ndarray | None = None) -> None:
        """Replot every valid point (radius > 0).

        All arrays are per frame, aligned with *frames* (1-based numbers).
        *quality_scores* in [0, 1] colour the markers (NaN = unscored, grey);
        *sources* (FitSource codes) and *locked* pick the marker shape;
        *flags* (per-frame sets) appear in the hover text.
        """
        frames = np.asarray(frames)
        radii = np.asarray(radii, dtype=float)
        mask = np.isfinite(radii) & (radii > 0)
        sel = lambda a: None if a is None else np.asarray(a)[mask]  # noqa: E731
        self._set_points(
            frames[mask], radii[mask], sel(quality_scores), sel(sources), sel(locked),
            None if flags is None or isinstance(flags, np.ndarray)
            else [flags[i] for i in np.flatnonzero(mask)],
        )
        self.data_changed.emit()

    def _set_points(self, frames, radii, scores=None, sources=None, locked=None,
                    flag_sets=None) -> None:
        self._frames = np.asarray(frames, dtype=float)
        self._radii = np.asarray(radii, dtype=float)
        self._quality_scores = None if scores is None else np.asarray(scores, dtype=float)
        self._sources = sources
        self._locked = locked
        self._flag_sets = flag_sets
        n = len(self._frames)
        if n == 0:
            self._scatter.setData([], [])
            self._highlight.setData([], [])
            return
        symbols = cs.marker_symbols(sources, locked, n)
        self._scatter.setData(
            x=self._frames, y=self._radii, symbol=symbols, size=cs.marker_sizes(symbols),
            brush=cs.score_brushes(self._quality_scores, n),
        )
        self._refresh_highlight()

    def plot_series(self, frames: np.ndarray, raw: np.ndarray | None,
                    solved: np.ndarray | None) -> None:
        """Overlay raw measurements (faint dots) and a solved series (solid line).

        Non-positive or NaN values are gaps.  Pass ``None`` to hide a layer.
        """
        frames = np.asarray(frames, dtype=float)

        def clean(values):
            v = np.asarray(values, dtype=float)
            return np.where(np.isfinite(v) & (v > 0), v, np.nan)

        if raw is None:
            self._raw_series.setData([], [])
        else:
            r = clean(raw)
            ok = np.isfinite(r)
            self._raw_series.setData(frames[ok], r[ok])
        if solved is None:
            self._solved_series.setData([], [])
        else:
            self._solved_series.setData(frames, clean(solved))

    def clear_series(self) -> None:
        self.plot_series(np.array([]), None, None)

    def set_frame_labels(self, labels: Sequence[str]) -> None:
        """File-number tags per frame, shown in the hover text."""
        self._frame_labels = tuple(labels)

    def set_score_name(self, name: str) -> None:
        """Legend wording of the colour ramp, e.g. ``"confidence"``."""
        self._score_name.setText(f"{name}: poor")

    def set_total_frames(self, n: int) -> None:
        """Fix the x-axis range to [1, n]."""
        self._total_frames = n
        if n > 0:
            self._plot.setXRange(0.5, n + 0.5, padding=0)

    def clear(self) -> None:
        """Remove all data points and annotations from the chart."""
        self._set_points(np.array([]), np.array([]))
        self.clear_series()
        self._hover_label.hide()
        self._overlays.clear()
        self._selection = self._review_span = None

    def set_current_frame(self, idx: int | None) -> None:
        """Move the current-frame cursor to 0-based *idx* (None hides it)."""
        self._overlays.set_cursor(None if idx is None else idx + 1)
        self._current = None if idx is None else idx + 1
        self._refresh_highlight()

    def current_frame(self) -> int | None:
        """0-based frame under the cursor, or None."""
        return None if self._current is None else self._current - 1

    def set_rmax_info(self, frame_1based: float | None, window_lo: int | None = None,
                      window_hi: int | None = None) -> None:
        """Mark t = 0 at Rmax (*frame_1based*, may be fractional) and shade the
        inclusive 1-based fit window; ``None`` removes both."""
        self._overlays.set_rmax(frame_1based, window_lo, window_hi)

    def rmax_info(self):
        return self._overlays.rmax_info()

    def set_cycles(self, collapse_frames: Iterable[int]) -> None:
        """Dashed lines at collapse frames (0-based indices); empty clears."""
        self._overlays.set_cycles(int(f) + 1 for f in collapse_frames)

    def cycle_frames(self) -> list[int]:
        return [int(round(x)) - 1 for x in self._overlays.cycle_positions()]

    def highlight_frame(self, frame_1based: int) -> None:
        """Same as ``set_current_frame(frame_1based - 1)`` (kept for callers)."""
        self.set_current_frame(frame_1based - 1)

    @contextmanager
    def export_view(self):
        """Yield the PlotItem with the cursor, highlight, hover text and
        selection hidden, e.g. for ``ImageExporter``; restores them after."""
        items = (self._overlays.cursor, self._overlays.selection, self._overlays.review_span,
                 self._hover_label, self._highlight)
        shown = [item.isVisible() for item in items]
        for item in items:
            item.hide()
        try:
            yield self._plot.plotItem
        finally:
            for item, visible in zip(items, shown):
                item.setVisible(visible)

    def _refresh_highlight(self) -> None:
        cur = self._current
        hit = np.flatnonzero(self._frames == cur) if cur is not None else []
        if len(hit):
            self._highlight.setData([self._frames[hit[0]]], [self._radii[hit[0]]])
        else:
            self._highlight.setData([], [])

    @property
    def select_mode(self) -> bool:
        return self._select_btn.isChecked()

    def set_select_mode(self, on: bool) -> None:
        self._select_btn.setChecked(on)

    def selected_range(self) -> tuple[int, int] | None:
        """The selected frames as a 0-based inclusive (start, end), or None."""
        return self._selection

    def set_selection(self, start: int, end: int) -> None:
        """Select 0-based frames start..end (inclusive) without emitting."""
        lo, hi = sorted((int(start), int(end)))
        self._selection = (lo, hi)
        self._overlays.show_selection(lo + 0.5, hi + 1.5)

    def clear_selection(self) -> None:
        self._selection = None
        self._overlays.hide_selection()

    def set_review_span(self, span: tuple[int, int] | None) -> None:
        """Shade 0-based frames start..end (the selected Review row), or none.
        Only a marker: clicks and menus never act on it."""
        self._review_span = None if span is None else tuple(sorted(int(v) for v in span))
        lo, hi = self._review_span or (None, None)
        self._overlays.set_review_span(None if lo is None else (lo + 0.5, hi + 1.5))

    def review_span(self) -> tuple[int, int] | None:
        return self._review_span

    def _drag_selection(self, x0: float, x1: float, finished: bool) -> None:
        """Rubber-band a range while dragging; commit and emit on release."""
        self._overlays.show_selection(x0, x1)
        if not finished:
            return
        span = frames_in_span(x0, x1, self._total_frames)
        if span is None:
            self.clear_selection()
            return
        self.set_selection(span[0] - 1, span[1] - 1)
        self.range_selected.emit(*self._selection)

    def set_editing_enabled(self, enabled: bool) -> None:
        """Enable/disable the delete/refit/lock menus (off while a fit runs)."""
        self._editing_enabled = enabled

    def _find_nearest_point(self, x: float, y: float, tol: float | None = None) -> int | None:
        """Position (into the plotted arrays) of the point nearest (x, y), or None."""
        view = self._plot.plotItem.vb.viewRange()
        if tol is None:
            return nearest_point(self._frames, self._radii, view, x, y)
        return nearest_point(self._frames, self._radii, view, x, y, tol)

    def _frame_at(self, x: float, y: float) -> int | None:
        k = self._find_nearest_point(x, y)
        return None if k is None else int(self._frames[k]) - 1

    def _on_mouse_clicked(self, event) -> None:
        """Double-click resets zoom; left-click jumps to the nearest point."""
        if event.double():
            self._reset_zoom()
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        pos = self._plot.plotItem.vb.mapSceneToView(event.scenePos())
        frame_idx = self._frame_at(pos.x(), pos.y())
        if self._selection is not None and not self._in_selection(pos.x()):
            self.clear_selection()
        if frame_idx is not None:
            self.highlight_frame(frame_idx + 1)
            self.point_clicked.emit(frame_idx)

    def _in_selection(self, x: float) -> bool:
        return self._selection is not None and \
            self._selection[0] + 0.5 <= x <= self._selection[1] + 1.5

    def _context_menu_at(self, x: float, y: float) -> QMenu | None:
        """The edit menu for a right-click at view (x, y); None = default menu."""
        if not self._editing_enabled:
            return None
        if self._in_selection(x):
            return build_range_menu(self, *self._selection)
        frame_idx = self._frame_at(x, y)
        return None if frame_idx is None else build_point_menu(self, frame_idx)

    def _reset_zoom(self) -> None:
        """Reset axes to auto-range (with the fixed frame range when set)."""
        self._plot.enableAutoRange()
        if self._total_frames > 0:
            self._plot.setXRange(0.5, self._total_frames + 0.5, padding=0)

    def _on_mouse_moved(self, pos) -> None:
        """Show frame, file number, radius, score and provenance on hover."""
        p = self._plot.plotItem.vb.mapSceneToView(pos)
        k = self._find_nearest_point(p.x(), p.y(), HOVER_TOLERANCE)
        if k is None:
            self._hover_label.hide()
            return
        self._hover_label.setText(self._hover_text(k))
        self._hover_label.setPos(self._frames[k], self._radii[k])
        self._hover_label.show()

    def _hover_text(self, k: int) -> str:
        frame = int(self._frames[k])
        at = lambda arr: None if arr is None else arr[k]  # noqa: E731
        return hover_text(
            frame, float(self._radii[k]),
            file_tag=self._frame_labels[frame - 1] if 0 < frame <= len(self._frame_labels) else "",
            score=at(self._quality_scores), source=at(self._sources),
            locked=bool(at(self._locked)), flags=at(self._flag_sets),
        )
