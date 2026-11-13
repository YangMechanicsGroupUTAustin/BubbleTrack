"""Mouse interaction helpers of the R(t) chart: picking, menus, hover text."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QFont
from PyQt6.QtWidgets import QMenu, QWidget

from bubbletrack.ui import chart_style as cs

PICK_TOLERANCE = 0.03  # click distance, as a fraction of the visible axis ranges
HOVER_TOLERANCE = 0.02
_X, _Y = 0, 1  # pyqtgraph axis numbers


class ChartViewBox(pg.ViewBox):
    """ViewBox of the R(t) chart.

    * Shift+drag (or any drag in select mode) selects a frame range;
    * in the plot, the wheel and drags zoom / pan the frame axis only (the
      radius axis follows the visible frames); on the radius axis they zoom /
      pan y and the user takes y over (``chart._release_y()``);
    * right-clicks open the chart's menus (pyqtgraph's own menu is off).

    *chart* provides ``select_mode``, ``min_x_span``, ``_drag_selection(x0, x1,
    finished)``, ``_release_y()``, ``_context_menu_at(x, y) -> QMenu | None``
    and ``_view_menu() -> QMenu``.
    """

    def __init__(self, chart) -> None:
        super().__init__()
        self._chart = chart

    def mouseDragEvent(self, ev, axis=None):  # noqa: N802 (pyqtgraph override)
        shift = bool(ev.modifiers() & Qt.KeyboardModifier.ShiftModifier)
        if axis is None and ev.button() == Qt.MouseButton.LeftButton \
                and (shift or self._chart.select_mode):
            ev.accept()
            x0 = self.mapSceneToView(ev.buttonDownScenePos()).x()
            x1 = self.mapSceneToView(ev.scenePos()).x()
            self._chart._drag_selection(x0, x1, finished=ev.isFinish())
            return
        if axis == _Y:
            self._chart._release_y()
        super().mouseDragEvent(ev, _X if axis is None else axis)

    def wheelEvent(self, ev, axis=None):  # noqa: N802 (pyqtgraph override)
        axis = _X if axis is None else axis
        if axis == _Y:
            self._chart._release_y()
        elif self._at_deepest_x_zoom(ev.delta()):
            ev.accept()  # zooming further would only slide the view off the cursor
            return
        super().wheelEvent(ev, axis)

    def _at_deepest_x_zoom(self, delta: float) -> bool:
        zooming_in = delta * self.state["wheelScaleFactor"] < 0
        lo, hi = self.viewRange()[_X]
        return zooming_in and hi - lo <= self._chart.min_x_span * (1 + 1e-6)

    def mouseClickEvent(self, ev):  # noqa: N802 (pyqtgraph override)
        if ev.button() == Qt.MouseButton.RightButton:
            pos = self.mapSceneToView(ev.scenePos())
            menu = self._chart._context_menu_at(pos.x(), pos.y())
            if menu is None:
                menu = self._chart._view_menu()
            ev.accept()
            menu.exec(ev.screenPos().toPoint())
            menu.deleteLater()  # one menu per right-click: do not pile them up on the chart
            return
        super().mouseClickEvent(ev)


def make_hover_label() -> pg.TextItem:
    """The hidden text box that shows a point's details under the mouse."""
    label = pg.TextItem(color="#e4e4e7", anchor=(0, 1), fill=pg.mkBrush(12, 13, 18, 220))
    label.setFont(QFont("Segoe UI", 9))
    label.setZValue(50)
    label.hide()
    return label


def nearest_point(frames: np.ndarray, radii: np.ndarray, view_range,
                  x: float, y: float, tol: float = PICK_TOLERANCE) -> int | None:
    """Position of the point nearest view (x, y) within *tol*, or None.

    Distances are measured in fractions of the visible axis ranges, so the
    tolerance feels the same at any zoom.
    """
    if len(frames) == 0 or not (np.isfinite(x) and np.isfinite(y)):
        return None
    (x_lo, x_hi), (y_lo, y_hi) = view_range
    if x_hi <= x_lo or y_hi <= y_lo:
        return None
    dists = ((frames - x) / (x_hi - x_lo)) ** 2 + ((radii - y) / (y_hi - y_lo)) ** 2
    k = int(np.argmin(dists))
    return k if dists[k] <= 2 * tol ** 2 else None


def _menu(parent: QWidget, entries) -> QMenu:
    menu = QMenu(parent)
    for text, callback in entries:
        if text is None:
            menu.addSeparator()
            continue
        action = QAction(text, menu)
        action.triggered.connect(lambda _checked=False, cb=callback: cb())
        menu.addAction(action)
    return menu


def build_point_menu(chart, idx: int) -> QMenu:
    """"Fix frame N…" for one 0-based frame (the fixing happens in Review).

    No "Delete": an R(t) curve with holes is useless; "Mark as no bubble"
    on a selected range says a frame is empty.
    """
    return _menu(chart, (
        (f"Fix frame {idx + 1}…", lambda: chart.fix_requested.emit(idx)),
    ))


def build_range_menu(chart, start: int, end: int) -> QMenu:
    """Refit / lock / unlock / mark empty for the 0-based inclusive range start..end."""
    span = f"frames {start + 1}–{end + 1}"
    return _menu(chart, (
        (f"Refit range ({span})", lambda: chart.refit_range_requested.emit(start, end)),
        (f"Lock range ({span})", lambda: chart.lock_range_requested.emit(start, end)),
        (f"Unlock range ({span})", lambda: chart.unlock_range_requested.emit(start, end)),
        (f"Mark as no bubble ({span})", lambda: chart.no_bubble_range_requested.emit(start, end)),
        (None, None),
        ("Clear selection", chart.clear_selection),
    ))


def build_view_menu(chart) -> QMenu:
    """Right-click away from points and selection (replaces pyqtgraph's menu)."""
    return _menu(chart, (("Reset zoom", chart.reset_zoom),))


def hover_text(frame: int, radius: float, *, file_tag: str = "", score: float | None = None,
               source: int | None = None, locked: bool = False,
               flags: Sequence[str] | frozenset | None = None) -> str:
    """``Frame 44 · _054  R 16.4 px  Q 25%  manual  flags: a, b``."""
    parts = [f"Frame {frame}" + (f" · {file_tag}" if file_tag else ""),
             f"R {radius:.1f} px"]
    if score is not None and np.isfinite(score):
        parts.append(f"Q {int(round(score * 100))}%")
    name = cs.source_name(source, locked)
    if name:
        parts.append(name)
    if flags:
        parts.append("flags: " + ", ".join(sorted(flags)))
    return "  ".join(parts)
