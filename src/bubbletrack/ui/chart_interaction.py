"""Mouse interaction helpers of the R(t) chart: picking, menus, hover text."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import QMenu, QWidget

from bubbletrack.ui import chart_style as cs

PICK_TOLERANCE = 0.03  # click distance, as a fraction of the visible axis ranges
HOVER_TOLERANCE = 0.02


class ChartViewBox(pg.ViewBox):
    """ViewBox that turns Shift+drag (or any drag in select mode) into a
    frame-range selection and routes right-clicks to the chart's menus.

    *chart* provides ``select_mode``, ``_drag_selection(x0, x1, finished)``
    and ``_context_menu_at(x, y) -> QMenu | None``.
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
        super().mouseDragEvent(ev, axis)

    def mouseClickEvent(self, ev):  # noqa: N802 (pyqtgraph override)
        if ev.button() == Qt.MouseButton.RightButton:
            pos = self.mapSceneToView(ev.scenePos())
            menu = self._chart._context_menu_at(pos.x(), pos.y())
            if menu is not None:
                ev.accept()
                menu.exec(ev.screenPos().toPoint())
                return
        super().mouseClickEvent(ev)


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
    """Delete / refit / neighbour-refit for one 0-based frame."""
    label = idx + 1
    return _menu(chart, (
        (f"Delete point (Frame {label})", lambda: chart.delete_requested.emit(idx)),
        (f"Refit frame {label}", lambda: chart.refit_requested.emit(idx)),
        (f"Refit using neighbor frames (Frame {label})",
         lambda: chart.neighbor_refit_requested.emit(idx)),
    ))


def build_range_menu(chart, start: int, end: int) -> QMenu:
    """Delete / refit / lock / unlock for the 0-based inclusive range start..end."""
    span = f"frames {start + 1}–{end + 1}"
    return _menu(chart, (
        (f"Delete range ({span})", lambda: chart.delete_range_requested.emit(start, end)),
        (f"Refit range ({span})", lambda: chart.refit_range_requested.emit(start, end)),
        (f"Lock range ({span})", lambda: chart.lock_range_requested.emit(start, end)),
        (f"Unlock range ({span})", lambda: chart.unlock_range_requested.emit(start, end)),
        (f"Mark as no bubble ({span})", lambda: chart.no_bubble_range_requested.emit(start, end)),
        (None, None),
        ("Clear selection", chart.clear_selection),
    ))


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
