"""The R(t) chart's view: always on the data, whatever the user drags or zooms.

* x (frames) is limited to [0.5, N + 0.5] and spans at least ``MIN_X_SPAN``
  frames; the wheel and plain drags in the plot zoom and pan x only.
* y (radius) is limited to [0, max R * ``Y_HEADROOM``] and follows the
  visible frames from zero, so zooming into a collapse shows its small radii.
  Zooming or dragging on the radius axis takes y over (at least
  ``MIN_Y_SPAN`` px) until the view is reset.
* The full view (all frames, 0..max R) is restored by :meth:`ChartNavigator.reset`.

The limits are pyqtgraph ``ViewBox.setLimits``, so no drag or zoom can leave them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg

Y_HEADROOM = 1.15  # the top of the y axis, as a multiple of the largest radius
MIN_X_SPAN = 10.0  # frames: the deepest x zoom
MIN_Y_SPAN = 3.0  # px: the deepest y zoom (on the radius axis)
EMPTY_Y_MAX = 100.0  # px: the y range before any result
EMPTY_FRAMES = 10  # the x range before frames are known


@dataclass(frozen=True)
class ViewLimits:
    """Where the view may go: x and y extents and the smallest spans."""

    x: tuple[float, float]
    y: tuple[float, float]
    min_x_span: float
    min_y_span: float

    def as_kwargs(self) -> dict[str, float]:
        """Keyword arguments of ``pyqtgraph.ViewBox.setLimits``."""
        return {"xMin": self.x[0], "xMax": self.x[1], "yMin": self.y[0], "yMax": self.y[1],
                "minXRange": self.min_x_span, "maxXRange": self.x[1] - self.x[0],
                "minYRange": self.min_y_span, "maxYRange": self.y[1] - self.y[0]}


def _valid(xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = np.asarray(xs, dtype=float).ravel(), np.asarray(ys, dtype=float).ravel()
    ok = np.isfinite(xs) & np.isfinite(ys) & (ys > 0)
    return xs[ok], ys[ok]


def view_limits(total_frames: int, xs: np.ndarray, ys: np.ndarray) -> ViewLimits:
    """Limits for *total_frames* frames (0: the frames plotted) and the points (xs, ys)."""
    xs, ys = _valid(xs, ys)
    if total_frames > 0:
        n = int(total_frames)
    else:
        n = max(int(np.ceil(xs.max())), 1) if len(xs) else EMPTY_FRAMES
    y_max = float(ys.max()) * Y_HEADROOM if len(ys) else EMPTY_Y_MAX
    return ViewLimits((0.5, n + 0.5), (0.0, y_max), min(MIN_X_SPAN, float(n)),
                      min(MIN_Y_SPAN, y_max))


def fitted_y(limits: ViewLimits, xs: np.ndarray, ys: np.ndarray,
             x_range: tuple[float, float]) -> tuple[float, float]:
    """y range showing the points within *x_range*: from 0 to their largest
    radius with headroom (the full height when none is visible)."""
    xs, ys = _valid(xs, ys)
    lo, hi = x_range
    visible = ys[(xs >= lo) & (xs <= hi)]
    top = float(visible.max()) * Y_HEADROOM if len(visible) else limits.y[1]
    return 0.0, float(np.clip(top, limits.min_y_span, limits.y[1]))


class ChartNavigator:
    """Applies the limits to a ViewBox and keeps y on the visible frames."""

    def __init__(self, view_box: pg.ViewBox) -> None:
        self._vb = view_box
        self._total = 0
        self._xs = self._ys = np.empty(0)
        self._limits = view_limits(0, self._xs, self._ys)
        self.y_follows = True
        view_box.disableAutoRange()
        view_box.sigXRangeChanged.connect(self._fit_y)
        self._apply_limits()
        self.reset()

    @property
    def limits(self) -> ViewLimits:
        return self._limits

    def set_total_frames(self, n: int) -> None:
        """A new recording: new x limits and the full view."""
        self._total = max(int(n), 0)
        self._apply_limits()
        self.reset()

    def set_data(self, xs: np.ndarray, ys: np.ndarray) -> None:
        """Every plotted point (frame, radius): the y limit follows the largest radius."""
        showing_all = self._showing_all_x()
        self._xs, self._ys = _valid(xs, ys)
        self._apply_limits()
        if showing_all:  # the frame range may come from the data (no frame count)
            self._vb.setXRange(*self._limits.x, padding=0)
        self._fit_y()

    def reset(self) -> None:
        """The full view: all frames, y from 0 to the largest radius."""
        self.y_follows = True
        self._vb.setXRange(*self._limits.x, padding=0)
        self._fit_y()

    def release_y(self) -> None:
        """The user zoomed or dragged the radius axis: keep their y until a reset."""
        self.y_follows = False

    def _showing_all_x(self) -> bool:
        lo, hi = self._vb.viewRange()[0]
        return abs(lo - self._limits.x[0]) < 1e-6 and abs(hi - self._limits.x[1]) < 1e-6

    def _apply_limits(self) -> None:
        self._limits = view_limits(self._total, self._xs, self._ys)
        self._vb.setLimits(**self._limits.as_kwargs())

    def _fit_y(self, *_args) -> None:
        if self.y_follows:
            x_range = tuple(self._vb.viewRange()[0])
            self._vb.setYRange(*fitted_y(self._limits, self._xs, self._ys, x_range), padding=0)
