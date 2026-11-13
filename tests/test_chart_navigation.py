"""R(t) chart navigation: the view stays on the data whatever the user drags or
zooms; x zooms around the cursor, y follows the visible frames; double-click
and Reset Zoom restore the full view; the menus never delete results."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyqtgraph as pg
import pytest
from PyQt6.QtCore import QPointF

import bubbletrack
from bubbletrack.ui import chart_view as cv
from bubbletrack.ui.radius_chart import CHART_MIN_HEIGHT, RadiusChart
from test_viz_chart import _FakeDrag

N = 100


def _radii(n: int = N) -> np.ndarray:
    """A bubble that grows to 50 px, collapses at frame 60 and rebounds to 20 px."""
    x = np.arange(1, n + 1, dtype=float)
    r = np.where(x <= 60, 50 * np.sin(np.pi * x / 60), 20 * np.sin(np.pi * (x - 60) / 40))
    return np.clip(r, 1.0, None)


@pytest.fixture
def chart(qapp):
    c = RadiusChart()
    c.resize(900, 320)
    c.show()
    qapp.processEvents()
    c.set_total_frames(N)
    c.plot_all(np.arange(1, N + 1), _radii())
    qapp.processEvents()
    yield c
    c.close()


def _vb(chart: RadiusChart) -> pg.ViewBox:
    return chart._plot.plotItem.vb


def _x(chart):
    return tuple(_vb(chart).viewRange()[0])


def _y(chart):
    return tuple(_vb(chart).viewRange()[1])


def _full_y() -> float:
    return float(_radii().max()) * cv.Y_HEADROOM


class _FakeWheel:
    """Just enough of a QGraphicsSceneWheelEvent for ViewBox.wheelEvent."""

    def __init__(self, chart, x: float, y: float, steps: int) -> None:
        vb = _vb(chart)
        self._pos = vb.mapFromView(QPointF(x, y))
        self._delta = 120 * steps

    def delta(self):
        return self._delta

    def pos(self):
        return self._pos

    def accept(self):
        pass

    def ignore(self):
        pass


def _wheel(chart, x, y, steps, axis=None):
    _vb(chart).wheelEvent(_FakeWheel(chart, x, y, steps), axis)


# -- pure limits ------------------------------------------------------------------

class TestLimits:
    def test_limits_follow_frames_and_largest_radius(self):
        lim = cv.view_limits(100, np.array([1.0, 50.0]), np.array([10.0, 40.0]))
        assert lim.x == (0.5, 100.5) and lim.y == pytest.approx((0.0, 40.0 * cv.Y_HEADROOM))
        assert lim.min_x_span == cv.MIN_X_SPAN and lim.min_y_span == cv.MIN_Y_SPAN

    def test_fallbacks_without_data_or_frame_count(self):
        empty = cv.view_limits(0, np.array([]), np.array([]))
        assert empty.y == (0.0, cv.EMPTY_Y_MAX) and empty.x[0] == 0.5
        from_data = cv.view_limits(0, np.array([3.0, 7.0]), np.array([5.0, 6.0]))
        assert from_data.x == (0.5, 7.5)
        short = cv.view_limits(4, np.array([1.0]), np.array([5.0]))
        assert short.min_x_span == 4  # never more than every frame

    def test_y_fits_the_visible_frames_from_zero(self):
        lim = cv.view_limits(100, np.arange(1.0, 101.0), _radii())
        xs, ys = np.arange(1.0, 101.0), _radii()
        assert cv.fitted_y(lim, xs, ys, (0.5, 100.5)) == pytest.approx((0.0, _full_y()))
        lo, hi = cv.fitted_y(lim, xs, ys, (55.5, 65.5))
        assert lo == 0.0 and hi == pytest.approx(ys[55:65].max() * cv.Y_HEADROOM)
        assert cv.fitted_y(lim, xs, ys, (200.0, 300.0)) == pytest.approx((0.0, lim.y[1]))


# -- the widget --------------------------------------------------------------------

class TestFullView:
    def test_starts_at_the_full_view_with_limits(self, chart):
        assert _x(chart) == pytest.approx((0.5, N + 0.5))
        assert _y(chart) == pytest.approx((0.0, _full_y()))
        limits = _vb(chart).state["limits"]
        assert limits["xLimits"] == pytest.approx([0.5, N + 0.5])
        assert limits["yLimits"] == pytest.approx([0.0, _full_y()])
        assert limits["xRange"][0] == cv.MIN_X_SPAN and limits["yRange"][0] == cv.MIN_Y_SPAN

    def test_new_results_raise_the_y_limit(self, chart):
        radii = _radii()
        radii[10] = 80.0
        chart.plot_all(np.arange(1, N + 1), radii)
        assert _y(chart) == pytest.approx((0.0, 80.0 * cv.Y_HEADROOM))

    def test_no_data_has_a_sensible_view(self, qapp):
        c = RadiusChart()
        c.resize(600, 300)
        c.show()
        qapp.processEvents()
        c.set_total_frames(30)
        assert tuple(c._plot.plotItem.vb.viewRange()[0]) == pytest.approx((0.5, 30.5))
        assert tuple(c._plot.plotItem.vb.viewRange()[1]) == pytest.approx((0.0, cv.EMPTY_Y_MAX))
        c.close()


class TestNeverWayOff:
    def test_panning_stops_at_the_ends(self, chart):
        _wheel(chart, 30, 20, 5)  # zoom in first so there is room to pan
        span = _x(chart)[1] - _x(chart)[0]
        for dx in (-500, 500, -5000):
            chart._plot.plotItem.vb.mouseDragEvent(
                _FakeDrag(chart, 50, 50 + dx, shift=False, finish=False))
            lo, hi = _x(chart)
            assert lo >= 0.5 - 1e-6 and hi <= N + 0.5 + 1e-6
            assert hi - lo == pytest.approx(span, rel=1e-6)

    def test_plain_drag_pans_time_only(self, chart):
        _vb(chart).setXRange(65.5, 85.5, padding=0)  # the rebound: R <= 20 px
        assert _y(chart)[1] == pytest.approx(20.0 * cv.Y_HEADROOM, rel=0.01)
        chart._plot.plotItem.vb.mouseDragEvent(_FakeDrag(chart, 60, 80, shift=False, finish=False))
        assert chart.y_follows_data
        lo, hi = _x(chart)
        assert (lo, hi) == pytest.approx((45.5, 65.5))  # moved back 20 frames, same span
        visible = _radii()[int(np.ceil(lo)) - 1:int(np.floor(hi))]
        assert _y(chart) == pytest.approx((0.0, visible.max() * cv.Y_HEADROOM))

    def test_zooming_out_stops_at_the_full_view(self, chart):
        for _ in range(10):
            _wheel(chart, 20, 20, -10)
        assert _x(chart) == pytest.approx((0.5, N + 0.5))
        assert _y(chart) == pytest.approx((0.0, _full_y()))

    def test_zooming_in_stops_at_ten_frames(self, chart):
        for _ in range(20):
            _wheel(chart, 40, 20, 10)
        lo, hi = _x(chart)
        assert hi - lo == pytest.approx(cv.MIN_X_SPAN, abs=1e-6)
        assert lo <= 40 <= hi  # still around the cursor

    def test_wheel_zooms_time_around_the_cursor(self, chart):
        _wheel(chart, 25, 30, 3)
        lo, hi = _x(chart)
        assert hi - lo < N * 0.8
        # The frame under the cursor stays where it was on screen (a quarter in).
        assert (25 - lo) / (hi - lo) == pytest.approx((25 - 0.5) / N, abs=0.01)
        assert _y(chart)[0] == 0.0  # y follows the visible frames, from zero

    def test_y_follows_a_zoom_into_the_collapse(self, chart):
        chart._plot.plotItem.vb.setXRange(56.5, 66.5, padding=0)
        assert _y(chart)[1] == pytest.approx(_radii()[56:66].max() * cv.Y_HEADROOM)

    def test_zooming_y_on_its_axis_is_kept_and_bounded(self, chart):
        _wheel(chart, 50, 20, 4, axis=1)  # wheel over the radius axis
        assert not chart.y_follows_data
        lo, hi = _y(chart)
        assert hi - lo < _full_y() and lo >= 0.0
        for _ in range(30):
            _wheel(chart, 50, 20, 10, axis=1)
        lo, hi = _y(chart)
        assert hi - lo == pytest.approx(cv.MIN_Y_SPAN, abs=1e-6) and lo >= 0.0
        for _ in range(30):
            _wheel(chart, 50, 20, -10, axis=1)
        assert _y(chart) == pytest.approx((0.0, _full_y()))


class TestReset:
    def test_reset_zoom_and_double_click_restore_the_full_view(self, chart):
        _wheel(chart, 50, 20, 6)
        _wheel(chart, 50, 20, 4, axis=1)
        chart._reset_btn.click()
        assert _x(chart) == pytest.approx((0.5, N + 0.5))
        assert _y(chart) == pytest.approx((0.0, _full_y())) and chart.y_follows_data

        _wheel(chart, 50, 20, 6)

        class _DoubleClick:
            def double(self):
                return True

        chart._on_mouse_clicked(_DoubleClick())
        assert _x(chart) == pytest.approx((0.5, N + 0.5))

    def test_a_new_frame_count_resets_the_view(self, chart):
        _wheel(chart, 50, 20, 6)
        chart.set_total_frames(40)
        assert _x(chart) == pytest.approx((0.5, 40.5))

    def test_brushing_still_works_when_zoomed(self, chart):
        _wheel(chart, 30, 20, 4)
        got = []
        chart.range_selected.connect(lambda a, b: got.append((a, b)))
        chart._plot.plotItem.vb.mouseDragEvent(_FakeDrag(chart, 24.6, 29.4))
        assert got == [(24, 28)] and chart.selected_range() == (24, 28)


class TestNoPyqtgraphMenus:
    def test_default_menus_and_auto_button_are_gone(self, chart):
        item = chart._plot.plotItem
        assert not item.vb.menuEnabled()
        assert not item.autoBtn.isVisible()

    def test_blank_area_offers_reset_zoom(self, chart):
        assert chart._context_menu_at(5.0, 55.0) is None  # no point, no selection there
        menu = chart._view_menu()
        assert [a.text() for a in menu.actions()] == ["Reset zoom"]
        _wheel(chart, 50, 20, 6)
        menu.actions()[0].trigger()
        assert _x(chart) == pytest.approx((0.5, N + 0.5))


class TestMenusNeverDelete:
    def test_point_menu_is_one_fix_entry(self, chart):
        got = []
        chart.fix_requested.connect(got.append)
        menu = chart._context_menu_at(12.0, float(_radii()[11]))
        texts = [a.text() for a in menu.actions() if not a.isSeparator()]
        assert texts == ["Fix frame 12…"]
        menu.actions()[0].trigger()
        assert got == [11]

    def test_range_menu_has_no_delete(self, chart):
        chart.set_selection(4, 8)
        menu = chart._context_menu_at(6.0, 20.0)
        texts = [a.text() for a in menu.actions() if not a.isSeparator()]
        assert texts == ["Refit range (frames 5–9)", "Lock range (frames 5–9)",
                         "Unlock range (frames 5–9)", "Mark as no bubble (frames 5–9)",
                         "Clear selection"]
        assert not any("Delete" in t for t in texts)

    def test_the_old_delete_and_refit_signals_are_gone(self, chart):
        for name in ("delete_requested", "refit_requested", "neighbor_refit_requested",
                     "delete_range_requested"):
            assert not hasattr(chart, name)


class TestHeight:
    def test_tall_enough_and_mostly_plot(self, qapp):
        style = (Path(bubbletrack.__file__).parent / "resources" / "style.qss").read_text(
            encoding="utf-8")
        qapp.setStyleSheet(style)  # the app's button styles set their own minimum heights
        try:
            c = RadiusChart()
            c.resize(900, CHART_MIN_HEIGHT)
            c.show()
            qapp.processEvents()
            assert c.minimumHeight() == CHART_MIN_HEIGHT >= 260
            assert _vb(c).height() >= 0.7 * c.height()  # the plot, not margins
            assert c._select_btn.height() <= 26  # a compact legend row
            c.close()
        finally:
            qapp.setStyleSheet("")
