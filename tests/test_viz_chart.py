"""R(t) chart: marker encoding, cursor, Rmax window, cycles, series, range selection."""

from __future__ import annotations

import time

import numpy as np
import pyqtgraph as pg
import pytest
from PyQt6.QtCore import QPointF, Qt

from bubbletrack.controller.display_mixin import refresh_chart, rmax_window
from bubbletrack.model.export import locate_rmax
from bubbletrack.model.provenance import FitSource
from bubbletrack.model.state import AppState, update_state
from bubbletrack.ui import chart_style as cs
from bubbletrack.ui.chart_interaction import hover_text, nearest_point
from bubbletrack.ui.chart_overlays import frames_in_span
from bubbletrack.ui.radius_chart import RadiusChart


# -- pure helpers ----------------------------------------------------------------

class TestMarkerEncoding:
    def test_ramp_runs_from_red_poor_to_green_good(self):
        assert cs.ramp_colour(0.0) == cs.RAMP_STOPS[-1]
        assert cs.ramp_colour(1.0) == cs.RAMP_STOPS[0]
        assert cs.ramp_colour(float("nan")) == cs.UNSCORED_COLOUR

        def rgb(hex_colour):
            return tuple(int(hex_colour[i:i + 2], 16) for i in (1, 3, 5))

        red, green = rgb(cs.ramp_colour(0.0)), rgb(cs.ramp_colour(1.0))
        assert red[0] > 200 and red[1] < 100      # poor = red
        assert green[1] > 150 and green[0] < 100  # good = green
        # The red share falls steadily as the score rises.
        share = [rgb(cs.ramp_colour(s))[0] / max(sum(rgb(cs.ramp_colour(s))), 1)
                 for s in np.linspace(0, 1, 11)]
        assert all(b <= a + 1e-9 for a, b in zip(share, share[1:]))

    def test_bins(self):
        bins = cs.ramp_bins(np.array([0.0, 1.0, np.nan, 2.0, -1.0]))
        assert bins.tolist() == [0, cs.RAMP_BINS - 1, -1, cs.RAMP_BINS - 1, 0]

    def test_two_shapes_computed_or_decided_by_the_user(self):
        sources = np.array([FitSource.AUTO, FitSource.MANUAL, FitSource.REFIT,
                            FitSource.NEIGHBOR, FitSource.TRACKED, FitSource.AUTO])
        locked = np.array([False, False, False, False, False, True])
        symbols = cs.marker_symbols(sources, locked, 6)
        assert symbols.tolist() == ["o", "d", "o", "o", "o", "d"]
        assert cs.marker_symbols(None, None, 2).tolist() == ["o", "o"]
        assert [name for _, name in cs.SHAPE_LEGEND] == ["computed", "manual / locked"]

    def test_hover_text(self):
        text = hover_text(44, 16.39, file_tag="_054", score=0.25,
                          source=int(FitSource.MANUAL), flags=frozenset({"b", "a"}))
        assert text == "Frame 44 · _054  R 16.4 px  Q 25%  manual  flags: a, b"
        assert hover_text(3, 10.0, locked=True, source=1) == "Frame 3  R 10.0 px  locked"

    def test_nearest_point(self):
        frames, radii = np.array([1.0, 5.0, 9.0]), np.array([10.0, 20.0, 30.0])
        view = ((0.0, 10.0), (0.0, 40.0))
        assert nearest_point(frames, radii, view, 5.1, 20.3) == 1
        assert nearest_point(frames, radii, view, 3.0, 35.0) is None

    @pytest.mark.parametrize("x0, x1, total, span", [
        (2.4, 6.6, 10, (3, 6)),
        (6.6, 2.4, 10, (3, 6)),
        (-5, 50, 10, (1, 10)),
        (3.2, 3.8, 10, None),
    ])
    def test_frames_in_span(self, x0, x1, total, span):
        assert frames_in_span(x0, x1, total) == span


# -- widget ------------------------------------------------------------------------

@pytest.fixture
def chart(qapp):
    c = RadiusChart()
    c.resize(800, 300)
    c.show()
    qapp.processEvents()
    yield c
    c.close()


def _series(n=20):
    frames = np.arange(1, n + 1)
    radii = 40 * np.sin(np.linspace(0.1, np.pi - 0.1, n)) + 5
    return frames, radii


class TestPlotAll:
    def test_markers_carry_source_score_and_flags(self, chart):
        frames, radii = _series(6)
        radii[2] = -1  # unfitted: not drawn
        sources = np.array([1, 2, 1, 3, 4, 1], dtype=np.int8)
        locked = np.array([0, 0, 0, 0, 0, 1], dtype=bool)
        flags = [frozenset(), frozenset({"fragmented"}), frozenset(), frozenset(),
                 frozenset({"low_contrast"}), frozenset()]
        scores = np.array([0.9, 0.1, 0.5, np.nan, 0.6, 1.0])
        chart.plot_all(frames, radii, scores, sources=sources, flags=flags, locked=locked)

        data = chart._scatter.data
        assert data["x"].tolist() == [1, 2, 4, 5, 6]
        assert list(data["symbol"]) == ["o", "d", "o", "o", "d"]
        bins = cs.RAMP_BINS - 1
        assert data["brush"][1].color().name() == cs.ramp_colour(round(0.1 * bins) / bins)
        assert data["brush"][2].color().name() == cs.UNSCORED_COLOUR  # NaN score
        assert not hasattr(chart, "_flag_rings")  # flags live in the hover text and Review

    def test_legacy_call_without_extras(self, chart):
        frames, radii = _series(5)
        chart.plot_all(frames, radii)
        assert len(chart._scatter.data) == 5
        assert set(chart._scatter.data["symbol"]) == {"o"}

    def test_data_changed_emitted(self, chart):
        hits = []
        chart.data_changed.connect(lambda: hits.append(1))
        chart.plot_all(*_series(4))
        assert hits == [1]

    def test_refresh_under_0_3_s_for_2000_points(self, chart):
        n = 2000
        frames, radii = _series(n)
        rng = np.random.default_rng(0)
        scores = rng.random(n)
        sources = rng.integers(0, 5, n).astype(np.int8)
        locked = rng.random(n) < 0.05
        flags = [frozenset({"x"}) if v < 0.03 else frozenset() for v in rng.random(n)]
        chart.set_total_frames(n)
        chart.plot_all(frames, radii, scores, sources=sources, flags=flags, locked=locked)
        t0 = time.perf_counter()
        for _ in range(3):
            chart.plot_all(frames, radii, scores, sources=sources, flags=flags, locked=locked)
            chart.set_current_frame(500)
        chart.grab()  # include the paint
        assert (time.perf_counter() - t0) / 3 < 0.3


class TestAnnotations:
    def test_cursor_follows_current_frame(self, chart):
        chart.plot_all(*_series(10))
        chart.set_current_frame(4)
        assert chart._overlays.cursor.isVisible()
        assert chart._overlays.cursor.value() == 5  # x axis is 1-based
        assert chart.current_frame() == 4
        assert chart._highlight.data["x"].tolist() == [5]
        chart.set_current_frame(None)
        assert not chart._overlays.cursor.isVisible()

    def test_rmax_band_and_t0(self, chart):
        chart.set_rmax_info(10.4, 6, 15)
        t0, band = chart.rmax_info()
        assert t0 == pytest.approx(10.4)
        assert band == pytest.approx((5.5, 15.5))
        chart.set_rmax_info(None)
        assert chart.rmax_info() is None

    def test_cycles(self, chart):
        chart.set_cycles([9, 30])
        assert chart.cycle_frames() == [9, 30]
        assert chart._overlays.cycle_positions() == [10.0, 31.0]
        chart.set_cycles([])
        assert chart.cycle_frames() == []

    def test_plot_series_raw_faint_solved_solid(self, chart):
        frames = np.arange(1, 6)
        raw = np.array([10, -1, 12, 13, np.nan])
        solved = np.array([10.5, 11.0, 12.0, -1, 14.0])
        chart.plot_series(frames, raw, solved)
        assert chart._raw_series.data["x"].tolist() == [1, 3, 4]
        x, y = chart._solved_series.getData()
        assert np.isnan(y[3]) and y[0] == 10.5  # gaps break the line
        chart.clear_series()
        assert len(chart._raw_series.data) == 0

    def test_export_view_hides_transient_items(self, chart):
        chart.plot_all(*_series(10))
        chart.highlight_frame(3)  # legacy 1-based API moves the cursor too
        assert chart.current_frame() == 2
        chart.set_selection(1, 4)
        with chart.export_view() as plot_item:
            assert plot_item is chart._plot.plotItem
            assert not chart._overlays.cursor.isVisible()
            assert not chart._overlays.selection.isVisible()
        assert chart._overlays.cursor.isVisible() and chart._overlays.selection.isVisible()

    def test_review_span_is_shaded_but_not_a_selection(self, chart):
        chart.plot_all(*_series(10))
        chart.set_review_span((2, 5))  # 0-based frames 2..5
        assert chart.review_span() == (2, 5) and chart.selected_range() is None
        assert tuple(chart._overlays.review_span.getRegion()) == (2.5, 6.5)
        assert not chart._in_selection(4.0)  # its points keep the point menu
        with chart.export_view():
            assert not chart._overlays.review_span.isVisible()
        assert chart._overlays.review_span.isVisible()
        chart.set_review_span(None)
        assert chart.review_span() is None

    def test_clear_removes_everything(self, chart):
        chart.plot_all(*_series(5))
        chart.set_current_frame(1)
        chart.set_rmax_info(3, 1, 5)
        chart.set_cycles([2])
        chart.set_selection(1, 3)
        chart.set_review_span((0, 2))
        chart.clear()
        assert len(chart._frames) == 0 and chart.rmax_info() is None
        assert chart.cycle_frames() == [] and chart.selected_range() is None
        assert chart.review_span() is None


class _FakeDrag:
    """Just enough of pyqtgraph's MouseDragEvent for ViewBox.mouseDragEvent."""

    def __init__(self, chart, x0, x1, *, shift=True, finish=True):
        vb = chart._plot.plotItem.vb
        self._down = vb.mapViewToScene(QPointF(x0, 20))
        self._pos = vb.mapViewToScene(QPointF(x1, 20))
        self._item_down = pg.Point(vb.mapFromScene(self._down))
        self._item_pos = pg.Point(vb.mapFromScene(self._pos))
        self._shift, self._finish, self.accepted = shift, finish, False

    def pos(self):
        return self._item_pos

    def lastPos(self):  # noqa: N802
        return self._item_down

    def buttonDownPos(self, _button=None):  # noqa: N802
        return self._item_down

    def modifiers(self):
        return Qt.KeyboardModifier.ShiftModifier if self._shift else Qt.KeyboardModifier.NoModifier

    def button(self):
        return Qt.MouseButton.LeftButton

    def buttonDownScenePos(self):  # noqa: N802
        return self._down

    def scenePos(self):  # noqa: N802
        return self._pos

    def isFinish(self):  # noqa: N802
        return self._finish

    def isStart(self):  # noqa: N802
        return False

    def accept(self):
        self.accepted = True

    def ignore(self):
        pass


class TestRangeSelection:
    def _ready(self, chart):
        chart.set_total_frames(20)
        chart.plot_all(*_series(20))
        chart._plot.setXRange(0.5, 20.5, padding=0)
        chart._plot.setYRange(0, 50, padding=0)

    def test_shift_drag_selects_frames(self, chart):
        self._ready(chart)
        got = []
        chart.range_selected.connect(lambda a, b: got.append((a, b)))
        ev = _FakeDrag(chart, 4.6, 9.4)
        chart._plot.plotItem.vb.mouseDragEvent(ev)
        assert ev.accepted
        assert got == [(4, 8)]  # 0-based frames 5..9
        assert chart.selected_range() == (4, 8)
        assert chart._overlays.selection.isVisible()

    def test_plain_drag_pans_unless_select_mode(self, chart):
        self._ready(chart)
        got = []
        chart.range_selected.connect(lambda a, b: got.append((a, b)))
        chart._plot.plotItem.vb.mouseDragEvent(_FakeDrag(chart, 4.6, 9.4, shift=False, finish=False))
        assert got == [] and chart.selected_range() is None
        chart.set_select_mode(True)
        chart._plot.plotItem.vb.mouseDragEvent(_FakeDrag(chart, 2.6, 3.4, shift=False))
        assert got == [(2, 2)]

    def test_range_menu_emits_range_signals(self, chart):
        self._ready(chart)
        chart.set_selection(4, 8)
        seen = {}
        names = ("delete_range_requested", "refit_range_requested", "lock_range_requested",
                 "unlock_range_requested")
        for name in names:
            getattr(chart, name).connect(lambda a, b, n=name: seen.setdefault(n, (a, b)))
        menu = chart._context_menu_at(6.0, 20.0)
        actions = [a for a in menu.actions() if not a.isSeparator()]
        assert [a.text() for a in actions] == [
            "Delete range (frames 5–9)", "Refit range (frames 5–9)",
            "Lock range (frames 5–9)", "Unlock range (frames 5–9)", "Clear selection"]
        for action in actions[:4]:
            action.trigger()
        assert seen == {n: (4, 8) for n in names}
        actions[4].trigger()
        assert chart.selected_range() is None

    def test_point_menu_outside_selection(self, chart):
        self._ready(chart)
        frames, radii = _series(20)
        got = []
        chart.refit_requested.connect(got.append)
        menu = chart._context_menu_at(frames[11], radii[11])
        [a for a in menu.actions() if a.text().startswith("Refit frame")][0].trigger()
        assert got == [11]

    def test_no_edit_menus_while_a_fit_runs(self, chart):
        self._ready(chart)
        chart.set_selection(4, 8)
        chart.set_editing_enabled(False)
        assert chart._context_menu_at(6.0, 20.0) is None


# -- wiring through display_mixin -------------------------------------------------

def _fitted_state(n=30) -> AppState:
    state = update_state(AppState.create_empty(), images=tuple(f"f_{i:03d}.tif" for i in range(n)),
                         rmax_fit_length=7).with_results_initialized(n)
    state.radius[:] = 40 * np.sin(np.linspace(0.05, np.pi - 0.05, n)) + 5
    state.radius[3] = -1
    state.circle_fit_par[:] = [100.0, 100.0]
    state.fit_source[:] = FitSource.AUTO
    state.fit_source[7] = FitSource.MANUAL
    state.fit_locked[9] = True
    state.fit_flags[12] = frozenset({"fragmented"})
    return state


class TestChartWiring:
    def test_refresh_chart_passes_provenance_and_rmax(self, gui):
        window, _ = gui
        state = _fitted_state()
        refresh_chart(state, window)
        chart = window.radius_chart
        symbols = dict(zip(chart._scatter.data["x"].tolist(), chart._scatter.data["symbol"]))
        assert symbols[8] == "d" and symbols[10] == "d" and symbols[1] == "o"
        est = locate_rmax(state.radius, state.rmax_fit_length)
        t0, band = chart.rmax_info()
        assert t0 == pytest.approx(est.frame)
        assert band == pytest.approx((est.window[0] - 0.5, est.window[1] + 0.5))

    def test_edge_mode_names_the_ramp_confidence(self, gui):
        window, _ = gui
        state = update_state(_fitted_state(), detection_mode="edge")
        state.fit_confidence[:] = 0.8
        refresh_chart(state, window)
        assert window.radius_chart._score_name.text().startswith("confidence")

    def test_no_valid_frames_clears_rmax(self, gui):
        window, _ = gui
        state = _fitted_state()
        refresh_chart(state, window)
        state.radius[:] = -1
        assert rmax_window(state) is None
        refresh_chart(state, window)
        assert window.radius_chart.rmax_info() is None

    def test_frame_change_moves_the_cursor(self, gui, make_sequence):
        window, controller = gui
        controller.file_ctrl.on_folder_selected(str(make_sequence("seq", 5)))
        assert window.radius_chart.current_frame() == 0
        controller.file_ctrl.on_frame_changed(3)
        assert window.radius_chart.current_frame() == 3
