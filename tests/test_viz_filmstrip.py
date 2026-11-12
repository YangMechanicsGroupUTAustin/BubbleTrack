"""Key-frame filmstrip: selection, cached thumbnails with outlines, wiring."""

from __future__ import annotations

import time

import numpy as np
import pytest

from bubbletrack.controller.filmstrip_controller import (
    _spread, choose_key_frames, find_collapses, install_filmstrip, make_thumb_base,
    render_thumbnail,
)
from bubbletrack.model.provenance import FitSource
from bubbletrack.model.state import update_state
from bubbletrack.ui.filmstrip import THUMB_HEIGHT, THUMB_MAX_WIDTH, Filmstrip, Thumbnail


def _two_cycles(n: int = 60) -> np.ndarray:
    """Expand, collapse at ~frame 25, rebound, collapse at ~frame 45, settle."""
    t = np.arange(n)
    r = 70 * np.abs(np.sin(np.pi * t / 25)) * np.exp(-t / 60) + 6
    return r + np.random.default_rng(1).normal(0, 0.3, n)


class TestKeyFrameChoice:
    def test_first_rmax_collapses_last(self):
        r = _two_cycles()
        keys = choose_key_frames(r)
        kinds = {k.frame: k.kinds for k in keys}
        assert kinds[0] == ("first",)
        assert kinds[int(np.argmax(r))] == ("Rmax",)
        assert kinds[len(r) - 1] == ("last",)
        collapses = [f for f, k in kinds.items() if "collapse" in k]
        assert collapses == find_collapses(r)
        assert [abs(c - e) <= 1 for c, e in zip(collapses, (25, 50))] == [True, True]
        assert [k.frame for k in keys] == sorted(kinds)

    def test_noise_is_not_a_collapse(self):
        r = 50 + np.random.default_rng(0).normal(0, 0.5, 40)
        assert find_collapses(r) == []

    def test_unfitted_frames_are_skipped(self):
        r = _two_cycles()
        r[:5] = -1
        r[-3:] = np.nan
        keys = choose_key_frames(r)
        assert keys[0].frame == 5 and keys[-1].frame == len(r) - 4

    def test_nothing_fitted(self):
        assert choose_key_frames(np.full(10, -1.0)) == []

    def test_up_to_four_flagged_frames_spread_out(self):
        r = _two_cycles()
        flags = [frozenset({"x"}) if i % 5 == 2 else frozenset() for i in range(len(r))]
        keys = choose_key_frames(r, flags)
        flagged = [k.frame for k in keys if "flagged" in k.kinds]
        assert len(flagged) == 4
        assert flagged[0] == 2 and flagged[-1] == 57  # first and last flagged kept
        assert len(keys) <= 12

    def test_low_scores_stand_in_when_nothing_is_flagged(self):
        r = _two_cycles()
        scores = np.full(len(r), 0.9)
        scores[[7, 33, 40]] = [0.2, 0.1, 0.3]
        keys = choose_key_frames(r, [frozenset()] * len(r), scores)
        low = [k.frame for k in keys if "low score" in k.kinds]
        assert low == [7, 33, 40]

    def test_spread(self):
        assert _spread([1, 2, 3], 4) == [1, 2, 3]
        assert _spread(list(range(10)), 3) == [0, 4, 9]  # linspace(0, 9, 3) rounds 4.5 to 4
        assert _spread([5], 0) == []


class TestThumbnails:
    def test_roi_crop_and_scale(self):
        gray = np.zeros((500, 800), np.uint8)
        base = make_thumb_base(gray, (101, 300), (201, 600))  # 200 x 400 ROI
        assert base.gray.shape[0] <= THUMB_HEIGHT and base.gray.shape[1] <= THUMB_MAX_WIDTH
        assert base.origin == (100, 200)
        assert base.scale == pytest.approx(min(THUMB_HEIGHT / 200, THUMB_MAX_WIDTH / 400))

    def test_outline_is_drawn(self, qapp):
        base = make_thumb_base(np.full((100, 100), 128, np.uint8), (1, 100), (1, 100))
        t = np.linspace(0, 2 * np.pi, 80, endpoint=False)
        contour = np.column_stack([50 + 30 * np.sin(t), 50 + 30 * np.cos(t)])
        pix = render_thumbnail(base, (50.0, 50.0, 30.0), contour)
        img = pix.toImage()
        x = int(round(80 * base.scale))
        y = int(round(50 * base.scale))
        colours = {img.pixelColor(x + dx, y).name() for dx in (-1, 0, 1)}
        assert any(c != "#808080" for c in colours)  # something drawn on the rim
        assert img.pixelColor(int(50 * base.scale), int(50 * base.scale)).name() == "#808080"


class TestFilmstripWidget:
    def test_buttons_jump_and_mark_current(self, qapp):
        from PyQt6.QtGui import QPixmap
        strip = Filmstrip()
        pix = QPixmap(40, 30)
        strip.set_thumbnails([Thumbnail(3, pix, "first · 4"), Thumbnail(9, pix, "Rmax · 10")])
        assert strip.frames() == [3, 9]
        assert "(2)" in strip._toggle.text()
        got = []
        strip.frame_clicked.connect(got.append)
        strip.buttons()[1].click()
        assert got == [9]
        assert [b.isChecked() for b in strip.buttons()] == [False, True]
        strip.set_thumbnails([])
        assert strip.frames() == [] and not strip._placeholder.isHidden()

    def test_collapse(self, qapp):
        strip = Filmstrip()
        seen = []
        strip.expanded_changed.connect(seen.append)
        strip.set_expanded(False)
        assert not strip.is_expanded() and strip._scroll.isHidden() and seen == [False]


def _fill_results(controller, radii) -> None:
    state = controller._get_state()
    n = len(state.images)
    state.radius[:] = radii[:n]
    state.circle_fit_par[:] = [125.0, 200.0]
    t = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    for i in range(n):
        r = state.radius[i]
        state.circle_xy[i] = np.column_stack([125 + r * np.sin(t), 200 + r * np.cos(t)])
    state.fit_source[:] = FitSource.AUTO
    controller._state = update_state(state)


def _wait(qapp, predicate, timeout_s=5.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        qapp.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


class TestInstalledFilmstrip:
    def _setup(self, gui, make_sequence, n=30):
        window, controller = gui
        ctrl = install_filmstrip(window, controller)
        controller.file_ctrl.on_folder_selected(str(make_sequence("seq", n, r_min=15, r_max=60)))
        return window, controller, ctrl

    def test_installed_above_the_chart_and_empty_before_fitting(self, gui, make_sequence):
        window, controller, ctrl = self._setup(gui, make_sequence)
        strip = window.filmstrip
        assert isinstance(strip, Filmstrip) and strip.controller is ctrl
        assert window.content_splitter.widget(1).isAncestorOf(strip)
        assert strip.frames() == []

    def test_refresh_after_a_run_then_click_jumps(self, gui, make_sequence, qapp):
        window, controller, ctrl = self._setup(gui, make_sequence)
        _fill_results(controller, _two_cycles(30) + 10)
        controller.bus.emit("run_state_changed", True)
        controller.bus.emit("run_state_changed", False)
        strip = window.filmstrip
        expected = [k.frame for k in choose_key_frames(controller._get_state().radius)]
        assert strip.frames() == expected
        assert ctrl.cache.loads == len(expected)

        target = strip.buttons()[-1]
        target.click()
        assert controller._get_state().image_no == int(target.property("frame"))
        assert window.frame_scrubber.value() == int(target.property("frame"))

    def test_thumbnails_are_cached(self, gui, make_sequence):
        window, controller, ctrl = self._setup(gui, make_sequence)
        _fill_results(controller, _two_cycles(30) + 10)
        ctrl.refresh()
        loads = ctrl.cache.loads
        ctrl.refresh()
        ctrl.refresh()
        assert ctrl.cache.loads == loads  # no disk reads for unchanged key frames

    def test_unchanged_results_are_not_rebuilt(self, gui, make_sequence):
        window, controller, ctrl = self._setup(gui, make_sequence)
        _fill_results(controller, _two_cycles(30) + 10)
        ctrl.refresh()
        first = window.filmstrip.buttons()
        controller.file_ctrl.on_frame_changed(4)  # navigating changes no result
        ctrl.refresh()
        assert window.filmstrip.buttons() == first
        assert [b.isChecked() for b in first].count(True) <= 1

        controller._get_state().radius[5] = 99.0  # e.g. a refit: new Rmax
        ctrl.refresh()
        assert 5 in window.filmstrip.frames()

    def test_frame_change_is_debounced(self, gui, make_sequence, qapp):
        window, controller, ctrl = self._setup(gui, make_sequence)
        _fill_results(controller, _two_cycles(30) + 10)
        calls = []
        original = ctrl.refresh
        ctrl.refresh = lambda: (calls.append(1), original())
        ctrl._timer.timeout.disconnect()
        ctrl._timer.timeout.connect(ctrl.refresh)
        for i in range(5):
            controller.file_ctrl.on_frame_changed(i)
        assert calls == []
        assert _wait(qapp, lambda: len(calls) == 1)
        time.sleep(0.3)
        qapp.processEvents()
        assert len(calls) == 1
        assert window.filmstrip.frames()  # refreshed with the new results

    def test_new_source_clears_cache_and_strip(self, gui, make_sequence):
        window, controller, ctrl = self._setup(gui, make_sequence)
        _fill_results(controller, _two_cycles(30) + 10)
        ctrl.refresh()
        assert window.filmstrip.frames()
        controller.file_ctrl.on_folder_selected(str(make_sequence("other", 5)))
        assert window.filmstrip.frames() == []
        assert len(ctrl.cache._items) == 0

    def test_no_refresh_while_running(self, gui, make_sequence, qapp):
        window, controller, ctrl = self._setup(gui, make_sequence)
        controller.bus.emit("run_state_changed", True)
        _fill_results(controller, _two_cycles(30) + 10)
        ctrl.schedule_refresh()
        assert not ctrl._timer.isActive()
        controller.bus.emit("run_state_changed", False)
        assert window.filmstrip.frames()

    def test_collapsed_strip_skips_work_until_expanded(self, gui, make_sequence):
        window, controller, ctrl = self._setup(gui, make_sequence)
        window.filmstrip.set_expanded(False)
        _fill_results(controller, _two_cycles(30) + 10)
        ctrl.refresh()
        assert ctrl.cache.loads == 0
        window.filmstrip.set_expanded(True)
        assert window.filmstrip.frames() and ctrl.cache.loads > 0
