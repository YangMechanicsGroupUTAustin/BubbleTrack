"""Key-frame filmstrip: selection, cached thumbnails with outlines, wiring."""

from __future__ import annotations

import time

import numpy as np
import pytest

from PyQt6.QtCore import QSize
from PyQt6.QtGui import QPixmap

from bubbletrack.controller.filmstrip_controller import (
    install_filmstrip, make_thumb_base, render_thumbnail,
)
from bubbletrack.controller.key_frames import (
    MAX_TO_CHECK, choose_key_frames, find_collapses, key_frame_tooltip, to_check_frames,
)
from bubbletrack.controller.review_queue import ReviewItem
from bubbletrack.controller.review_segments import ReviewSegment
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

    def test_frames_to_check_come_from_review_most_likely_wrong_first(self):
        r = _two_cycles()
        keys = choose_key_frames(r, to_check=[40, 2, 57, 33, 10])
        checked = [k.frame for k in keys if "to check" in k.kinds]
        assert MAX_TO_CHECK == 4
        assert checked == [2, 33, 40, 57]  # the first four Review rows, shown in frame order
        assert 10 not in [k.frame for k in keys]
        assert len(keys) <= 12

    def test_without_review_rows_nothing_is_to_check(self):
        keys = choose_key_frames(_two_cycles())
        kinds = {kind for k in keys for kind in k.kinds}
        assert kinds <= {"first", "Rmax", "collapse", "last"}  # no "flagged", no "low score"

    def test_a_key_frame_in_review_is_also_to_check(self):
        r = _two_cycles()
        peak = int(np.argmax(r))
        kinds = {k.frame: k.kinds for k in choose_key_frames(r, to_check=[peak])}
        assert kinds[peak] == ("Rmax", "to check")

    def test_one_frame_per_review_row_to_check_in_queue_order(self):
        def row(category, severity, *frames):
            return ReviewSegment(category, tuple(ReviewItem(f, ("x",), severity) for f in frames))
        rows = [row("no_result", 90, 0, 1, 2), row("jump", 60, 111),
                row("collapse", 50, 65, 66, 67)]  # red, amber, grey
        assert to_check_frames(rows) == [0, 111]  # grey rows are for information
        assert to_check_frames([]) == []


class TestTooltips:
    def test_plain_words_and_no_engine_flags(self):
        item = ReviewItem(0, ("rejected", "guess", "low_confidence"), 99, 0.13)
        text = key_frame_tooltip(0, "_001", ("to check",), -1.0, 0.13, item)
        assert text.splitlines() == [
            "Frame 1 · _001 — to check",
            "No result · confidence 0.13",
            "To check: no result, Auto has a guess, low confidence 0.13",
        ]

    def test_a_fitted_key_frame(self):
        text = key_frame_tooltip(11, "_012", ("Rmax",), 51.62, float("nan"), None)
        assert text.splitlines() == ["Frame 12 · _012 — largest radius (Rmax)", "R = 51.6 px"]
        text = key_frame_tooltip(3, "", ("first", "to check"), 33.3, 0.82,
                                 ReviewItem(3, ("non_circular",), 55, 0.82))
        assert text.splitlines() == ["Frame 4 — first fit + to check",
                                     "R = 33.3 px · confidence 0.82",
                                     "To check: outline not round"]
        for internal in ("flags", "non_circular", "R = -1"):
            assert internal not in text


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

    def test_hint_says_to_check(self, qapp):
        strip = Filmstrip()
        hint = strip._hint.text()
        assert "to check" in hint and "flagged" not in hint

    def test_thumbnails_shrink_to_fit_the_width(self, qapp):
        strip = Filmstrip()
        pix = QPixmap(THUMB_MAX_WIDTH, THUMB_HEIGHT)
        # Short captions: the offscreen test font is much wider than the app's.
        strip.set_thumbnails([Thumbnail(i, pix, f"c{i}") for i in range(7)])
        strip.resize(720, strip.sizeHint().height())
        strip.show()
        qapp.processEvents()
        assert not strip._scroll.horizontalScrollBar().isVisible()
        used = sum(b.sizeHint().width() for b in strip.buttons()) + 6 * 6
        assert used <= strip._scroll.viewport().width()
        assert all(b.iconSize().width() < THUMB_MAX_WIDTH for b in strip.buttons())

        strip.resize(1600, strip.height())  # room to spare: full size
        qapp.processEvents()
        assert all(b.iconSize() == QSize(THUMB_MAX_WIDTH, THUMB_HEIGHT) for b in strip.buttons())

        strip.resize(200, strip.height())  # far too narrow: half size, then it scrolls
        qapp.processEvents()
        assert all(b.iconSize().width() == THUMB_MAX_WIDTH // 2 for b in strip.buttons())
        strip.close()


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
    controller.review.refresh()  # as after a run: Review lists the new results


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
        rows = to_check_frames(controller.review.panel.segments())
        expected = [k.frame for k in choose_key_frames(controller._get_state().radius, rows)]
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


class TestToCheckFollowsReview:
    """The strip's "to check" frames are the Review queue's, in plain words."""

    def _fitted(self, gui, make_sequence):
        window, controller = gui
        controller.file_ctrl.on_folder_selected(str(make_sequence("seq", 30, r_min=15, r_max=60)))
        _fill_results(controller, _two_cycles(30) + 10)
        state = controller._get_state()
        state.fit_confidence[:] = 0.95
        state.fit_flags[7] = frozenset({"saturation"})  # an engine note only: not in Review
        state.radius[20] = -1.0  # no result: Review lists it
        state.circle_fit_par[20] = [np.nan, np.nan]
        controller.review.refresh()
        controller.filmstrip.refresh()
        return window, controller

    @staticmethod
    def _to_check(window) -> list[int]:
        return [int(b.property("frame")) for b in window.filmstrip.buttons()
                if b.text().startswith("to check")]

    def test_only_frames_in_the_review_queue(self, gui, make_sequence):
        window, controller = self._fitted(gui, make_sequence)
        assert controller.review.queued_frames() == {20}
        assert self._to_check(window) == [20]
        assert 7 not in window.filmstrip.frames()
        button = window.filmstrip.buttons()[window.filmstrip.frames().index(20)]
        assert button.text() == "to check · 21"
        assert button.toolTip().splitlines()[1:] == [
            "No result · confidence 0.95", "To check: no result"]
        for b in window.filmstrip.buttons():
            assert "flags" not in b.toolTip() and "saturation" not in b.toolTip()

    def test_a_settled_frame_leaves_the_strip(self, gui, make_sequence, qapp):
        window, controller = self._fitted(gui, make_sequence)
        controller._get_state().fit_locked[20] = True  # e.g. "No bubble here" in Review
        controller.review.refresh()  # the queue shrinks: the strip follows by itself
        assert _wait(qapp, lambda: self._to_check(window) == [])
