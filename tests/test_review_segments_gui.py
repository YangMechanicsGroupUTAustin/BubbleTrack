"""Review rows that span several frames, driven through the real window offscreen."""

from __future__ import annotations

import numpy as np
import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest

from bubbletrack.controller.review_controller import ReviewItem, install_review
from bubbletrack.model.provenance import FitSource
from bubbletrack.ui.review_panel import EMPTY_ALL_CLEAR, EMPTY_BEFORE_FIT, EMPTY_FILTERED
from bubbletrack.ui.review_rows import ROW_ROLE

N_FRAMES = 10


@pytest.fixture
def review(gui, make_sequence):
    window, controller = gui
    controller.file_ctrl.on_folder_selected(str(make_sequence("A", N_FRAMES)))
    return window, controller, install_review(window, controller)


def _flag(controller, rc, low=(), shape=(), conf=0.2):
    """All frames fitted and confident, except *low* (low confidence) and
    *shape* (not circular); then refresh the queue."""
    state = controller._get_state()
    state.radius[:] = np.linspace(30.0, 40.0, len(state.radius))
    state.circle_fit_par[:] = [[125.0, 200.0]] * len(state.radius)
    state.fit_source[:] = FitSource.AUTO
    state.fit_confidence[:] = 0.9
    state.fit_confidence[list(low)] = conf
    for i in shape:
        state.fit_flags[i] = frozenset({"non_circular"})
    rc.refresh()
    return state


def _shown(controller) -> int:
    return controller._get_state().image_no


class TestRows:
    def test_consecutive_frames_with_a_one_frame_gap_are_one_row(self, review):
        _, controller, rc = review
        _flag(controller, rc, low=(2, 3, 5))
        (segment,) = rc.panel.segments()
        assert (segment.start, segment.end, segment.frames) == (2, 5, (2, 3, 5))
        assert rc.panel.texts() == ["Frames 3–6 · Low confidence · 3 frames"]
        assert rc.panel.frames() == [2, 3, 5] and rc.count == 3

    def test_header_tooltip_and_badge(self, review):
        _, controller, rc = review
        _flag(controller, rc, low=(2, 3, 5), shape=(8,))
        assert rc.panel._summary.text() == "2 segments · 4 frames"
        item = rc.panel._list.item(0)
        assert item.data(ROW_ROLE).badge == "3"
        assert "Why:" in item.toolTip() and "←/→" in item.toolTip()
        assert rc.panel._list.item(1).data(ROW_ROLE).badge == ""  # one frame: no badge

    def test_dock_is_not_wider_than_before(self, review):
        panel = review[2].panel
        assert panel.minimumWidth() <= 260
        before = panel.minimumSizeHint().width()  # set by the buttons
        # A long header (huge queue, filtered) must not push the dock wider.
        panel._summary.setText("1234 segments · 12345 frames")
        panel._filter_btn.setText("3 of 7 types ▾")
        panel._filter_btn.setVisible(True)
        assert panel.minimumSizeHint().width() <= before


class TestSelection:
    def test_a_row_shows_its_first_frame_and_its_span_on_the_chart(self, review):
        window, controller, rc = review
        _flag(controller, rc, low=(2, 3, 5), shape=(8,), conf=0.05)
        rc.panel.step(1)
        assert _shown(controller) == 2 and rc.panel.current_frame() == 2
        assert window.radius_chart.selected_range() == (2, 5)
        assert rc.panel._detail.text().startswith("Frame 3")
        rc.panel.step(1)  # a single frame: no span to show
        assert _shown(controller) == 8
        assert window.radius_chart.selected_range() is None

    def test_a_range_the_user_brushed_is_left_alone(self, review):
        window, controller, rc = review
        _flag(controller, rc, low=(4,))
        rc.panel.step(1)
        window.radius_chart.set_selection(0, 7)  # brushed by the user
        controller.file_ctrl.on_frame_changed(9)  # off the queue
        assert rc.panel.current_segment() is None
        assert window.radius_chart.selected_range() == (0, 7)

    def test_stepping_through_the_span_keeps_the_row(self, review):
        _, controller, rc = review
        _flag(controller, rc, low=(2, 3, 5))
        rc.panel.step(1)
        segment = rc.panel.current_segment()
        controller.file_ctrl.on_frame_changed(4)  # the unflagged frame inside
        assert rc.panel.current_segment() == segment and rc.panel.current_frame() == 4
        assert "not flagged" in rc.panel._detail.text()
        controller.file_ctrl.on_frame_changed(6)  # past the end
        assert rc.panel.current_segment() is None

    def test_a_frame_of_another_type_inside_the_span_shows_its_own_reasons(self, review):
        _, controller, rc = review
        state = _flag(controller, rc, low=(4,), conf=0.3)
        rc.set_extra([ReviewItem(f, ("lost",), 0) for f in (3, 5)])
        rc.panel.step(1)  # Frames 4-6, lost track
        assert rc.panel.current_segment().frames == (3, 5)
        controller.file_ctrl.on_frame_changed(4)
        assert rc.panel.current_segment().frames == (3, 5)  # the row is kept
        assert "low confidence 0.30" in rc.panel._detail.text()
        assert state.fit_confidence[4] == pytest.approx(0.3)

    def test_scrubbing_onto_a_flagged_frame_selects_its_row(self, review):
        window, controller, rc = review
        _flag(controller, rc, low=(2, 3, 5))
        window.frame_scrubber.value_changed.emit(3)
        assert rc.panel.current_segment().frames == (2, 3, 5)
        assert window.radius_chart.selected_range() == (2, 5)


class TestRowActions:
    def test_accept_locks_every_frame_of_the_row_and_moves_on(self, review):
        _, controller, rc = review
        state = _flag(controller, rc, low=(2, 3, 5), shape=(8,), conf=0.05)
        rc.panel.step(1)
        rc.perform("accept")
        assert state.fit_locked[2:6].all()  # the unflagged frame 4 too
        assert not state.fit_locked[[0, 1, 6, 7, 8, 9]].any()
        assert rc.panel.frames() == [8] and _shown(controller) == 8

    def test_accept_keeps_frames_without_a_result(self, review):
        _, controller, rc = review
        state = _flag(controller, rc)
        state.radius[[3, 4]] = -1.0  # rejected detections ("no result")
        state.fit_confidence[[3, 4]] = 0.1
        rc.refresh()
        assert rc.panel.texts() == ["Frames 4–5 · No result · 2 frames"]
        rc.panel.step(1)
        rc.perform("accept")
        assert state.fit_locked[[3, 4]].all() and rc.count == 0

    def test_refit_runs_a_range_fit_over_the_row(self, review, monkeypatch):
        _, controller, rc = review
        _flag(controller, rc, low=(2, 3, 5), shape=(8,), conf=0.05)
        calls = []

        def fit_range(start, end):  # starts a background run
            calls.append((start, end))
            controller.bus.emit("run_state_changed", True)
        monkeypatch.setattr(controller.auto_ctrl, "fit_range", fit_range)
        rc.panel.step(1)
        rc.perform("refit")
        assert calls == [(2, 5)]
        assert _shown(controller) == 8  # still listed until the fit lands; move on
        controller.bus.emit("run_state_changed", False)

    def test_a_range_refit_that_did_not_start_stays_on_the_row(self, review, monkeypatch):
        _, controller, rc = review
        _flag(controller, rc, low=(2, 3, 5), shape=(8,), conf=0.05)
        monkeypatch.setattr(controller.auto_ctrl, "fit_range", lambda s, e: None)  # cancelled
        rc.panel.step(1)
        rc.perform("refit")
        assert _shown(controller) == 2 and rc.panel.current_segment().start == 2

    def test_delete_clears_the_span(self, review):
        _, controller, rc = review
        state = _flag(controller, rc, low=(2, 3, 5))
        rc.panel.step(1)
        rc.perform("delete")
        assert (state.radius[2:6] == -1.0).all()
        assert (state.radius[[1, 6]] > 0).all()
        assert rc.count == 0

    def test_skip_drops_the_row_until_a_result_changes(self, review):
        _, controller, rc = review
        state = _flag(controller, rc, low=(2, 3, 5), shape=(8,), conf=0.05)
        rc.panel.step(1)
        rc.perform("skip")
        assert rc.panel.frames() == [8] and not state.fit_locked.any()
        state.fit_confidence[3] = 0.3  # a new result for one frame of it
        rc.refresh()
        assert rc.panel.frames() == [3, 8]  # back on its own (it ranks first)

    def test_neighbours_refits_the_shown_frame_then_the_next_one(self, review, monkeypatch):
        _, controller, rc = review
        _flag(controller, rc, low=(2, 3, 5))
        calls = []
        monkeypatch.setattr(controller, "_on_refit_point_with_neighbors", calls.append)
        rc.panel.step(1)
        controller.file_ctrl.on_frame_changed(3)
        rc.perform("refit_neighbors")
        assert calls == [3]
        assert _shown(controller) == 5 and rc.panel.current_segment().start == 2

    def test_manual_on_an_unflagged_frame_inside_continues_after_it(self, review):
        window, controller, rc = review
        state = _flag(controller, rc, low=(2, 3, 5))
        rc.panel.step(1)
        controller.file_ctrl.on_frame_changed(4)
        rc.perform("manual")
        assert window.left_panel.tab_bar.current_name() == "Manual" and _shown(controller) == 4
        for a in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            controller.manual_ctrl.on_point_clicked(200 + 30 * np.cos(a), 125 + 30 * np.sin(a))
        window.left_panel.manual_tab.done_clicked.emit()
        assert state.fit_source[4] == FitSource.MANUAL
        assert _shown(controller) == 5

    def test_solver_items_expire_frame_by_frame(self, review):
        _, controller, rc = review
        state = _flag(controller, rc)
        rc.set_extra([ReviewItem(f, ("lost",), 0) for f in (3, 4, 5)])
        assert rc.panel.texts() == ["Frames 4–6 · Lost track · 3 frames"]
        state.radius[4] = 33.0  # frame 4 got a new result
        rc.refresh()
        assert rc.panel.segments()[0].frames == (3, 5)


class TestKeysAndFilter:
    def test_j_k_move_between_rows_and_arrows_step_inside_one(self, qapp, review):
        window, controller, rc = review
        window.show()
        window.activateWindow()
        assert QTest.qWaitForWindowActive(window, 5000)
        _flag(controller, rc, low=(2, 3, 5), shape=(8,), conf=0.05)
        rc.panel._list.setFocus(Qt.FocusReason.MouseFocusReason)
        QTest.keyClick(rc.panel._list, Qt.Key.Key_J)
        assert _shown(controller) == 2
        QTest.keyClick(rc.panel._list, Qt.Key.Key_Right)
        assert _shown(controller) == 3 and rc.panel.current_segment().start == 2
        QTest.keyClick(rc.panel._list, Qt.Key.Key_J)
        assert _shown(controller) == 8
        QTest.keyClick(rc.panel._list, Qt.Key.Key_K)
        assert _shown(controller) == 2

    def test_type_filter_hides_rows(self, review):
        _, controller, rc = review
        _flag(controller, rc, low=(2,), shape=(6,))
        panel = rc.panel
        assert panel._filter_btn.text() == "All types ▾"
        panel._fill_filter_menu()
        labels = [a.text() for a in panel._filter_menu.actions() if a.isCheckable()]
        assert labels == ["Low confidence  (1)", "Irregular shape  (1)"]
        next(a for a in panel._filter_menu.actions() if a.text().startswith("Irregular")).toggle()
        assert panel.frames() == [2] and rc.count == 2  # hidden, still waiting
        assert panel._filter_btn.text() == "1 of 2 types ▾"
        panel.set_hidden_categories({"low_confidence", "shape"})
        assert panel._empty.text() == EMPTY_FILTERED
        panel.set_hidden_categories(())
        assert panel.frames() == [2, 6]

    def test_empty_messages(self, review):
        _, controller, rc = review
        assert rc.panel._empty.text() == EMPTY_BEFORE_FIT
        assert rc.panel._summary.text() == "Nothing to review"
        _flag(controller, rc)
        assert rc.panel._empty.text() == EMPTY_ALL_CLEAR

    def test_clicking_the_selected_row_goes_back_to_its_first_frame(self, review):
        _, controller, rc = review
        _flag(controller, rc, low=(2, 3, 5))
        rc.panel.step(1)
        controller.file_ctrl.on_frame_changed(5)
        rc.panel._list.clicked.emit(rc.panel._list.model().index(0, 0))
        assert _shown(controller) == 2


def test_an_unchanged_queue_is_not_rebuilt(review):
    _, controller, rc = review
    state = _flag(controller, rc)
    state.fit_confidence[:] = np.nan  # legacy mode: unscored frames
    state.radius[4] = 80.0  # a jump
    rc.refresh()
    item = rc.panel._list.item(0)
    controller.file_ctrl.on_frame_changed(4)  # refreshes again
    assert rc.panel._list.item(0) is item
