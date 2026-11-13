"""Review rows that span several frames, driven through the real window offscreen."""

from __future__ import annotations

import numpy as np
import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest

from bubbletrack.controller.review_controller import ReviewItem, install_review
from bubbletrack.model.provenance import FitSource
from bubbletrack.ui.review_panel import (
    EMPTY_ALL_CLEAR, EMPTY_ALL_DONE, EMPTY_BEFORE_FIT, EMPTY_FILTERED,
)
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


def _points(controller, radius=30.0, centre=(200.0, 125.0)):
    for a in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        controller.manual_ctrl.on_point_clicked(centre[0] + radius * np.cos(a),
                                                centre[1] + radius * np.sin(a))


class TestRows:
    def test_consecutive_frames_with_a_one_frame_gap_are_one_row(self, review):
        _, controller, rc = review
        _flag(controller, rc, low=(2, 3, 5))
        (segment,) = rc.panel.segments()
        assert (segment.start, segment.end, segment.frames) == (2, 5, (2, 3, 5))
        assert rc.panel.texts() == ["Frames 3–6 · Not sure: low confidence · 3 frames"]
        assert rc.panel.frames() == [2, 3, 5] and rc.count == 3

    def test_header_tooltip_and_badge(self, review):
        _, controller, rc = review
        _flag(controller, rc, low=(2, 3, 5), shape=(8,))
        assert rc.panel._summary.text() == "4 frames to check"
        item = rc.panel._list.item(0)
        assert item.data(ROW_ROLE).badge == "3"
        assert "Why:" in item.toolTip() and "←/→" in item.toolTip()
        assert rc.panel._list.item(1).data(ROW_ROLE).badge == ""  # one frame: no badge

    def test_dock_is_not_wider_than_before(self, review):
        panel = review[2].panel
        assert panel.minimumWidth() <= 260
        before = panel.minimumSizeHint().width()
        # A long header (huge queue, filtered) must not push the dock wider.
        panel._summary.setText("12345 of 23456 frames to check")
        panel.kind_filter.setText("3 of 7 kinds ▾")
        panel.kind_filter.setVisible(True)
        assert panel.minimumSizeHint().width() <= max(before, 260)


class TestSelection:
    def test_a_row_shows_its_first_frame_and_its_span_on_the_chart(self, review):
        window, controller, rc = review
        chart = window.radius_chart
        _flag(controller, rc, low=(2, 3, 5), shape=(8,), conf=0.05)
        rc.panel.step(1)
        assert _shown(controller) == 2 and rc.panel.current_frame() == 2
        assert chart.review_span() == (2, 5)
        assert rc.panel.card.title.text().startswith("Frame 3")
        assert rc.panel.card.position.text() == "3 left in this row"
        rc.panel.step(1)  # a single frame: no span to show
        assert _shown(controller) == 8
        assert chart.review_span() is None

    def test_the_span_is_not_a_chart_selection(self, review):
        window, controller, rc = review
        chart = window.radius_chart
        _flag(controller, rc, low=(2, 3, 5), shape=(8,), conf=0.05)
        rc.panel.step(1)
        assert chart.selected_range() is None  # right-click keeps the point menu
        assert not chart._in_selection(4.0)  # frame 4 (1-based) inside the span
        chart.set_selection(0, 9)  # brushed by the user
        rc.panel.step(1)
        rc.panel.step(-1)
        assert chart.selected_range() == (0, 9) and chart.review_span() == (2, 5)

    def test_no_span_while_the_dock_is_closed(self, qapp, review):
        window, controller, rc = review
        window.show()
        qapp.processEvents()
        _flag(controller, rc, low=(2, 3, 5))
        rc.panel.step(1)
        toggle = rc.dock.toggleViewAction()  # View > Review Queue
        toggle.trigger()
        assert rc.dock.isHidden() and window.radius_chart.review_span() is None
        toggle.trigger()
        assert window.radius_chart.review_span() == (2, 5)

    def test_stepping_through_the_span_keeps_the_row(self, review):
        _, controller, rc = review
        _flag(controller, rc, low=(2, 3, 5))
        rc.panel.step(1)
        segment = rc.panel.current_segment()
        controller.file_ctrl.on_frame_changed(4)  # the unflagged frame inside
        assert rc.panel.current_segment() == segment and rc.panel.current_frame() == 4
        assert rc.panel.card.advice.text().startswith("Not flagged.")
        assert rc.panel.card.position.text() == "3 left in this row"
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
        assert rc.panel.card.advice.text().startswith("Low confidence (0.30).")
        assert state.fit_confidence[4] == pytest.approx(0.3)

    def test_scrubbing_onto_a_flagged_frame_selects_its_row(self, review):
        window, controller, rc = review
        _flag(controller, rc, low=(2, 3, 5))
        window.frame_scrubber.value_changed.emit(3)
        assert rc.panel.current_segment().frames == (2, 3, 5)
        assert window.radius_chart.review_span() == (2, 5)

    def test_a_rebuild_keeps_the_row_while_its_span_holds_the_frame(self, review):
        _, controller, rc = review
        _flag(controller, rc, low=(4,), shape=(8,), conf=0.3)
        rc.set_extra([ReviewItem(f, ("lost",), 0) for f in (3, 5)])
        rc.panel.step(1)  # lost track 3-5
        controller.file_ctrl.on_frame_changed(4)  # a low-confidence frame inside
        rc.panel.set_hidden_categories({"shape"})  # an unrelated change rebuilds the list
        assert rc.panel.current_segment().frames == (3, 5)

    def test_after_a_row_is_skipped_the_next_row_is_selected(self, review):
        _, controller, rc = review
        _flag(controller, rc, low=(4, 6, 8), conf=0.2)
        rc.set_extra([ReviewItem(5, ("lost",), 0), ReviewItem(7, ("jump",), 0)])
        assert [s.category for s in rc.panel.segments()] == ["lost", "jump", "low_confidence"]
        rc.panel.step(1)
        rc.perform("skip")  # lost 5 goes: the jump at 7 is next
        assert _shown(controller) == 7
        assert rc.panel.current_segment().category == "jump"

    def test_j_after_leaving_a_row_continues_from_it(self, review):
        _, controller, rc = review
        _flag(controller, rc, low=(1, 4, 7))
        rc.panel.step(1)
        rc.panel.step(1)  # the row of frame 4 (rows: 1, 4, 7)
        controller.file_ctrl.on_frame_changed(5)  # stepped past it
        assert rc.panel.current_segment() is None
        rc.panel.step(1)
        assert _shown(controller) == 7


class TestMovingOn:
    def test_accept_goes_to_the_rows_next_frame_then_to_the_next_row(self, review):
        _, controller, rc = review
        state = _flag(controller, rc, low=(2, 3, 5), shape=(8,), conf=0.05)
        rc.panel.step(1)
        rc.perform("accept")
        assert state.fit_locked[2] and not state.fit_locked[[3, 5]].any()
        assert _shown(controller) == 3 and rc.panel.card.position.text() == "2 left in this row"
        rc.perform("accept")
        assert _shown(controller) == 5
        rc.perform("accept")  # the row is done: the next row
        assert _shown(controller) == 8 and rc.panel.current_segment().frames == (8,)
        assert rc.panel.feedback.text().endswith("That row is done.")

    def test_likely_wrong_rows_above_come_before_the_next_amber_row(self, review):
        """Done with a red row low in the list: back to the red rows above, not
        on to the amber ones below them."""
        _, controller, rc = review
        state = _flag(controller, rc, low=(1, 4), conf=0.05)  # red rows: 1, 4
        state.fit_confidence[7] = 0.4  # an amber row: 7
        rc.refresh()
        assert [s.tier for s in rc.panel.segments()] == ["high", "high", "medium"]
        controller.file_ctrl.on_frame_changed(4)
        rc.perform("accept")
        assert _shown(controller) == 1
        rc.perform("accept")
        assert _shown(controller) == 7

    def test_a_row_started_in_the_middle_wraps_to_its_first_frames(self, review):
        _, controller, rc = review
        _flag(controller, rc, low=(2, 3, 5), shape=(8,), conf=0.05)
        controller.file_ctrl.on_frame_changed(5)
        rc.perform("no_bubble")
        assert _shown(controller) == 2  # 2 and 3 of the row still need a look

    def test_fixing_an_unflagged_frame_inside_continues_after_it(self, review):
        window, controller, rc = review
        state = _flag(controller, rc, low=(2, 3, 5))
        rc.panel.step(1)
        controller.file_ctrl.on_frame_changed(4)
        rc.perform("fix")
        rc.fix.strip._points_btn.click()  # "Click edge points instead"
        assert rc.fix.active and window.original_panel.mode == "point"
        _points(controller)
        rc.fix.strip._done_btn.click()
        assert state.fit_source[4] == FitSource.MANUAL
        assert _shown(controller) == 5 and not rc.fix.active


class TestRowActions:
    def test_accept_all_locks_every_frame_of_the_row_and_moves_on(self, review):
        _, controller, rc = review
        state = _flag(controller, rc, low=(2, 3, 5), shape=(8,), conf=0.05)
        rc.panel.step(1)
        rc.perform("accept_all")
        assert state.fit_locked[[2, 3, 5]].all()  # "Accept all 3": exactly those
        assert not state.fit_locked[[0, 1, 4, 6, 7, 8, 9]].any()
        assert rc.panel.frames() == [8] and _shown(controller) == 8
        controller._on_undo()  # one Ctrl+Z undoes the whole Accept all
        assert not state.fit_locked.any()

    def test_accept_all_keeps_frames_without_a_result(self, review):
        _, controller, rc = review
        state = _flag(controller, rc)
        state.radius[[3, 4]] = -1.0  # rejected detections ("no result")
        state.fit_confidence[[3, 4]] = 0.1
        rc.refresh()
        assert rc.panel.texts() == ["Frames 4–5 · No result · 2 frames"]
        rc.panel.step(1)
        assert rc.panel.card.buttons["accept_all"].text() == "Accept all 2"
        rc.perform("accept_all")
        assert state.fit_locked[[3, 4]].all() and rc.count == 0

    def test_no_bubble_on_all_marks_the_rows_frames_only(self, review):
        _, controller, rc = review
        state = _flag(controller, rc, low=(2, 3, 5))
        rc.panel.step(1)
        rc.perform("no_bubble_all")
        assert (state.radius[[2, 3, 5]] == -1.0).all() and state.fit_locked[[2, 3, 5]].all()
        assert (state.radius[[1, 4, 6]] > 0).all() and not state.fit_locked[4]
        assert rc.count == 0
        controller._on_undo()
        assert (state.radius[2:6] > 0).all() and not state.fit_locked.any()

    def test_row_actions_leave_another_rows_frame_inside_the_span_alone(self, review):
        _, controller, rc = review
        state = _flag(controller, rc, low=(2, 4), conf=0.4)  # an amber row 2-4 ...
        state.fit_flags[3] = frozenset({"jump"})  # ... around a red "Radius jumps" frame
        rc.refresh()
        controller.file_ctrl.on_frame_changed(2)
        assert rc.panel.current_segment().frames == (2, 4)
        rc.perform("accept_all")
        assert state.fit_locked[[2, 4]].all() and not state.fit_locked[3]
        assert 3 in rc.panel.frames()  # still waiting for its own look

    def test_skip_drops_the_row_until_a_result_changes(self, review):
        _, controller, rc = review
        state = _flag(controller, rc, low=(2, 3, 5), shape=(8,), conf=0.05)
        rc.panel.step(1)
        rc.perform("skip")
        assert rc.panel.frames() == [8] and not state.fit_locked.any()
        state.fit_confidence[3] = 0.3  # a new result for one frame of it
        rc.refresh()
        assert rc.panel.frames() == [3, 8]  # back on its own (it ranks first)

    def test_the_more_menu_exports_the_row_as_a_hard_case(self, review, monkeypatch):
        _, controller, rc = review
        _flag(controller, rc, low=(2, 3, 5))
        rc.panel.step(1)
        asked = []
        monkeypatch.setattr(controller.hard_cases, "ask_and_export",
                            lambda start, end: asked.append((start, end)))
        rc.panel.card._more_menu.actions()[-1].trigger()
        assert asked == [(2, 5)]

    def test_solver_items_expire_frame_by_frame(self, review):
        _, controller, rc = review
        state = _flag(controller, rc)
        rc.set_extra([ReviewItem(f, ("lost",), 0) for f in (3, 4, 5)])
        assert rc.panel.texts() == ["Frames 4–6 · Auto lost the bubble · 3 frames"]
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
        assert panel.kind_filter.text() == "All kinds ▾"
        panel.kind_filter.fill_menu()
        labels = [a.text() for a in panel.kind_filter.menu_.actions() if a.isCheckable()]
        assert labels == ["Not sure: low confidence  (1)", "Outline not round  (1)"]
        next(a for a in panel.kind_filter.menu_.actions()
             if a.text().startswith("Outline")).toggle()
        assert panel.frames() == [2] and rc.count == 2  # hidden, still waiting
        assert panel.kind_filter.text() == "1 of 2 kinds ▾"
        panel.set_hidden_categories({"low_confidence", "shape"})
        assert panel.done_page.text() == EMPTY_FILTERED and not panel.done_page.offers_export()
        panel.set_hidden_categories(())
        assert panel.frames() == [2, 6]

    def test_empty_messages(self, review):
        _, controller, rc = review
        page = rc.panel.done_page
        assert page.text() == EMPTY_BEFORE_FIT and not page.offers_export()
        assert rc.panel._summary.text() == "Nothing to check yet"
        _flag(controller, rc)
        assert page.text() == EMPTY_ALL_CLEAR and page.offers_export()
        _flag(controller, rc, low=(3,))
        rc.panel.step(1)
        rc.perform("accept")
        assert page.text() == EMPTY_ALL_DONE and page.offers_export()

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
