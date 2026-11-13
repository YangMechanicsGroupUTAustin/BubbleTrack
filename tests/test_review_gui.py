"""Review queue driven through the real MainWindow + AppController offscreen."""

from __future__ import annotations

import time

import numpy as np
import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QShortcut
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QComboBox, QDockWidget, QSpinBox

from bubbletrack.controller.review_controller import build_review_items, install_review
from bubbletrack.controller.review_segments import build_segments, queue_frames
from bubbletrack.model.provenance import FitSource
from bubbletrack.ui.shortcuts import FIX_BINDINGS, REVIEW_BINDINGS

N_FRAMES = 8


@pytest.fixture
def review(gui, make_sequence):
    window, controller = gui
    controller.file_ctrl.on_folder_selected(str(make_sequence("A", N_FRAMES)))
    rc = install_review(window, controller)
    return window, controller, rc


def _fitted(controller, conf=0.9):
    """Every frame fitted (smooth radii) and scored with *conf*."""
    state = controller._get_state()
    state.radius[:] = np.linspace(30.0, 40.0, len(state.radius))
    state.circle_fit_par[:] = [[125.0, 200.0]] * len(state.radius)
    state.fit_source[:] = FitSource.AUTO
    state.fit_confidence[:] = conf
    return state


def _flag(controller, rc, **conf_by_frame):
    """Give frames 'f<i>=c' a low confidence c, then refresh the queue."""
    state = _fitted(controller)
    for key, value in conf_by_frame.items():
        state.fit_confidence[int(key[1:])] = value
    rc.refresh()
    return state


def _show(qapp, window):
    window.show()
    window.activateWindow()
    assert QTest.qWaitForWindowActive(window, 5000)
    qapp.processEvents()


def _shown(controller) -> int:
    return controller._get_state().image_no


class TestInstall:
    def test_dock_and_view_menu_toggle(self, qapp, review):
        window, _, rc = review
        _show(qapp, window)
        docks = [d for d in window.findChildren(QDockWidget) if d.windowTitle() == "Review"]
        assert docks == [rc.dock]
        view = [a for a in window.menuBar().actions() if a.text().replace("&", "") == "View"]
        assert len(view) == 1
        toggle = next(a for a in view[0].menu().actions() if a.text() == "&Review Queue")
        assert toggle.isCheckable() and toggle.isChecked()
        toggle.trigger()
        assert rc.dock.isHidden()
        toggle.trigger()
        assert not rc.dock.isHidden()

    def test_installed_once_at_startup(self, gui):
        window, controller = gui
        assert install_review(window, controller) is controller.review
        view = [a for a in window.menuBar().actions() if a.text().replace("&", "") == "View"]
        assert len(view) == 1
        assert [a.text() for a in view[0].menu().actions()].count("&Review Queue") == 1

    def test_empty_until_something_is_suspicious(self, review):
        _, controller, rc = review
        assert rc.count == 0
        _fitted(controller)
        rc.refresh()
        assert rc.count == 0

    def test_refresh_lists_flagged_frames_in_plain_words(self, review):
        _, controller, rc = review
        counts = []
        rc.count_changed.connect(counts.append)
        state = _fitted(controller)
        state.fit_confidence[2] = 0.2
        state.fit_flags[5] = frozenset({"non_circular"})
        rc.refresh()
        assert rc.panel.frames() == [2, 5]
        assert rc.panel.texts() == ["Frame 3 · _003 · Not sure: low confidence",
                                    "Frame 6 · _006 · Outline not round"]
        assert counts == [2]

    def test_locked_frames_are_excluded(self, review):
        _, controller, rc = review
        state = _fitted(controller)
        state.fit_confidence[[1, 4]] = 0.1
        state.fit_locked[4] = True
        rc.refresh()
        assert rc.panel.frames() == [1]

    def test_legacy_jump_is_listed(self, review):
        _, controller, rc = review
        state = _fitted(controller, conf=np.nan)
        state.radius[3] = 60.0
        rc.refresh()
        assert rc.panel.texts() == ["Frame 4 · _004 · Radius jumps"]

    def test_set_extra_merges_and_new_source_resets(self, review, make_sequence):
        from bubbletrack.controller.review_controller import ReviewItem

        _, controller, rc = review
        _fitted(controller)
        rc.set_extra([ReviewItem(6, ("disagree",), 85)])
        assert rc.panel.frames() == [6]
        controller.file_ctrl.on_folder_selected(str(make_sequence("B", 4)))
        assert rc.count == 0


class TestActions:
    def test_selecting_a_row_shows_its_frame(self, review):
        _, controller, rc = review
        _flag(controller, rc, f2=0.3, f5=0.1)
        rc.panel.step(1)
        assert rc.panel.current_frame() == 5  # lowest confidence first
        assert _shown(controller) == 5

    def test_accept_locks_the_frame_keeps_its_source_and_moves_on(self, review):
        _, controller, rc = review
        state = _flag(controller, rc, f2=0.1, f5=0.3)
        rc.panel.step(1)
        rc.perform("accept")
        assert state.fit_locked[2] and state.fit_source[2] == FitSource.AUTO
        assert rc.panel.frames() == [5]
        assert rc.panel.current_frame() == 5 == _shown(controller)

    def test_accept_is_undone_by_ctrl_z(self, review):
        _, controller, rc = review
        state = _flag(controller, rc, f2=0.1, f5=0.3)
        rc.panel.step(1)
        rc.perform("accept")
        controller._on_undo()
        assert not state.fit_locked[2] and 2 in rc.panel.frames()

    def test_no_bubble_clears_and_locks_the_frame_and_undo_restores(self, review):
        _, controller, rc = review
        state = _flag(controller, rc, f3=0.2, f6=0.4)
        rc.panel.step(1)
        rc.perform("no_bubble")
        assert state.radius[3] == -1.0 and state.fit_locked[3]
        assert rc.panel.frames() == [6] and _shown(controller) == 6
        controller._on_undo()
        assert state.radius[3] > 0 and not state.fit_locked[3]

    def test_skip_drops_the_row_and_changes_nothing(self, review):
        _, controller, rc = review
        state = _flag(controller, rc, f1=0.1, f6=0.2)
        before = state.radius.copy()
        rc.panel.step(1)
        rc.perform("skip")
        np.testing.assert_array_equal(state.radius, before)
        assert not state.fit_locked.any()
        assert rc.panel.frames() == [6] and _shown(controller) == 6

    def test_review_never_deletes(self, review):
        assert not {"delete", "refit", "refit_neighbors", "manual"} & set(
            review[2].panel.ACTIONS)
        with pytest.raises(ValueError):
            review[2].perform("delete")

    def test_navigating_off_the_queue_clears_the_row(self, review):
        window, controller, rc = review
        state = _flag(controller, rc, f2=0.1)
        rc.panel.step(1)
        controller.file_ctrl.on_frame_changed(7)
        assert rc.panel.current_frame() is None
        rc.perform("accept_all")  # a row action: no row, nothing happens
        assert not state.fit_locked.any()
        window.frame_scrubber.value_changed.emit(2)  # back onto a queued frame
        assert rc.panel.current_frame() == 2

    def test_frame_actions_work_on_any_frame_on_screen(self, review):
        _, controller, rc = review
        state = _flag(controller, rc, f2=0.1)
        controller.file_ctrl.on_frame_changed(7)  # not flagged, in no row
        assert rc.panel.card.isVisibleTo(rc.panel)
        assert rc.panel.card.title.text() == "Frame 8 · _008"
        assert rc.panel.card.advice.text().startswith("Not flagged.")
        assert rc.panel.card.buttons["accept_all"].isHidden()
        rc.perform("no_bubble")
        assert state.radius[7] == -1.0 and state.fit_locked[7]
        assert _shown(controller) == 7  # no row: nothing to move on to

    def test_buttons_drive_the_same_actions(self, review):
        _, controller, rc = review
        state = _flag(controller, rc, f2=0.1, f5=0.3)
        rc.panel.step(1)
        rc.panel.card.buttons["accept"].click()
        assert state.fit_locked[2] and _shown(controller) == 5

    def test_edits_outside_the_queue_refresh_it(self, review):
        window, controller, rc = review
        _flag(controller, rc, f2=0.1)
        controller.auto_ctrl.on_auto_clear()  # Clear All Results wipes everything
        assert rc.count == 0

    def test_session_load_rebuilds_the_queue(self, review, make_sequence, tmp_path):
        _, controller, rc = review
        state = _flag(controller, rc, f3=0.1)
        state.fit_flags[6] = frozenset({"non_circular"})
        path = tmp_path / "s.brt"
        assert controller.save_session_file(str(path))
        controller.file_ctrl.on_folder_selected(str(make_sequence("B", 4)))
        assert rc.count == 0
        assert controller.load_session_file(str(path))
        assert rc.panel.frames() == [3, 6]


class TestCardAndFeedback:
    def test_the_recommended_action_is_the_filled_button(self, review):
        _, controller, rc = review
        state = _flag(controller, rc, f3=0.2)
        state.fit_flags[5] = frozenset({"jump"})
        rc.refresh()
        controller.file_ctrl.on_frame_changed(3)
        card = rc.panel.card
        assert card.recommended() == "accept"
        assert [b.text() for b in (card.buttons["accept"], card.buttons["fix"],
                                   card.buttons["no_bubble"])] == [
            "Accept this frame", "Fix with a circle", "No bubble here"]
        assert card.advice.text() == ("Low confidence (0.20). "
                                      "Accept it if the outline sits on the bubble wall.")
        controller.file_ctrl.on_frame_changed(5)
        assert card.recommended() == "fix"

    def test_feedback_says_what_happened_and_undo_reverts_it(self, review):
        _, controller, rc = review
        state = _flag(controller, rc, f2=0.1, f5=0.3)
        rc.panel.step(1)
        rc.perform("accept")
        feedback = rc.panel.feedback
        assert feedback.text() == "✓ Frame 3 accepted and locked. That row is done."
        assert feedback.offers_undo()
        feedback._undo.click()
        assert not state.fit_locked[2] and _shown(controller) == 2
        assert feedback.isHidden()

    def test_undo_of_a_skip_brings_the_row_back(self, review):
        _, controller, rc = review
        _flag(controller, rc, f1=0.1, f6=0.2)
        rc.panel.step(1)
        rc.perform("skip")
        assert rc.panel.feedback.text().startswith("✓ Frame 2 skipped")
        rc.panel.feedback._undo.click()
        assert rc.panel.frames() == [1, 6] and _shown(controller) == 1

    def test_progress_counts_down_then_all_done_offers_export(self, review):
        window, controller, rc = review
        _flag(controller, rc, f2=0.1, f5=0.3)
        rc.panel.reset_progress()
        assert rc.panel._summary.text() == "2 frames to check"
        rc.panel.step(1)
        rc.perform("accept")
        assert rc.panel._summary.text() == "1 of 2 frames to check"
        rc.perform("accept")
        assert rc.panel._summary.text() == "All 2 checked"
        page = rc.panel.done_page
        assert page.text().startswith("All done") and page.offers_export()
        assert rc.panel.feedback.text().endswith("All done.")
        exported = []
        window.export_requested.disconnect()  # not the real (modal) Export Results dialog
        window.export_requested.connect(lambda: exported.append(1))
        page._export.click()
        assert exported == [1]


class TestRunLock:
    def test_actions_disabled_while_a_run_is_active(self, qapp, review):
        window, controller, rc = review
        _show(qapp, window)
        state = _flag(controller, rc, f2=0.1)
        rc.panel.step(1)
        controller.bus.emit("run_state_changed", True)
        assert not rc.panel.isEnabled()
        assert not any(sc.isEnabled() for sc in rc.shortcuts.shortcuts.values())
        rc.perform("accept")
        assert not state.fit_locked[2]
        controller.bus.emit("run_state_changed", False)
        assert rc.panel.isEnabled()
        assert all(sc.isEnabled() for sc in rc.shortcuts.shortcuts.values())

    def test_real_fit_all_locks_then_refreshes(self, qapp, review):
        window, controller, rc = review
        controller.auto_ctrl.on_auto_fit()
        assert not rc.panel.isEnabled()
        deadline = time.monotonic() + 90
        while controller.auto_ctrl.is_busy():
            assert time.monotonic() < deadline, "Fit All did not finish"
            qapp.processEvents()
            time.sleep(0.01)
        assert rc.panel.isEnabled()
        state = controller._get_state()
        assert (state.radius > 0).any()
        assert rc.panel.frames() == queue_frames(build_segments(build_review_items(state)))


class TestKeys:
    def test_review_keys_do_not_collide_with_existing_shortcuts(self, review):
        window, _, _ = review
        keys = [sc.key().toString() for sc in window.findChildren(QShortcut)]
        keys += [a.shortcut().toString() for a in window.findChildren(QAction)
                 if not a.shortcut().isEmpty()]
        assert len(keys) == len(set(keys))
        assert set(REVIEW_BINDINGS) | set(FIX_BINDINGS) <= set(keys)

    def test_the_bindings(self):
        assert REVIEW_BINDINGS == {"A": "accept", "Shift+A": "accept_all", "F": "fix",
                                   "N": "no_bubble", "Shift+N": "no_bubble_all", "S": "skip",
                                   "J": "next", "K": "prev"}

    def test_j_k_navigate_and_a_accepts(self, qapp, review):
        window, controller, rc = review
        _show(qapp, window)
        state = _flag(controller, rc, f1=0.1, f4=0.2, f7=0.3)  # three single-frame rows
        rc.panel._list.setFocus(Qt.FocusReason.MouseFocusReason)  # clicked into the dock
        for key, expected in ((Qt.Key.Key_J, 1), (Qt.Key.Key_J, 4), (Qt.Key.Key_K, 1)):
            QTest.keyClick(rc.panel._list, key)
            assert _shown(controller) == expected
        QTest.keyClick(rc.panel._list, Qt.Key.Key_A)
        assert state.fit_locked[1]
        assert rc.panel.frames() == [4, 7] and _shown(controller) == 4
        QTest.keyClick(rc.panel._list, Qt.Key.Key_N)
        assert state.radius[4] == -1.0 and state.fit_locked[4] and _shown(controller) == 7

    def test_shift_keys_act_on_the_whole_row(self, qapp, review):
        window, controller, rc = review
        _show(qapp, window)
        state = _flag(controller, rc, f2=0.1, f3=0.2, f4=0.15, f7=0.4)
        rc.panel._list.setFocus(Qt.FocusReason.MouseFocusReason)
        QTest.keyClick(rc.panel._list, Qt.Key.Key_J)
        assert rc.panel.current_segment().frames == (2, 3, 4)
        QTest.keyClick(rc.panel._list, Qt.Key.Key_A, Qt.KeyboardModifier.ShiftModifier)
        assert state.fit_locked[2:5].all() and _shown(controller) == 7
        QTest.keyClick(rc.panel._list, Qt.Key.Key_N, Qt.KeyboardModifier.ShiftModifier)
        assert state.radius[7] == -1.0 and state.fit_locked[7]

    @pytest.mark.parametrize("widget_type", [QSpinBox, QComboBox])
    def test_keys_do_nothing_while_a_field_has_focus(self, qapp, review, widget_type):
        window, controller, rc = review
        _show(qapp, window)
        state = _flag(controller, rc, f2=0.1)
        rc.panel.step(1)
        window.left_panel.measure_panel.edge_box.advanced._toggle.click()  # its Bubble combo
        field = next(w for w in window.findChildren(widget_type)
                     if w.isVisible() and w.isEnabled())
        field.setFocus()
        qapp.processEvents()
        for key in (Qt.Key.Key_A, Qt.Key.Key_N):
            QTest.keyClick(field, key)
        assert not state.fit_locked[2] and state.radius[2] > 0

    def test_keys_are_off_while_the_dock_is_hidden(self, qapp, review):
        window, controller, rc = review
        _show(qapp, window)
        state = _flag(controller, rc, f2=0.1)
        rc.panel.step(1)
        rc.dock.hide()
        qapp.processEvents()
        assert not any(sc.isEnabled() for sc in rc.shortcuts.shortcuts.values())
        QTest.keyClick(window, Qt.Key.Key_A)
        assert not state.fit_locked[2]
