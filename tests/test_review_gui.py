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
from bubbletrack.model.provenance import FitSource
from bubbletrack.ui.shortcuts import REVIEW_BINDINGS

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

    def test_refresh_lists_flagged_frames_and_emits_count(self, review):
        _, controller, rc = review
        counts = []
        rc.count_changed.connect(counts.append)
        state = _fitted(controller)
        state.fit_confidence[2] = 0.2
        state.fit_flags[5] = frozenset({"non_circular"})
        rc.refresh()
        assert rc.panel.frames() == [2, 5]
        assert rc.panel.texts()[0] == "Frame 3 (file _003) · low confidence 0.20"
        assert rc.panel.texts()[1] == "Frame 6 (file _006) · non_circular"
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
        assert rc.panel.texts() == ["Frame 4 (file _004) · jump"]

    def test_set_extra_merges_and_new_source_resets(self, review, make_sequence):
        from bubbletrack.controller.review_controller import ReviewItem

        _, controller, rc = review
        _fitted(controller)
        rc.set_extra([ReviewItem(6, ("disagree",), 85)])
        assert rc.panel.frames() == [6]
        controller.file_ctrl.on_folder_selected(str(make_sequence("B", 4)))
        assert rc.count == 0


class TestActions:
    def test_selecting_an_item_shows_its_frame(self, review):
        _, controller, rc = review
        _flag(controller, rc, f2=0.3, f5=0.1)
        rc.panel.step(1)
        assert rc.panel.current_frame() == 5  # lowest confidence first
        assert controller._get_state().image_no == 5

    def test_accept_locks_keeps_source_and_advances(self, review):
        _, controller, rc = review
        state = _flag(controller, rc, f2=0.1, f5=0.3)
        rc.panel.step(1)
        rc.perform("accept")
        assert state.fit_locked[2] and state.fit_source[2] == FitSource.AUTO
        assert rc.panel.frames() == [5]
        assert rc.panel.current_frame() == 5 == controller._get_state().image_no

    def test_delete_clears_result_and_drops_item(self, review):
        _, controller, rc = review
        state = _flag(controller, rc, f3=0.2)
        rc.panel.step(1)
        rc.perform("delete")
        assert state.radius[3] == -1.0 and state.fit_source[3] == FitSource.NONE
        assert rc.count == 0
        # A new low-confidence result for that frame is reviewed again (deleting
        # cleared the old score, so the new fit brings its own).
        state.radius[3] = 35.0
        state.fit_confidence[3] = 0.2
        rc.refresh()
        assert rc.panel.frames() == [3]

    def test_skip_drops_item_and_changes_nothing(self, review):
        _, controller, rc = review
        state = _flag(controller, rc, f1=0.1, f6=0.2)
        before = state.radius.copy()
        rc.panel.step(1)
        rc.perform("skip")
        np.testing.assert_array_equal(state.radius, before)
        assert not state.fit_locked.any()
        assert rc.panel.frames() == [6] and controller._get_state().image_no == 6

    @pytest.mark.parametrize("action, method", [
        ("refit", "_on_refit_point"),
        ("refit_neighbors", "_on_refit_point_with_neighbors"),
    ])
    def test_refit_calls_through_and_advances(self, review, monkeypatch, action, method):
        _, controller, rc = review
        _flag(controller, rc, f2=0.1, f4=0.3)
        calls = []
        monkeypatch.setattr(controller, method, calls.append)
        rc.panel.step(1)
        rc.perform(action)
        assert calls == [2]
        assert rc.panel.current_frame() == 4  # frame 2 is still listed, but we move on

    def test_manual_opens_manual_tab_in_point_mode_then_advances(self, review):
        window, controller, rc = review
        state = _flag(controller, rc, f4=0.1, f6=0.3)
        rc.panel.step(1)
        rc.perform("manual")
        assert window.left_panel.tab_bar.current_name() == "Manual"
        assert window.original_panel._mode == "point"
        assert controller._get_state().image_no == 4
        for a in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            controller.manual_ctrl.on_point_clicked(200 + 30 * np.cos(a), 125 + 30 * np.sin(a))
        window.left_panel.manual_tab.done_clicked.emit()
        assert state.fit_source[4] == FitSource.MANUAL
        assert state.radius[4] == pytest.approx(30.0, abs=0.5)
        assert rc.panel.frames() == [6]
        assert controller._get_state().image_no == 6

    def test_navigating_off_the_queue_clears_the_selection(self, review):
        window, controller, rc = review
        state = _flag(controller, rc, f2=0.1)
        rc.panel.step(1)
        controller.file_ctrl.on_frame_changed(7)
        assert rc.panel.current_frame() is None
        rc.perform("accept")
        assert not state.fit_locked.any()
        window.frame_scrubber.value_changed.emit(2)  # back onto a queued frame
        assert rc.panel.current_frame() == 2

    def test_panel_buttons_drive_the_same_actions(self, review):
        _, controller, rc = review
        state = _flag(controller, rc, f2=0.1, f5=0.3)
        rc.panel._next_btn.click()
        assert controller._get_state().image_no == 2
        rc.panel._action_buttons["accept"].click()
        assert state.fit_locked[2] and controller._get_state().image_no == 5

    def test_edits_outside_the_queue_refresh_it(self, review):
        window, controller, rc = review
        _flag(controller, rc, f2=0.1)
        window.left_panel.automatic_tab.clear_clicked.emit()  # Clear wipes all results
        assert rc.count == 0

    def test_session_load_rebuilds_the_queue(self, review, make_sequence, tmp_path):
        _, controller, rc = review
        state = _flag(controller, rc, f3=0.1)
        state.fit_flags[6] = frozenset({"fragmented"})
        path = tmp_path / "s.brt"
        assert controller.save_session_file(str(path))
        controller.file_ctrl.on_folder_selected(str(make_sequence("B", 4)))
        assert rc.count == 0
        assert controller.load_session_file(str(path))
        assert rc.panel.frames() == [3, 6]


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
        while not window.left_panel.automatic_tab._fit_btn.isEnabled():
            assert time.monotonic() < deadline, "Fit All did not finish"
            qapp.processEvents()
            time.sleep(0.01)
        assert rc.panel.isEnabled()
        state = controller._get_state()
        assert (state.radius > 0).any()
        assert rc.panel.frames() == [i.frame for i in build_review_items(state)]


class TestKeys:
    def test_review_keys_do_not_collide_with_existing_shortcuts(self, review):
        window, _, _ = review
        keys = [sc.key().toString() for sc in window.findChildren(QShortcut)]
        keys += [a.shortcut().toString() for a in window.findChildren(QAction)
                 if not a.shortcut().isEmpty()]
        assert len(keys) == len(set(keys))
        assert set(REVIEW_BINDINGS) <= set(keys)

    def test_j_k_navigate_and_a_accepts(self, qapp, review):
        window, controller, rc = review
        _show(qapp, window)
        state = _flag(controller, rc, f1=0.1, f3=0.2, f5=0.3)
        rc.panel._list.setFocus(Qt.FocusReason.MouseFocusReason)  # clicked into the dock
        for key, expected in ((Qt.Key.Key_J, 1), (Qt.Key.Key_J, 3), (Qt.Key.Key_K, 1)):
            QTest.keyClick(rc.panel._list, key)
            assert controller._get_state().image_no == expected
        QTest.keyClick(rc.panel._list, Qt.Key.Key_A)
        assert state.fit_locked[1]
        assert rc.panel.frames() == [3, 5] and controller._get_state().image_no == 3

    @pytest.mark.parametrize("widget_type", [QSpinBox, QComboBox])
    def test_keys_do_nothing_while_a_field_has_focus(self, qapp, review, widget_type):
        window, controller, rc = review
        _show(qapp, window)
        state = _flag(controller, rc, f2=0.1)
        rc.panel.step(1)
        field = next(w for w in window.left_panel.pretune_tab.findChildren(widget_type)
                     if w.isVisible() and w.isEnabled())
        field.setFocus()
        qapp.processEvents()
        for key in (Qt.Key.Key_A, Qt.Key.Key_D):
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
