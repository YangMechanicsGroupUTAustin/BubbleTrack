"""Fix (F) in the Review dock, offscreen: the fix strip, its edge-points tool,
its keys, ``start_fix`` for the chart, and how a fix lives with runs."""

from __future__ import annotations

import inspect

import numpy as np
import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest

from bubbletrack.controller import (
    circle_controller, fix_mode, manual_controller, review_actions, review_controller,
)
from bubbletrack.model.provenance import FitSource
from bubbletrack.model.state import update_state
from bubbletrack.ui.circle_tool import TOOL_POINTS

N_FRAMES = 8


@pytest.fixture
def app(gui, make_sequence):
    """Eight fitted frames (intensity mode: Fix clicks edge points); 2 and 5 flagged."""
    window, controller = gui
    controller.file_ctrl.on_folder_selected(str(make_sequence("A", N_FRAMES)))
    controller._state = update_state(controller._get_state(), detection_mode="intensity")
    state = controller._get_state()
    state.radius[:] = np.linspace(30.0, 40.0, N_FRAMES)
    state.circle_fit_par[:] = [[125.0, 200.0]] * N_FRAMES
    state.fit_source[:] = FitSource.AUTO
    state.fit_confidence[:] = 0.9
    state.fit_confidence[[2, 5]] = 0.1
    controller.review.refresh()
    return window, controller, controller.review


def _points(controller, n=8, radius=30.0):
    for a in np.linspace(0, 2 * np.pi, n, endpoint=False):
        controller.manual_ctrl.on_point_clicked(200 + radius * np.cos(a), 125 + radius * np.sin(a))


def _show(qapp, window):
    window.show()
    window.activateWindow()
    assert QTest.qWaitForWindowActive(window, 5000)
    qapp.processEvents()


class TestPointsTool:
    def test_the_strip_counts_points_and_done_waits_for_three(self, app):
        window, controller, rc = app
        rc.panel.step(1)
        rc.perform("fix")
        strip = rc.fix.strip
        assert rc.fix.tool == TOOL_POINTS and strip.frame() == 2
        assert strip._title.text() == "Fixing frame 3" and not strip._done_btn.isEnabled()
        _points(controller, n=2)
        assert strip.hint_text().endswith("2 so far") and not strip._done_btn.isEnabled()
        _points(controller, n=4, radius=31.0)  # two land on the first two: ignored
        assert strip._done_btn.isEnabled() and strip.hint_text().startswith("4 points")

    def test_done_stores_a_manual_fit_and_moves_on(self, app):
        window, controller, rc = app
        state = controller._get_state()
        rc.panel.step(1)
        rc.perform("fix")
        _points(controller)
        rc.fix.strip._done_btn.click()
        assert state.fit_source[2] == FitSource.MANUAL
        assert state.radius[2] == pytest.approx(30.0, abs=0.5)
        assert not rc.fix.active and controller._get_state().image_no == 5
        assert rc.panel.feedback.text().startswith("✓ Frame 3 fixed: R = 30.0 px")

    def test_too_few_points_keep_the_fix_going(self, app):
        window, controller, rc = app
        rc.panel.step(1)
        rc.perform("fix")
        _points(controller, n=2)
        rc.fix.finish_points()
        assert rc.fix.active and window.original_panel.mode == "point"
        assert "Need at least 3 points" in window.header.status_text()

    def test_a_refused_fit_lets_the_user_click_again(self, app):
        window, controller, rc = app
        controller._set_max_radius(10.0)  # every circle is an outlier
        rc.panel.step(1)
        rc.perform("fix")
        _points(controller)
        rc.fix.finish_points()
        assert rc.fix.active and controller.manual_ctrl.selecting
        assert "Radius outlier" in window.header.status_text()
        assert controller._get_state().fit_source[2] == FitSource.AUTO

    def test_while_fixing_the_strip_is_the_one_place_to_act(self, app):
        window, controller, rc = app
        rc.panel.step(1)
        buttons = rc.panel.card.buttons
        assert not buttons["accept"].isHidden() and not buttons["skip"].isHidden()
        rc.perform("fix")
        assert all(b.isHidden() for b in buttons.values()) and rc.panel.card.more.isHidden()
        assert rc.panel.card.advice.text()  # what is wrong stays in view
        rc.fix.cancel()
        assert not buttons["accept"].isHidden() and not buttons["skip"].isHidden()

    def test_an_empty_frame_offers_no_accept_button(self, app):
        """Nothing to accept: the card offers Fix and No bubble only."""
        window, controller, rc = app
        state = controller._get_state()
        state.radius[2], state.fit_confidence[2] = -1.0, 0.1  # rejected, no guess
        rc.refresh()
        controller.file_ctrl.on_frame_changed(2)
        card = rc.panel.card
        assert card.buttons["accept"].isHidden()
        assert card.recommended() == "fix" and not card.buttons["no_bubble"].isHidden()

    def test_no_bubble_during_a_fix_ends_it(self, app):
        window, controller, rc = app
        state = controller._get_state()
        rc.panel.step(1)
        rc.perform("fix")
        _points(controller, n=2)
        rc.perform("no_bubble")  # the user decided there is nothing to fix
        assert state.radius[2] == -1.0 and state.fit_locked[2]
        assert not rc.fix.active and not controller.manual_ctrl.selecting
        assert controller._get_state().image_no == 5

    def test_cancel_leaves_the_frame_and_stays_on_it(self, app):
        window, controller, rc = app
        state = controller._get_state()
        rc.panel.step(1)
        rc.perform("fix")
        _points(controller)
        rc.fix.strip._cancel_btn.click()
        assert not rc.fix.active and window.original_panel.mode == "normal"
        assert state.fit_source[2] == FitSource.AUTO and controller._get_state().image_no == 2
        assert "Fix cancelled; frame 3 is unchanged" in window.header.status_text()


class TestKeys:
    def test_esc_cancels_and_enter_is_done_only_while_fixing(self, qapp, app):
        window, controller, rc = app
        _show(qapp, window)
        assert not any(sc.isEnabled() for sc in rc.fix_keys.shortcuts.values())  # keys free
        rc.panel.step(1)
        rc.panel._list.setFocus(Qt.FocusReason.MouseFocusReason)
        QTest.keyClick(rc.panel._list, Qt.Key.Key_F)
        assert rc.fix.active
        assert all(sc.isEnabled() for sc in rc.fix_keys.shortcuts.values())
        _points(controller)
        QTest.keyClick(rc.panel._list, Qt.Key.Key_Return)
        assert controller._get_state().fit_source[2] == FitSource.MANUAL
        QTest.keyClick(rc.panel._list, Qt.Key.Key_F)  # now on frame 5
        assert rc.fix.frame == 5
        QTest.keyClick(rc.panel._list, Qt.Key.Key_Escape)
        assert not rc.fix.active
        assert not any(sc.isEnabled() for sc in rc.fix_keys.shortcuts.values())

    def test_hiding_the_dock_cancels_the_fix(self, qapp, app):
        window, controller, rc = app
        _show(qapp, window)
        rc.panel.step(1)
        rc.perform("fix")
        rc.dock.hide()
        qapp.processEvents()
        assert not rc.fix.active and window.original_panel.mode == "normal"


class TestStartFix:
    def test_any_frame_from_the_chart(self, app):
        window, controller, rc = app
        assert rc.start_fix(7)
        assert controller._get_state().image_no == 7 and rc.fix.frame == 7
        _points(controller)
        rc.fix.finish_points()
        assert controller._get_state().fit_source[7] == FitSource.MANUAL
        assert controller._get_state().image_no == 7  # not in the queue: no move

    def test_the_charts_fix_frame_entry_starts_it(self, app):
        window, controller, rc = app
        window.radius_chart.fix_requested.emit(6)  # a point's "Fix frame 7..."
        assert rc.fix.active and rc.fix.frame == 6 and controller._get_state().image_no == 6

    def test_it_opens_a_closed_dock(self, qapp, app):
        window, controller, rc = app
        _show(qapp, window)
        rc.dock.close()
        assert rc.start_fix(3) and not rc.dock.isHidden()

    @pytest.mark.parametrize("frame", [-1, N_FRAMES])
    def test_out_of_range(self, app, frame):
        assert not app[2].start_fix(frame) and not app[2].fix.active

    def test_not_while_fit_all_runs(self, app):
        window, controller, rc = app
        controller.auto_ctrl._running = True
        controller.bus.emit("run_state_changed", True)
        assert not rc.start_fix(3)
        assert "Wait for the running fit" in window.header.status_text()
        controller.auto_ctrl._running = False
        controller.bus.emit("run_state_changed", False)

    def test_starting_a_fix_cancels_a_pending_retrack(self, app):
        """A re-track landing during a fix would redraw the frame under the points."""
        window, controller, rc = app
        controller.retrack.timer.start()
        assert rc.start_fix(3)
        assert not controller.retrack.timer.isActive()


class TestRuns:
    def test_a_fix_goes_on_while_a_retrack_runs(self, app, monkeypatch):
        window, controller, rc = app
        rc.panel.step(1)
        rc.perform("fix")
        monkeypatch.setattr(controller.retrack, "is_running", lambda: True)
        controller.bus.emit("run_state_changed", True)  # a re-track (Fit All is not busy)
        assert rc.panel.isEnabled() and rc.fix.strip.isEnabled() and rc.fix.active
        _points(controller)
        rc.fix.finish_points()
        assert controller._get_state().fit_source[2] == FitSource.MANUAL
        controller.bus.emit("run_state_changed", False)

    def test_fit_all_ends_a_fix(self, app):
        window, controller, rc = app
        rc.panel.step(1)
        rc.perform("fix")
        controller.auto_ctrl._running = True
        controller.bus.emit("run_state_changed", True)
        assert not rc.fix.active and not rc.fix.strip.isEnabled()
        controller.auto_ctrl._running = False
        controller.bus.emit("run_state_changed", False)
        assert rc.fix.strip.isEnabled()


def test_the_status_bar_mode_is_manual_while_fixing(app):
    """The mode is kept for callers but no longer shown (it told users nothing)."""
    window, controller, rc = app
    rc.panel.step(1)
    rc.perform("fix")
    assert window.status_bar.mode() == "Manual"
    rc.fix.cancel()
    assert window.status_bar.mode() == "Measure"


def test_the_fix_tools_do_not_use_the_old_manual_tab():
    """The Manual tab left the panel: nothing of Review's fixing may reach for it."""
    for module in (fix_mode, manual_controller, circle_controller, review_controller,
                   review_actions):
        assert "manual_tab" not in inspect.getsource(module), module.__name__
