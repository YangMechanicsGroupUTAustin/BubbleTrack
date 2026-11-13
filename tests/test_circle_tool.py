"""Manual tab "Drag a circle" through the real MainWindow + AppController
offscreen: press at the centre, drag to the wall, release; Auto measures the
wall near the circle, the result shows at once, and Ctrl+Z undoes it."""

from __future__ import annotations

import numpy as np
import pytest
from PyQt6.QtCore import QEvent, QPointF, Qt
from PyQt6.QtGui import QMouseEvent, QShortcut
from PyQt6.QtWidgets import QApplication

from bubbletrack.controller.display_mixin import ENGINE_VIEW_TITLE
from bubbletrack.event_bus import CIRCLE_FIXED, RESULTS_CHANGED
from bubbletrack.model import guided_fit
from bubbletrack.model.provenance import FitSource
from bubbletrack.model.result_history import capture, same_result
from bubbletrack.model.state import update_state
from bubbletrack.ui.manual_tab import TOOL_CIRCLE, TOOL_POINTS
from bubbletrack.ui.shortcuts import REVIEW_BINDINGS
from conftest import write_sequence

RADII = [0.0, 0.0, 30.0, 40.0, 50.0, 45.0, 35.0]  # frames 1-2: bubble-free references
CENTRE_X, CENTRE_Y = 200.5, 125.5  # scene coordinates of the synthetic bubble centre


@pytest.fixture
def app(gui, tmp_path):
    window, controller = gui
    controller.file_ctrl.on_folder_selected(str(write_sequence(tmp_path / "seq", RADII)))
    controller._state = update_state(controller._get_state(), detection_mode="edge",
                                     reference_frame=0)
    window.left_panel.tab_bar.select("Manual")
    controller.file_ctrl.on_frame_changed(3)
    events = []
    controller.bus.subscribe(RESULTS_CHANGED, lambda *a: events.append("results"))
    controller.bus.subscribe(CIRCLE_FIXED, lambda idx: events.append(("fixed", idx)))
    return window, controller, events


def _drag(window, x=CENTRE_X + 3, y=CENTRE_Y + 3, r=32.0, dragged=True):
    window.original_panel.circle_drawn.emit(x, y, r, dragged)


def _status(window) -> str:
    return window.header._status_label.text()


# --------------------------------------------------------------------------- #
#  The two tools
# --------------------------------------------------------------------------- #

def test_manual_tab_has_two_tools_and_circle_is_the_default(app):
    window, _, _ = app
    tab = window.left_panel.manual_tab
    assert [b.text() for b in tab._tool_buttons.values()] == ["Drag a circle", "Click edge points"]
    assert tab.tool() == TOOL_CIRCLE and tab._tool_buttons[TOOL_CIRCLE].isChecked()
    assert window.original_panel.mode == "circle"


def test_circle_tool_is_disabled_outside_auto_mode(app):
    window, controller, _ = app
    tab = window.left_panel.manual_tab
    controller._state = update_state(controller._get_state(), detection_mode="intensity")
    controller.circle_ctrl.sync_tool()
    button = tab._tool_buttons[TOOL_CIRCLE]
    assert not button.isEnabled() and "Auto (reference frame)" in button.toolTip()
    assert tab.tool() == TOOL_POINTS and window.original_panel.mode == "normal"
    controller._state = update_state(controller._get_state(), detection_mode="edge")
    controller.circle_ctrl.sync_tool()
    assert tab.tool() == TOOL_CIRCLE and window.original_panel.mode == "circle"


def test_mode_change_in_pretune_updates_the_tool(app):
    window, _, _ = app
    pt = window.left_panel.pretune_tab
    params = pt.get_filter_params()
    pt.set_filter_params({**params, "detection_mode": "intensity"})
    pt.filters_changed.emit()
    assert not window.left_panel.manual_tab.circle_available()


def test_points_tool_keeps_its_behaviour(app):
    window, controller, _ = app
    tab = window.left_panel.manual_tab
    tab.set_tool(TOOL_POINTS)
    assert window.original_panel.mode == "normal"
    tab._select_btn.click()
    assert window.original_panel.mode == "point"
    tab.set_tool(TOOL_CIRCLE)  # leaving the tool cancels the selection
    assert window.original_panel.mode == "circle" and not controller.manual_ctrl._selecting


def test_other_tabs_leave_circle_mode(app):
    window, _, _ = app
    window.left_panel.tab_bar.select("Pre-tune")
    assert window.original_panel.mode == "normal"
    window.left_panel.tab_bar.select("Manual")
    assert window.original_panel.mode == "circle"


# --------------------------------------------------------------------------- #
#  Accepted circle
# --------------------------------------------------------------------------- #

def test_drag_measures_the_wall_and_shows_it_in_both_panels(app):
    window, controller, events = app
    _drag(window)
    state = controller._get_state()
    assert state.radius[3] == pytest.approx(40.0, abs=1.0)
    assert tuple(state.circle_fit_par[3]) == pytest.approx((126.0, 201.0), abs=1.0)
    assert state.fit_source[3] == FitSource.GUIDED
    assert np.isfinite(state.fit_confidence[3]) and state.circle_xy[3] is not None
    assert len(window.original_panel._overlay_items) >= 3  # ROI, circle, outline
    assert window.binary_panel._title_label.text() == ENGINE_VIEW_TITLE
    assert len(window.binary_panel._overlay_items) == 1  # the outline on the Auto view
    assert "R = 40" in _status(window) and "Ctrl+Z" in _status(window)
    assert events == ["results", ("fixed", 3)]


def _mouse(view, kind, pos: QPointF, buttons):
    button = Qt.MouseButton.LeftButton if kind != QEvent.Type.MouseMove else Qt.MouseButton.NoButton
    event = QMouseEvent(kind, pos, view.viewport().mapToGlobal(pos), button, buttons,
                        Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(view.viewport(), event)


def test_mouse_drag_shows_a_dashed_preview_then_the_result(qapp, app):
    window, controller, _ = app
    window.resize(1440, 900)
    window.show()
    qapp.processEvents()
    view = window.original_panel._view
    centre = QPointF(view.mapFromScene(QPointF(CENTRE_X, CENTRE_Y)))
    wall = QPointF(view.mapFromScene(QPointF(CENTRE_X + 36, CENTRE_Y)))
    held = Qt.MouseButton.LeftButton
    _mouse(view, QEvent.Type.MouseButtonPress, centre, held)
    _mouse(view, QEvent.Type.MouseMove, wall, held)
    preview = window.original_panel.circle_drag.preview
    assert preview is not None and preview.pen().style() == Qt.PenStyle.DashLine
    assert preview.rect().width() / 2 == pytest.approx(36.0, abs=1.5)
    _mouse(view, QEvent.Type.MouseButtonRelease, wall, Qt.MouseButton.NoButton)
    assert window.original_panel.circle_drag.preview is None
    state = controller._get_state()
    assert state.fit_source[3] == FitSource.GUIDED
    assert state.radius[3] == pytest.approx(40.0, abs=1.0)


def test_click_uses_the_nearest_fitted_frame_radius(app, monkeypatch):
    window, controller, _ = app
    state = controller._get_state()
    state.radius[5] = 45.0
    seen = []
    real = guided_fit.fit_guided

    def spy(raw, gridx, gridy, job, circle):
        seen.append(circle)
        return real(raw, gridx, gridy, job, circle)
    monkeypatch.setattr(guided_fit, "fit_guided", spy)
    _drag(window, CENTRE_X, CENTRE_Y, 0.0, dragged=False)
    assert seen[0].radius == 45.0 and not seen[0].dragged and seen[0].sized
    assert state.fit_source[3] == FitSource.GUIDED  # 40 vs 45: consistent


def test_click_without_fitted_frames_still_finds_the_bubble(app):
    window, controller, _ = app
    _drag(window, CENTRE_X + 2, CENTRE_Y - 2, 0.0, dragged=False)
    state = controller._get_state()
    assert state.fit_source[3] == FitSource.GUIDED
    assert state.radius[3] == pytest.approx(40.0, abs=1.0)


# --------------------------------------------------------------------------- #
#  Refused circle: non-blocking prompt
# --------------------------------------------------------------------------- #

def _prompt(window):
    return window.left_panel.manual_tab.circle_page


def test_circle_beside_the_bubble_is_not_stored_and_prompts(app):
    window, controller, events = app
    _drag(window, 60.0, 60.0, 25.0)
    state = controller._get_state()
    assert state.radius[3] == -1.0 and state.fit_source[3] == FitSource.NONE
    page = _prompt(window)
    assert page.prompt_visible() and page.offers_manual()
    assert "Nothing was stored" in page.prompt_text()
    assert events == []


def test_use_my_circle_stores_a_manual_fit_and_undo_restores(app):
    window, controller, events = app
    _drag(window, 60.0, 60.0, 25.0)
    _prompt(window)._use_btn.click()
    state = controller._get_state()
    assert state.fit_source[3] == FitSource.MANUAL and state.radius[3] == 25.0
    assert tuple(state.circle_fit_par[3]) == (60.5, 60.5)  # 1-based
    assert np.isnan(state.fit_confidence[3])
    assert not _prompt(window).prompt_visible()
    assert events == ["results", ("fixed", 3)]
    controller._on_undo()
    assert state.radius[3] == -1.0 and state.fit_source[3] == FitSource.NONE


def test_cancel_leaves_the_frame_as_it_was(app):
    window, controller, _ = app
    before = capture(controller._get_state(), 3)
    _drag(window, 60.0, 60.0, 25.0)
    _prompt(window)._cancel_btn.click()
    assert not _prompt(window).prompt_visible()
    assert same_result(capture(controller._get_state(), 3), before)


def test_circle_on_a_frame_without_bubble_is_refused(app):
    window, controller, _ = app
    controller.file_ctrl.on_frame_changed(1)
    _drag(window, CENTRE_X, CENTRE_Y, 40.0)
    assert controller._get_state().radius[1] == -1.0
    assert _prompt(window).prompt_visible()


def test_refused_click_offers_only_cancel(app):
    window, controller, _ = app
    controller.file_ctrl.on_frame_changed(1)
    _drag(window, CENTRE_X, CENTRE_Y, 0.0, dragged=False)
    page = _prompt(window)
    assert page.prompt_visible() and not page.offers_manual()


def test_changing_frame_dismisses_the_prompt(app):
    window, controller, _ = app
    _drag(window, 60.0, 60.0, 25.0)
    controller.file_ctrl.on_frame_changed(4)
    assert not _prompt(window).prompt_visible()
    _prompt(window)._use_btn.click()  # stale: nothing happens
    assert controller._get_state().radius[4] == -1.0


# --------------------------------------------------------------------------- #
#  Undo
# --------------------------------------------------------------------------- #

def _ctrl_z(window):
    shortcut = next(sc for sc in window.findChildren(QShortcut) if sc.key().toString() == "Ctrl+Z")
    shortcut.activated.emit()


def test_ctrl_z_restores_the_previous_result_exactly(app):
    window, controller, events = app
    state = controller._get_state()
    state.radius[3], state.circle_fit_par[3] = 33.0, [120.0, 190.0]
    state.circle_xy[3] = np.array([[1.0, 2.0], [3.0, 4.0]])
    state.fit_source[3], state.fit_confidence[3] = FitSource.AUTO, 0.21
    state.fit_flags[3] = frozenset({"non_circular"})
    state.fit_metrics["circularity"][3] = 0.7
    before = capture(state, 3)
    _drag(window)
    assert state.fit_source[3] == FitSource.GUIDED
    _ctrl_z(window)
    assert same_result(capture(state, 3), before)
    assert "Undid the circle fix on frame 4" in _status(window)
    assert events.count("results") == 2  # the fix and its undo


def test_ctrl_z_refuses_when_the_frame_changed_since(app):
    window, controller, _ = app
    _drag(window)
    controller._on_delete_point(3)
    _ctrl_z(window)
    assert controller._get_state().radius[3] == -1.0
    assert "changed since" in _status(window)


def _point_fit(controller, frame, radius=30.0):
    """A committed "Click edge points" fit on *frame*."""
    controller.file_ctrl.on_frame_changed(frame)
    controller.manual_ctrl.on_manual_select()
    for a in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        controller.manual_ctrl.on_point_clicked(CENTRE_X + radius * np.cos(a),
                                                CENTRE_Y + radius * np.sin(a))
    controller.manual_ctrl.on_manual_done()


def test_ctrl_z_after_a_point_fit_undoes_that_fit_not_an_older_circle(app):
    window, controller, _ = app
    state = controller._get_state()
    _drag(window)  # circle fix on frame 3
    circle_fix = capture(state, 3)
    before_points = capture(state, 5)
    _point_fit(controller, 5)
    assert state.fit_source[5] == FitSource.MANUAL
    _ctrl_z(window)
    assert same_result(capture(state, 5), before_points)  # the newest fix is undone
    assert same_result(capture(state, 3), circle_fix)  # the older one is kept
    _ctrl_z(window)
    assert state.fit_source[3] == FitSource.NONE  # then the circle fix


def test_ctrl_z_undoes_a_manual_clear(app):
    window, controller, _ = app
    state = controller._get_state()
    _drag(window)
    fixed = capture(state, 3)
    controller.manual_ctrl.on_manual_clear()
    assert state.radius[3] == -1.0
    _ctrl_z(window)
    assert same_result(capture(state, 3), fixed)


def test_ctrl_z_still_undoes_a_clicked_point_first(app):
    window, controller, _ = app
    _drag(window)
    radius = controller._get_state().radius[3]
    window.left_panel.manual_tab.set_tool(TOOL_POINTS)
    controller.manual_ctrl.on_manual_select()
    controller.manual_ctrl.on_point_clicked(200.0, 100.0)
    controller.manual_ctrl.on_point_clicked(220.0, 120.0)
    _ctrl_z(window)
    assert len(controller.manual_ctrl._manual_points) == 1
    assert controller._get_state().radius[3] == radius  # the circle fix is untouched


# --------------------------------------------------------------------------- #
#  Runs, Fit All protection, review
# --------------------------------------------------------------------------- #

def test_drags_are_ignored_while_a_fit_runs(app):
    window, controller, _ = app
    controller.auto_ctrl._running = True
    controller.bus.emit("run_state_changed", True)
    assert window.original_panel.mode == "normal"
    _drag(window)
    assert controller._get_state().radius[3] == -1.0
    controller.auto_ctrl._running = False
    controller.bus.emit("run_state_changed", False)
    assert window.original_panel.mode == "circle"


def test_fit_all_asks_before_overwriting_drawn_circles(app):
    window, controller, _ = app
    _drag(window)
    asked = []
    controller.auto_ctrl._ask_manual_policy = lambda n: asked.append(n) or "keep"
    assert controller.auto_ctrl._choose_protected_frames(0, 6) == frozenset({3})
    assert asked == [1]


def test_review_circle_key_is_bound():
    assert REVIEW_BINDINGS["C"] == "circle"


def _flag_frames(controller, low):
    state = controller._get_state()
    for i in range(2, 7):
        state.radius[i] = RADII[i] + 3.0  # deliberately a little off
        state.circle_fit_par[i] = [126.0, 201.0]
        state.fit_source[i] = FitSource.AUTO
        state.fit_confidence[i] = 0.1 if i in low else 0.9
    controller.review.refresh()
    return state


def test_review_circle_opens_the_tool_and_advances_after_a_fix(app):
    window, controller, _ = app
    rc = controller.review
    _flag_frames(controller, low={3, 5})
    window.left_panel.tab_bar.select("Pre-tune")
    rc.panel.step(1)
    assert controller._get_state().image_no == 3
    rc.perform("circle")
    assert window.left_panel.tab_bar.current_name() == "Manual"
    assert window.left_panel.manual_tab.tool() == TOOL_CIRCLE
    assert window.original_panel.mode == "circle"
    _drag(window)
    assert controller._get_state().fit_source[3] == FitSource.GUIDED
    assert rc.panel.frames() == [5] and controller._get_state().image_no == 5


def test_review_circle_explains_itself_outside_auto_mode(app):
    window, controller, _ = app
    rc = controller.review
    _flag_frames(controller, low={3})
    controller._state = update_state(controller._get_state(), detection_mode="intensity")
    window.left_panel.tab_bar.select("Pre-tune")
    rc.panel.step(1)
    rc.perform("circle")
    assert window.left_panel.tab_bar.current_name() == "Pre-tune"
    assert "Auto (reference frame)" in _status(window)
