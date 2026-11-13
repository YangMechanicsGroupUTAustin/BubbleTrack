"""Review's Fix (F), "Drag a circle", through the real MainWindow + AppController
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
from bubbletrack.ui.circle_tool import TOOL_CIRCLE, TOOL_POINTS
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
    assert controller.review.start_fix(3)
    events = []
    controller.bus.subscribe(RESULTS_CHANGED, lambda *a: events.append("results"))
    controller.bus.subscribe(CIRCLE_FIXED, lambda idx: events.append(("fixed", idx)))
    return window, controller, events


def _drag(window, x=CENTRE_X + 3, y=CENTRE_Y + 3, r=32.0, dragged=True):
    window.original_panel.circle_drawn.emit(x, y, r, dragged)


def _status(window) -> str:
    return window.header._status_label.text()


def _strip(controller):
    return controller.review.fix.strip


# --------------------------------------------------------------------------- #
#  The two tools
# --------------------------------------------------------------------------- #

def test_fix_starts_with_the_circle_tool(app):
    window, controller, _ = app
    strip = _strip(controller)
    assert strip.isVisibleTo(window) or not strip.isHidden()
    assert strip.tool() == TOOL_CIRCLE and strip.frame() == 3
    assert window.original_panel.mode == "circle"
    assert not strip._points_btn.isHidden() and strip._done_btn.isHidden()
    assert strip.hint_text().startswith("Drag from the bubble's centre to its wall")


def test_outside_auto_mode_fix_clicks_edge_points(app):
    window, controller, _ = app
    controller.review.fix.cancel()
    controller._state = update_state(controller._get_state(), detection_mode="intensity")
    assert controller.review.start_fix(3)
    strip = _strip(controller)
    assert strip.tool() == TOOL_POINTS and window.original_panel.mode == "point"
    assert not strip._circle_btn.isEnabled()
    assert "Auto (reference frame)" in strip._circle_btn.toolTip()


def test_mode_change_in_pretune_turns_the_circle_tool_off(app):
    window, controller, _ = app
    pt = window.left_panel.pretune_tab
    params = pt.get_filter_params()
    pt.set_filter_params({**params, "detection_mode": "intensity"})
    pt.filters_changed.emit()
    assert window.original_panel.mode == "normal"
    assert "Auto (reference frame)" in _strip(controller).hint_text()


def test_switching_between_the_tools(app):
    window, controller, _ = app
    strip = _strip(controller)
    strip._points_btn.click()
    assert strip.tool() == TOOL_POINTS and window.original_panel.mode == "point"
    assert controller.manual_ctrl.selecting and not controller.circle_ctrl.active
    strip._circle_btn.click()  # leaving the points tool cancels the selection
    assert strip.tool() == TOOL_CIRCLE and window.original_panel.mode == "circle"
    assert not controller.manual_ctrl.selecting


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
    assert "Frame 4 fixed: R = 40" in _status(window) and "Ctrl+Z" in _status(window)
    assert events == ["results", ("fixed", 3)]
    assert not controller.review.fix.active and window.original_panel.mode == "normal"


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


def test_drags_without_a_fix_do_nothing(app):
    window, controller, _ = app
    controller.review.fix.cancel()
    _drag(window)
    assert controller._get_state().radius[3] == -1.0


# --------------------------------------------------------------------------- #
#  Refused circle: non-blocking prompt
# --------------------------------------------------------------------------- #

def test_circle_beside_the_bubble_is_not_stored_and_prompts(app):
    window, controller, events = app
    _drag(window, 60.0, 60.0, 25.0)
    state = controller._get_state()
    assert state.radius[3] == -1.0 and state.fit_source[3] == FitSource.NONE
    strip = _strip(controller)
    assert strip.prompt_visible() and strip.offers_manual()
    assert "Nothing was stored" in strip.prompt_text()
    assert strip._cancel_btn.isHidden()  # one Cancel: the prompt's
    assert events == [] and controller.review.fix.active


def test_refused_circle_is_drawn_in_the_colour_its_key_names(app):
    window, _, _ = app
    _drag(window, 60.0, 60.0, 25.0)
    # The image key names the dashed circle only when it has the key's colour
    # (cyan, never the amber that read as Auto's yellow guess).
    assert "your circle (not stored)" in window.original_panel.legend_texts()


def _fixed_frame_3(controller):
    """Frame 4 holds a result (a stored circle fix); Fix starts on it again."""
    _drag(controller.w)
    assert controller._get_state().radius[3] > 0
    assert controller.review.start_fix(3)


STORED_KEYS = ["outline (measured)", "equal-area circle", "ROI"]
REFUSED_KEYS = ["your circle (not stored)", "Auto found (not stored)"]


def test_refused_state_shows_only_the_users_circle_and_what_auto_found(app):
    window, controller, _ = app
    _fixed_frame_3(controller)
    left, right = window.original_panel, window.binary_panel
    assert left.legend_texts() == STORED_KEYS
    _drag(window, CENTRE_X, CENTRE_Y, 90.0)                # twice too big: refused
    assert _strip(controller).prompt_visible()
    assert left.legend_texts() == REFUSED_KEYS             # the red / blue result is hidden
    assert len(left._overlay_items) == 2
    assert right.legend_texts() == REFUSED_KEYS and len(right._overlay_items) == 2
    controller.circle_ctrl._redraw(3)                      # any redraw keeps the refused view
    assert left.legend_texts() == REFUSED_KEYS
    _strip(controller).prompt._cancel_btn.click()          # closed: the result is back
    assert left.legend_texts() == STORED_KEYS
    assert right.legend_texts() == ["outline (measured)"]


def test_a_new_drag_or_another_frame_closes_the_refused_view(app):
    window, controller, _ = app
    _fixed_frame_3(controller)
    _drag(window, CENTRE_X, CENTRE_Y, 90.0)
    _drag(window)                                          # a good drag: stored, normal view
    assert window.original_panel.legend_texts() == STORED_KEYS
    assert controller.review.start_fix(3)
    _drag(window, CENTRE_X, CENTRE_Y, 90.0)
    controller.file_ctrl.on_frame_changed(4)
    assert "your circle (not stored)" not in window.original_panel.legend_texts()


def test_refusal_names_the_rule_that_failed(app):
    window, controller, _ = app
    strip = _strip(controller)
    _drag(window, CENTRE_X, CENTRE_Y, 90.0)                # radius rule
    text = strip.prompt_text()
    assert text.startswith("Frame 4: Auto found a wall (grey) that does not match your circle:")
    assert "its radius is 40 px, yours is 90 px" in text and "more than half" in text
    _drag(window, CENTRE_X + 26.0, CENTRE_Y, 30.0)         # centre rule
    text = strip.prompt_text()
    assert "its centre is" in text and "more than half your circle's radius" in text
    assert controller.review.start_fix(1)                   # a frame without a bubble
    _drag(window, CENTRE_X, CENTRE_Y, 40.0)
    text = strip.prompt_text()
    assert "Auto found no wall near your circle" in text or "is too faint: confidence" in text
    assert text.endswith("Nothing was stored. Keep your circle as this frame's result, or drag again.")
    assert strip.prompt._use_btn.text() == "Keep my circle"


def test_keeping_a_circle_says_it_replaces_the_frames_result(app):
    window, controller, _ = app
    _fixed_frame_3(controller)
    _drag(window, CENTRE_X, CENTRE_Y, 90.0)
    assert "Keeping it replaces this frame's result (R = 40" in _strip(controller).prompt_text()


def test_use_my_circle_stores_a_manual_fit_and_undo_restores(app):
    window, controller, events = app
    _drag(window, 60.0, 60.0, 25.0)
    _strip(controller).prompt._use_btn.click()
    state = controller._get_state()
    assert state.fit_source[3] == FitSource.MANUAL and state.radius[3] == 25.0
    assert tuple(state.circle_fit_par[3]) == (60.5, 60.5)  # 1-based
    assert np.isnan(state.fit_confidence[3])
    assert not _strip(controller).prompt_visible() and not controller.review.fix.active
    assert events == ["results", ("fixed", 3)]
    controller._on_undo()
    assert state.radius[3] == -1.0 and state.fit_source[3] == FitSource.NONE


def test_cancel_leaves_the_frame_as_it_was_and_ends_the_fix(app):
    window, controller, _ = app
    before = capture(controller._get_state(), 3)
    _drag(window, 60.0, 60.0, 25.0)
    _strip(controller).prompt._cancel_btn.click()
    assert not _strip(controller).prompt_visible() and not controller.review.fix.active
    assert same_result(capture(controller._get_state(), 3), before)
    assert "frame 4 is unchanged" in _status(window)


def test_dragging_again_after_a_refusal_replaces_the_prompt(app):
    window, controller, _ = app
    _drag(window, 60.0, 60.0, 25.0)
    _drag(window)
    assert controller._get_state().fit_source[3] == FitSource.GUIDED


def test_circle_on_a_frame_without_bubble_is_refused(app):
    window, controller, _ = app
    assert controller.review.start_fix(1)
    _drag(window, CENTRE_X, CENTRE_Y, 40.0)
    assert controller._get_state().radius[1] == -1.0
    assert _strip(controller).prompt_visible()


def test_refused_click_offers_only_cancel(app):
    window, controller, _ = app
    assert controller.review.start_fix(1)
    _drag(window, CENTRE_X, CENTRE_Y, 0.0, dragged=False)
    strip = _strip(controller)
    assert strip.prompt_visible() and not strip.offers_manual()


def test_changing_frame_ends_the_fix_and_its_prompt(app):
    window, controller, _ = app
    _drag(window, 60.0, 60.0, 25.0)
    controller.file_ctrl.on_frame_changed(4)
    assert not _strip(controller).prompt_visible() and not controller.review.fix.active
    _strip(controller).prompt._use_btn.click()  # stale: nothing happens
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
    controller._on_lock_range(3, 3, locked=False)  # a fix is locked: unlock it first
    controller._on_mark_no_bubble(3, 3)  # changed since (chart: Mark as no bubble)
    _ctrl_z(window)
    assert controller._get_state().radius[3] == -1.0
    assert "changed since" in _status(window)


def _point_fit(controller, frame, radius=30.0):
    """A committed "Click edge points" fit on *frame*, through Review's Fix."""
    assert controller.review.start_fix(frame)
    controller.review.fix.use_tool(TOOL_POINTS)
    for a in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        controller.manual_ctrl.on_point_clicked(CENTRE_X + radius * np.cos(a),
                                                CENTRE_Y + radius * np.sin(a))
    controller.review.fix.finish_points()


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


def test_ctrl_z_still_undoes_a_clicked_point_first(app):
    window, controller, _ = app
    _drag(window)
    radius = controller._get_state().radius[3]
    assert controller.review.start_fix(3)
    _strip(controller)._points_btn.click()
    controller.manual_ctrl.on_point_clicked(200.0, 100.0)
    controller.manual_ctrl.on_point_clicked(220.0, 120.0)
    assert _strip(controller).hint_text().endswith("2 so far")
    _ctrl_z(window)
    assert len(controller.manual_ctrl._manual_points) == 1
    assert _strip(controller).hint_text().endswith("1 so far")
    assert controller._get_state().radius[3] == radius  # the circle fix is untouched


# --------------------------------------------------------------------------- #
#  Runs, Fit All protection, review
# --------------------------------------------------------------------------- #

def test_fit_all_ends_the_fix(app):
    window, controller, _ = app
    controller.auto_ctrl._running = True
    controller.bus.emit("run_state_changed", True)
    assert window.original_panel.mode == "normal" and not controller.review.fix.active
    _drag(window)
    assert controller._get_state().radius[3] == -1.0
    controller.auto_ctrl._running = False
    controller.bus.emit("run_state_changed", False)


def test_a_drawn_circle_is_locked_so_fit_all_keeps_it_without_asking(app):
    """m6: every later Fit asked "1 frame(s) in this range were fixed by hand"."""
    window, controller, _ = app
    _drag(window)
    assert controller._get_state().fit_locked[3]
    asked = []
    controller.auto_ctrl._ask_manual_policy = lambda n: asked.append(n) or "keep"
    assert controller.auto_ctrl._choose_protected_frames(0, 6) == frozenset({3})
    assert asked == []
    controller._get_state().fit_locked[3] = False  # unlocked on the chart: asked again
    assert controller.auto_ctrl._choose_protected_frames(0, 6) == frozenset({3})
    assert asked == [1]


def test_undoing_a_circle_fix_unlocks_the_frame(app):
    window, controller, _ = app
    _drag(window)
    controller._on_undo()
    state = controller._get_state()
    assert not state.fit_locked[3] and state.radius[3] == -1.0


def test_review_fix_key_is_bound():
    assert REVIEW_BINDINGS["F"] == "fix"
    assert "C" not in REVIEW_BINDINGS and "M" not in REVIEW_BINDINGS


def _flag_frames(controller, low):
    state = controller._get_state()
    for i in range(2, 7):
        state.radius[i] = RADII[i] + 3.0  # deliberately a little off
        state.circle_fit_par[i] = [126.0, 201.0]
        state.fit_source[i] = FitSource.AUTO
        state.fit_confidence[i] = 0.1 if i in low else 0.9
    controller.review.refresh()
    return state


def test_review_fix_opens_the_tool_and_moves_on_after_a_fix(app):
    window, controller, _ = app
    rc = controller.review
    rc.fix.cancel()
    _flag_frames(controller, low={3, 5})
    rc.panel.step(1)
    assert controller._get_state().image_no == 3
    rc.perform("fix")
    assert rc.fix.tool == TOOL_CIRCLE and window.original_panel.mode == "circle"
    _drag(window)
    assert controller._get_state().fit_source[3] == FitSource.GUIDED
    assert rc.panel.frames() == [5] and controller._get_state().image_no == 5
    assert rc.panel.feedback.text().startswith("✓ Frame 4 fixed: R = 40")
