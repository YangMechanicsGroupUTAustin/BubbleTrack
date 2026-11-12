"""Every in-place edit of the results announces RESULTS_CHANGED, so the
autosave, the review queue and other views stay in step with the data."""

from __future__ import annotations

import numpy as np
import pytest

from bubbletrack.controller.review_controller import install_review
from bubbletrack.event_bus import RESULTS_CHANGED
from bubbletrack.model.provenance import FitSource
from bubbletrack.model.state import update_state


@pytest.fixture
def app(gui, make_sequence):
    window, controller = gui
    controller.file_ctrl.on_folder_selected(str(make_sequence("A", 8)))
    state = controller._get_state()
    state.radius[:] = np.linspace(30.0, 40.0, 8)
    state.circle_fit_par[:] = [[125.0, 200.0]] * 8
    state.fit_source[:] = FitSource.AUTO
    state.fit_confidence[:] = 0.9
    events = []
    controller.bus.subscribe(RESULTS_CHANGED, lambda *a: events.append(a))
    return window, controller, events


def _circle_clicks(controller, r=30.0):
    for a in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        controller.manual_ctrl.on_point_clicked(200 + r * np.cos(a), 125 + r * np.sin(a))


def test_chart_delete(app):
    _, controller, events = app
    controller._on_delete_point(3)
    assert events == [()]


def test_manual_done_and_clear(app):
    window, controller, events = app
    controller.manual_ctrl.on_manual_select()
    _circle_clicks(controller)
    controller.manual_ctrl.on_manual_done()
    assert len(events) == 1
    controller.manual_ctrl.on_manual_clear()
    assert len(events) == 2


def test_manual_done_with_too_few_points_changes_nothing(app):
    _, controller, events = app
    controller.manual_ctrl.on_manual_select()
    controller.manual_ctrl.on_point_clicked(200.0, 125.0)
    controller.manual_ctrl.on_manual_done()
    assert events == []


def test_clear_all(app):
    _, controller, events = app
    controller.auto_ctrl.on_auto_clear()
    assert len(events) == 1


def test_edge_refit_announces_even_a_rejected_fit(app):
    _, controller, events = app
    controller._state = update_state(controller._get_state(), detection_mode="edge")
    controller._on_refit_point(2)
    assert len(events) == 1


def test_refit_that_fails_keeps_quiet(app, monkeypatch):
    _, controller, events = app
    import bubbletrack.controller.controller as module

    monkeypatch.setattr(module, "detect_bubble_with_mode",
                        lambda *a, **k: (np.zeros((2, 2), bool), np.empty((0, 2))))
    controller._on_refit_point(2)
    assert events == []


def test_edit_schedules_an_autosave(app):
    _, controller, _ = app
    session = controller.session_ctrl
    session._autosave_dirty = False
    controller._on_delete_point(1)
    assert session._autosave_dirty


def test_manual_from_the_queue_advances_to_the_next_item(app):
    """Manual Done refreshes the queue first; the advance must still go to
    the item after the edited one, not back to the top of the queue."""
    window, controller, _ = app
    rc = install_review(window, controller)
    state = controller._get_state()
    for frame in (2, 4, 6):
        state.fit_confidence[frame] = 0.1
    rc.refresh()
    assert rc.panel.frames() == [2, 4, 6]
    rc.panel.select_frame(4)
    rc.perform("manual")
    _circle_clicks(controller)
    window.left_panel.manual_tab.done_clicked.emit()
    assert state.fit_source[4] == FitSource.MANUAL
    assert rc.panel.frames() == [2, 6]
    assert controller._get_state().image_no == 6


def test_chart_menu_edit_refreshes_the_queue(app):
    window, controller, _ = app
    rc = install_review(window, controller)
    state = controller._get_state()
    state.fit_confidence[5] = 0.1
    rc.refresh()
    assert rc.panel.frames() == [5]
    state.fit_confidence[5] = 0.95  # e.g. a refit raised it ...
    controller.bus.emit(RESULTS_CHANGED)  # ... and announced the change
    assert rc.panel.frames() == []
