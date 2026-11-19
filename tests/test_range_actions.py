"""Brushed-range actions on the R(t) chart, and what a lock protects."""

from __future__ import annotations

import time

import numpy as np
import pytest

from bubbletrack.event_bus import RESULTS_CHANGED
from bubbletrack.model.provenance import FitSource
from bubbletrack.model.state import clear_frame_result, update_state

N = 10


def _wait(qapp, predicate, timeout_s: float = 120.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        qapp.processEvents()
        if predicate():
            return
        time.sleep(0.01)
    raise TimeoutError("condition not reached")


@pytest.fixture
def app(gui, make_sequence):
    window, controller = gui
    controller.file_ctrl.on_folder_selected(str(make_sequence("A", N)))
    state = controller._get_state()
    state.radius[:] = np.linspace(20.0, 70.0, N)
    state.circle_fit_par[:] = [[125.0, 200.0]] * N
    state.fit_source[:] = FitSource.AUTO
    state.fit_confidence[:] = 0.9
    events = []
    controller.bus.subscribe(RESULTS_CHANGED, lambda *a: events.append(a))
    return window, controller, state, events


def test_lock_range_locks_fitted_frames_only_and_unlock_releases(app):
    window, _, state, events = app
    state.radius[6] = -1.0
    window.radius_chart.lock_range_requested.emit(5, 7)
    assert list(state.fit_locked[4:9]) == [False, True, False, True, False]
    window.radius_chart.unlock_range_requested.emit(0, N - 1)
    assert not state.fit_locked.any()
    assert len(events) == 2


def test_erasing_a_locked_frame_unlocks_it(app):
    """Otherwise Fit All would skip the now-empty frame forever."""
    _, _, state, _ = app
    state.fit_locked[0] = True
    clear_frame_result(state, 0)
    assert state.radius[0] == -1.0 and not state.fit_locked[0]


def test_refit_range_fits_only_that_range(qapp, app):
    window, controller, state, _ = app
    controller._state = update_state(controller._get_state(), detection_mode="intensity",
                                     img_thr=0.58, removing_factor=85, gaussian_sigma=3.0)
    state = controller._get_state()
    state.radius[:] = 5.0  # a recognisable wrong value everywhere
    state.fit_locked[4] = True
    window.radius_chart.refit_range_requested.emit(3, 5)
    _wait(qapp, lambda: not controller.auto_ctrl.is_busy())
    state = controller._get_state()
    expected = np.linspace(20.0, 70.0, N)
    assert state.radius[3] == pytest.approx(expected[3], rel=0.05)
    assert state.radius[5] == pytest.approx(expected[5], rel=0.05)
    assert state.radius[4] == 5.0  # locked: kept
    assert (state.radius[:3] == 5.0).all() and (state.radius[6:] == 5.0).all()


def test_mark_no_bubble_clears_and_locks_empty_frames(app):
    """Unlike Delete, it says "no bubble": the frames stay empty and locked."""
    window, _, state, events = app
    state.fit_locked[6] = True  # locked WITH a result: kept
    window.radius_chart.no_bubble_range_requested.emit(5, 7)
    assert state.radius[5] == -1.0 and state.radius[7] == -1.0
    assert state.fit_locked[5] and state.fit_locked[7]
    assert state.radius[6] > 0 and state.fit_locked[6]
    assert "Marked 2 frames as no bubble (1 locked kept)" in window.header._status_label.text()
    assert len(events) == 1


def test_frames_marked_no_bubble_export_as_no_bubble_labels(app):
    from bubbletrack.model.cases.labels import app_result, frame_label
    from bubbletrack.model.cases.schema import LABEL_NO_BUBBLE

    window, _, state, _ = app
    window.radius_chart.no_bubble_range_requested.emit(2, 2)
    assert frame_label(app_result(state, 2)).status == LABEL_NO_BUBBLE


def test_a_chart_no_bubble_mark_is_one_undo(app):
    """Ctrl+Z after the chart's "Mark as no bubble" brings the frames back, and
    does not undo an older edit instead."""
    window, controller, state, _ = app
    before = state.radius.copy()
    window.radius_chart.no_bubble_range_requested.emit(4, 8)
    assert (state.radius[4:9] == -1.0).all()
    controller._on_undo()
    np.testing.assert_array_equal(state.radius, before)
    assert not state.fit_locked.any()
    assert "no-bubble mark of frames 5–9" in window.header._status_label.text()


def test_ctrl_z_after_a_fit_keeps_an_edit_of_a_frame_it_ran_on(app, qapp, monkeypatch):
    """A lock kept frames 3-4 out of Fit All; undoing it afterwards would leave them
    unlocked with results of the older run, which the Fit never re-measured."""
    from bubbletrack.controller import worker as worker_module

    window, controller, state, _ = app
    window.radius_chart.lock_range_requested.emit(2, 3)
    monkeypatch.setattr(worker_module.BatchWorker, "inline", True)
    controller.auto_ctrl.on_auto_fit()
    _wait(qapp, lambda: not controller.auto_ctrl.is_busy())
    controller._on_undo()
    assert state.fit_locked[2] and state.fit_locked[3]
    assert "Cannot undo the lock of frames 3–4: a Fit ran on frame 3 since" \
        in window.header._status_label.text()
    window.radius_chart.lock_range_requested.emit(6, 6)  # an edit after the Fit
    controller._on_undo()
    assert not state.fit_locked[6]


def test_a_chart_lock_is_one_undo(app):
    window, controller, state, _ = app
    window.radius_chart.lock_range_requested.emit(2, 3)
    assert state.fit_locked[2] and state.fit_locked[3]
    controller._on_undo()
    assert not state.fit_locked.any()
