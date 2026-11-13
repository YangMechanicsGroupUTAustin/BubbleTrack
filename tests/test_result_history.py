"""Undo history of per-frame results (circle fixes and re-tracks): an undo
restores the frames exactly, and never clobbers a frame changed since."""

from __future__ import annotations

import numpy as np

from bubbletrack.model.provenance import FitSource
from bubbletrack.model.result_history import (
    ResultHistory, capture, record_edit, restore, same_result, snapshot,
)
from bubbletrack.model.state import AppState


def _state(n=4):
    state = AppState().with_results_initialized(n)
    state.radius[:] = [10.0, 20.0, -1.0, 40.0]
    state.circle_fit_par[:] = [[1.0, 2.0], [3.0, 4.0], [np.nan, np.nan], [7.0, 8.0]]
    state.circle_xy[1] = np.array([[1.0, 2.0], [3.0, 4.0]])
    state.fit_source[:] = [FitSource.AUTO, FitSource.AUTO, FitSource.NONE, FitSource.MANUAL]
    state.fit_confidence[:] = [0.9, 0.4, 0.1, np.nan]
    state.fit_flags[1] = frozenset({"non_circular"})
    state.fit_metrics["r_circle"][1] = 21.0
    state.fit_locked[0] = True
    state.region_count[1] = 2
    return state


def _edit_frame(state, idx, radius):
    state.radius[idx] = radius
    state.circle_fit_par[idx] = [50.0, 60.0]
    state.circle_xy[idx] = np.array([[9.0, 9.0]])
    state.fit_source[idx] = FitSource.GUIDED
    state.fit_confidence[idx] = 0.7
    state.fit_flags[idx] = frozenset({"circle_fallback"})
    state.fit_metrics["r_circle"][idx] = radius + 1
    state.region_count[idx] = 0


def test_capture_restore_round_trip():
    state = _state()
    before = capture(state, 1)
    _edit_frame(state, 1, 33.0)
    assert not same_result(capture(state, 1), before)
    restore(state, 1, before)
    assert same_result(capture(state, 1), before)
    assert state.radius[1] == 20.0 and state.fit_source[1] == FitSource.AUTO
    assert np.array_equal(state.circle_xy[1], [[1.0, 2.0], [3.0, 4.0]])
    assert state.fit_flags[1] == frozenset({"non_circular"}) and state.region_count[1] == 2


def test_capture_is_a_copy():
    state = _state()
    before = capture(state, 1)
    state.circle_xy[1][0, 0] = 99.0  # in-place edit of the stored array
    assert before.edge_xy[0, 0] == 1.0


def test_nan_results_compare_equal():
    state = _state()
    assert same_result(capture(state, 2), capture(state, 2))


def test_undo_restores_every_frame_of_the_edit():
    state = _state()
    history = ResultHistory()
    before = snapshot(state, [1, 2])
    _edit_frame(state, 1, 33.0)
    _edit_frame(state, 2, 44.0)
    history.push(record_edit(state, "re-track", before))
    outcome = history.undo(state)
    assert outcome.done and outcome.edit.frames == (1, 2)
    assert state.radius[1] == 20.0 and state.radius[2] == -1.0
    assert np.isnan(state.circle_fit_par[2]).all() and state.circle_xy[2] is None
    assert not history.can_undo()


def test_record_keeps_only_frames_that_changed():
    state = _state()
    before = snapshot(state, [0, 1, 2])
    _edit_frame(state, 1, 33.0)
    edit = record_edit(state, "re-track", before)
    assert edit.frames == (1,)


def test_undo_refuses_when_a_frame_changed_since():
    state = _state()
    history = ResultHistory()
    before = snapshot(state, [1])
    _edit_frame(state, 1, 33.0)
    history.push(record_edit(state, "circle fix on frame 2", before))
    state.radius[1] = 5.0  # e.g. deleted or refitted afterwards
    outcome = history.undo(state)
    assert not outcome.done and "frame 2" in outcome.message
    assert state.radius[1] == 5.0  # untouched


def test_undo_refuses_after_a_new_source_was_loaded():
    state = _state()
    history = ResultHistory()
    before = snapshot(state, [1])
    _edit_frame(state, 1, 33.0)
    history.push(record_edit(state, "circle fix", before))
    other = AppState().with_results_initialized(9)
    assert not history.undo(other).done


def test_undo_is_last_in_first_out():
    state = _state()
    history = ResultHistory()
    first = snapshot(state, [1])
    _edit_frame(state, 1, 33.0)
    history.push(record_edit(state, "first", first))
    second = snapshot(state, [1])
    _edit_frame(state, 1, 55.0)
    history.push(record_edit(state, "second", second))
    assert history.undo(state).edit.label == "second" and state.radius[1] == 33.0
    assert history.undo(state).edit.label == "first" and state.radius[1] == 20.0


def test_empty_history_has_nothing_to_undo():
    outcome = ResultHistory().undo(_state())
    assert not outcome.done and outcome.edit is None and "Nothing" in outcome.message


def test_an_edit_that_changed_nothing_is_not_recorded():
    state = _state()
    history = ResultHistory()
    history.push(record_edit(state, "noop", snapshot(state, [1])))
    assert not history.can_undo()
