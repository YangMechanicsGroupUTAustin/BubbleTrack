"""Regression tests from the 2026-11-18 bug hunt of ML assist, tracking and the filmstrip."""

from __future__ import annotations

import json

import numpy as np

from bubbletrack.controller.tracking import review_items
from bubbletrack.event_bus import RUN_STATE_CHANGED, TRACKING_REVIEW
from bubbletrack.model.sequence import FrameSolution, SequenceResult


def _solution(r: float = 20.0) -> FrameSolution:
    return FrameSolution(r, 50.0, 60.0, 0.8, "tracked", frozenset(), 0)


def test_unapplied_tracking_is_not_called_a_change():
    """Outside the frames a re-track may change, "disagree" described a circle that was
    not stored: Review said tracking changed the frame, and the next re-track did."""
    result = SequenceResult(tuple(_solution() for _ in range(6)), (), ((0, 5),),
                            ((4, ("disagree",)), (5, ("disagree", "lost"))), 0)
    assert [(i.frame, i.reasons) for i in review_items(result)] == [
        (4, ("disagree",)), (5, ("disagree", "lost"))]
    items = review_items(result, allowed_frames=frozenset({5}))
    assert [(i.frame, i.reasons) for i in items] == [(5, ("disagree", "lost"))]
    items = review_items(result, allowed_frames=frozenset())
    assert [(i.frame, i.reasons) for i in items] == [(5, ("lost",))]


def test_a_run_without_tracking_keeps_the_review_list(gui, make_sequence):
    """A stopped or untracked partial run posted an empty tracking list, which
    dropped every frame's solver reasons, also outside the run."""
    _window, controller = gui
    controller.file_ctrl.on_folder_selected(str(make_sequence("a", 6)))
    posted = []
    controller.bus.subscribe(TRACKING_REVIEW, lambda items: posted.append(items))
    controller.auto_ctrl._run_frames = (1, 2)
    controller.auto_ctrl._finish_run(stopped=True)
    assert posted == []


def test_the_roi_waits_for_the_running_fit(gui, make_sequence):
    window, controller = gui
    controller.file_ctrl.on_folder_selected(str(make_sequence("a", 6)))
    roi = (controller._get_state().gridx, controller._get_state().gridy)
    button = window.original_panel._select_roi_btn
    controller.bus.emit(RUN_STATE_CHANGED, True)
    assert not button.isEnabled()
    window.original_panel.roi_selected.emit(1, 20, 1, 30)
    assert (controller._get_state().gridx, controller._get_state().gridy) == roi
    controller.bus.emit(RUN_STATE_CHANGED, False)
    assert button.isEnabled()
    window.original_panel.roi_selected.emit(1, 20, 1, 30)
    assert controller._get_state().gridx == (1, 20)


def test_ml_examples_are_not_restored_without_their_results(gui, make_sequence, tmp_path):
    """Strokes and confirmed circles belong to the frames they were taught on."""
    from bubbletrack.model.state import update_state
    from bubbletrack.model.teach.examples import MLExamples

    _window, controller = gui
    folder = make_sequence("a", 6)
    controller.file_ctrl.on_folder_selected(str(folder))
    controller._get_state().radius[:] = 30.0
    controller._set_state(update_state(controller._get_state(), ml_assist=True,
                                       ml_examples=MLExamples().verify(2, (60.0, 80.0, 30.0))))
    path = tmp_path / "s.brt"
    assert controller.save_session_file(str(path))
    data = json.loads(path.read_text(encoding="utf-8"))
    data["image_names"] = [f"other_{i}.tif" for i in range(6)]  # another shot of the camera
    path.write_text(json.dumps(data), encoding="utf-8")
    assert controller.load_session_file(str(path))
    state = controller._get_state()
    assert np.all(state.radius < 0)
    assert state.ml_examples is None and not state.ml_assist


def test_an_unreadable_frame_keeps_its_guess(gui, make_sequence):
    from bubbletrack.model.suggestion import Suggestion, shown_suggestion

    _window, controller = gui
    controller.file_ctrl.on_folder_selected(str(make_sequence("a", 6)))
    state = controller._get_state()
    guess = Suggestion(23.5, 60.0, 80.0, ((50.0, 60.0), (70.0, 60.0), (60.0, 70.0)), 0.2,
                       frozenset({"low_confidence"}), ())
    state.radius[3] = -1.0
    state.fit_suggestion[3] = guess
    state.fit_confidence[3] = 0.2
    auto = controller.auto_ctrl
    auto._on_worker_frame_error(auto._worker, 3, "file is locked")
    auto._on_auto_frame_done(3, -1.0, np.empty((0, 2)), None, np.nan, np.nan, 0)
    assert shown_suggestion(state, 3) is guess and state.fit_confidence[3] == 0.2


def test_a_collapsed_filmstrip_forgets_the_previous_recording(gui, make_sequence):
    _window, controller = gui
    strip = controller.filmstrip._strip
    controller.file_ctrl.on_folder_selected(str(make_sequence("a", 12, r_min=10, r_max=40)))
    strip.set_expanded(True)
    controller.filmstrip.refresh(force=True)
    strip.set_expanded(False)
    controller.file_ctrl.on_folder_selected(str(make_sequence("b", 5)))
    assert strip.frames() == []


def test_teach_works_without_a_reference_frame(gui, make_sequence, monkeypatch):
    """A recording that starts with a bubble has no automatic reference frame:
    "Teach..." raised TypeError (None among the frames to skip)."""
    import pytest

    pytest.importorskip("sklearn")
    from bubbletrack.controller import edge_support
    from bubbletrack.model.state import update_state

    _window, controller = gui
    controller.file_ctrl.on_folder_selected(str(make_sequence("a", 9, r_min=15, r_max=40)))
    controller._set_state(update_state(controller._get_state(), detection_mode="edge"))
    state = controller._get_state()
    state.radius[:] = np.linspace(15, 40, 9)
    state.circle_fit_par[:] = [60.0, 80.0]
    monkeypatch.setattr(edge_support, "reference_index", lambda _state: None)
    controller.ml.open_teaching()  # must not raise
    assert controller.ml.teach.slots
