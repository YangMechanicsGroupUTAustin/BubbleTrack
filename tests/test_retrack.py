"""Automatic re-tracking after result edits (Auto mode): debounced, locked like
a run, cancelled by a new edit or Stop, limited to frames that do not look
good, and undone in one Ctrl+Z."""

from __future__ import annotations

import time

import cv2
import numpy as np
import pytest
from PyQt6.QtCore import QObject, pyqtSignal

from _sequence_synth import drift_spec, render_seq
from bubbletrack.controller import retrack_controller
from bubbletrack.controller.tracking import TrackingOutcome, TrackingWorker
from bubbletrack.event_bus import RESULTS_CHANGED
from bubbletrack.model.edge_pipeline import EdgeFrameFit
from bubbletrack.model.provenance import FitSource
from bubbletrack.model.result_history import capture, same_result
from bubbletrack.model.sequence import FrameSolution, SequenceResult
from bubbletrack.model.sequence_pipeline import retrack_frames
from bubbletrack.model.state import AppState, update_state

N = 8

# --------------------------------------------------------------------------- #
#  Which frames a re-track may change (pure)
# --------------------------------------------------------------------------- #


def _results(n=N):
    state = AppState().with_results_initialized(n)
    state.radius[:] = np.linspace(30.0, 40.0, n)
    state.circle_fit_par[:] = [[100.0, 150.0]] * n
    state.fit_source[:] = FitSource.AUTO
    state.fit_confidence[:] = 0.9
    return state


def test_retrack_frames_are_unfitted_or_queued_frames_only():
    state = _results()
    state.radius[2] = -1.0           # no result
    state.radius[6] = -1.0
    state.fit_locked[6] = True       # accepted "no bubble here"
    state.fit_source[5] = FitSource.GUIDED
    state.fit_source[1] = FitSource.MANUAL
    assert retrack_frames(state, queued=[4, 5, 1]) == frozenset({2, 4})


# --------------------------------------------------------------------------- #
#  GUI: a fake solver worker that finishes when the test says so
# --------------------------------------------------------------------------- #

class FakeWorker(QObject):
    solved = pyqtSignal(object)
    failed = pyqtSignal(str)
    stopped = pyqtSignal()
    made: list["FakeWorker"] = []

    def __init__(self, state, job) -> None:
        super().__init__()
        self.n = len(state.radius)
        self.started = self.stop_requested = False
        FakeWorker.made.append(self)

    def start(self):
        self.started = True

    def request_stop(self):
        self.stop_requested = True

    def isRunning(self):  # noqa: N802 (QThread API)
        return False

    def wait(self, _ms=0):
        return True

    def finish(self, delta=5.0):
        """The solver re-measured every frame (radius + delta)."""
        self.solved.emit(_outcome(self.n, delta))


class _Redetector:
    def fit_for(self, idx, solution):
        return EdgeFrameFit(solution.radius, 100.0, 150.0, np.zeros((0, 2)), None, 0.8,
                            frozenset(), "")


def _outcome(n, delta):
    radii = np.linspace(30.0, 40.0, n) + delta
    frames = tuple(FrameSolution(float(r), 100.0, 150.0, 0.8, "tracked", frozenset(), 0)
                   for r in radii)
    return TrackingOutcome(SequenceResult(frames, (), ((0, n - 1),), ((2, ("lost",)),), 0),
                           _Redetector())


def _wait(qapp, predicate, timeout_s=10.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        qapp.processEvents()
        if predicate():
            return
        time.sleep(0.005)
    raise TimeoutError("condition not reached")


def _settle(qapp, seconds=0.15):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.005)


@pytest.fixture
def app(qapp, gui, make_sequence):
    window, controller = gui
    controller.file_ctrl.on_folder_selected(str(make_sequence("seq", N)))
    controller._state = update_state(controller._get_state(), detection_mode="edge",
                                     reference_frame=0)
    state = controller._get_state()
    fitted = _results()
    for name in ("radius", "circle_fit_par", "fit_source", "fit_confidence"):
        getattr(state, name)[:] = getattr(fitted, name)
    state.fit_confidence[4] = 0.1  # listed for review
    state.radius[5], state.circle_fit_par[5] = -1.0, [np.nan, np.nan]  # no result
    controller.review.refresh()
    FakeWorker.made = []
    rt = controller.retrack
    rt.worker_factory = FakeWorker
    rt.timer.setInterval(20)
    _settle(qapp)  # leave the quiet turn after loading
    return window, controller


def _edit(controller):
    """A user edit elsewhere: lock frame 1."""
    controller._on_lock_range(1, 1, locked=True)


def _started(qapp):
    _wait(qapp, lambda: FakeWorker.made and FakeWorker.made[-1].started)
    return FakeWorker.made[-1]


def test_edit_starts_a_retrack_after_the_debounce_and_locks_editing(qapp, app):
    window, controller = app
    _edit(controller)
    assert not FakeWorker.made  # debounced
    worker = _started(qapp)
    lp = window.left_panel
    assert controller.retrack.is_running()
    assert not lp.measure_panel.isEnabled() and not controller.review.panel.isEnabled()
    assert lp.manual_tab.isEnabled()  # fixes there cancel and restart the re-track
    assert lp.fit_controls.stop_button.isVisibleTo(lp) and lp.fit_controls.stop_button.isEnabled()
    assert "Re-tracking" in lp.fit_controls.progress_text()
    worker.finish()
    assert not controller.retrack.is_running() and lp.measure_panel.isEnabled()
    assert lp.fit_controls.progress_row.isHidden()


def test_only_unfitted_and_queued_frames_change(qapp, app):
    window, controller = app
    state = controller._get_state()
    before = {i: capture(state, i) for i in range(N)}
    _edit(controller)
    _started(qapp).finish()
    changed = [i for i in range(N) if not same_result(capture(state, i), before[i])]
    assert changed == [1, 4, 5]  # 1 = the edit itself (locked); 4 queued, 5 unfitted
    assert state.fit_locked[1] and state.radius[1] == before[1].radius
    assert state.fit_source[4] == state.fit_source[5] == FitSource.TRACKED
    text = window.header._status_label.text()
    assert "Re-tracked: 2 frame(s) updated" in text and "Ctrl+Z" in text


def test_ctrl_z_undoes_the_whole_retrack(qapp, app):
    window, controller = app
    state = controller._get_state()
    _edit(controller)
    before = {i: capture(state, i) for i in range(N)}
    _started(qapp).finish()
    controller._on_undo()
    assert all(same_result(capture(state, i), before[i]) for i in range(N))
    assert "Undid the re-track" in window.header._status_label.text()
    _settle(qapp)
    assert len(FakeWorker.made) == 1  # an undo does not re-track again


def test_undoing_the_edit_during_a_retrack_drops_its_result(qapp, app):
    window, controller = app
    state = controller._get_state()
    window.left_panel.tab_bar.select("Manual")
    controller.file_ctrl.on_frame_changed(3)
    before = {i: capture(state, i) for i in range(N)}
    controller.circle_ctrl._store_guided(3, EdgeFrameFit(33.0, 100.0, 150.0, np.zeros((0, 2)),
                                                         None, 0.9, frozenset(), ""))
    worker = _started(qapp)
    controller._on_undo()  # undo the circle fix while its re-track runs
    assert worker.stop_requested and not controller.retrack.is_running()
    worker.finish()  # too late: ignored
    assert all(same_result(capture(state, i), before[i]) for i in range(N))


def test_nothing_to_undo_keeps_a_pending_retrack(qapp, app):
    _, controller = app
    _edit(controller)
    controller._on_undo()  # the lock is not in the undo history
    _started(qapp)


def test_retrack_results_do_not_trigger_another_retrack(qapp, app):
    _, controller = app
    _edit(controller)
    _started(qapp).finish()
    _settle(qapp)
    assert len(FakeWorker.made) == 1


def test_review_receives_the_solver_list(qapp, app):
    _, controller = app
    seen = []
    controller.bus.subscribe("tracking_review", seen.append)
    _edit(controller)
    _started(qapp).finish()
    assert [item.frame for item in seen[0]] == [2]


def test_edit_during_a_retrack_cancels_and_restarts_it(qapp, app):
    window, controller = app
    state = controller._get_state()
    _edit(controller)
    first = _started(qapp)
    controller._on_lock_range(2, 2, locked=True)  # edited again (e.g. a circle fix)
    assert first.stop_requested and not controller.retrack.is_running()
    assert window.left_panel.measure_panel.isEnabled()
    radius_4 = state.radius[4]
    first.finish()  # a late result of the stale solve is ignored
    assert state.radius[4] == radius_4
    _wait(qapp, lambda: len(FakeWorker.made) == 2 and FakeWorker.made[1].started)
    assert controller.retrack.is_running()


def test_stop_cancels_the_retrack(qapp, app):
    window, controller = app
    state = controller._get_state()
    _edit(controller)
    worker = _started(qapp)
    window.left_panel.fit_controls.stop_button.click()
    assert worker.stop_requested and not controller.retrack.is_running()
    assert "Re-track stopped" in window.header._status_label.text()
    worker.finish()
    assert state.fit_source[4] == FitSource.AUTO


def _no_retrack(qapp, controller):
    _edit(controller)
    _settle(qapp)
    return not FakeWorker.made


def test_switch_off_means_no_retrack(qapp, app, monkeypatch):
    _, controller = app
    monkeypatch.setattr(retrack_controller, "AUTO_RETRACK_ENABLED", False)
    assert _no_retrack(qapp, controller)


def test_tracking_checkbox_off_means_no_retrack(qapp, app):
    window, controller = app
    window.left_panel.measure_panel.edge_box._track_check.setChecked(False)
    assert _no_retrack(qapp, controller)


def test_other_modes_do_not_retrack(qapp, app):
    _, controller = app
    controller._state = update_state(controller._get_state(), detection_mode="intensity")
    assert _no_retrack(qapp, controller)


def test_no_retrack_without_results(qapp, app):
    _, controller = app
    controller._get_state().radius[:] = -1.0
    assert _no_retrack(qapp, controller)


def test_no_retrack_while_a_fit_runs(qapp, app):
    _, controller = app
    controller.auto_ctrl._running = True
    try:
        assert _no_retrack(qapp, controller)
    finally:
        controller.auto_ctrl._running = False


def test_a_finishing_run_is_not_an_edit(qapp, app):
    _, controller = app
    controller.bus.emit("run_state_changed", True)
    controller.bus.emit("run_state_changed", False)
    controller.bus.emit(RESULTS_CHANGED)  # as AutoController._finish_run does
    _settle(qapp)
    assert not FakeWorker.made


def test_a_circle_fix_triggers_a_retrack(qapp, app):
    window, controller = app
    window.left_panel.tab_bar.select("Manual")
    controller.file_ctrl.on_frame_changed(4)
    controller.circle_ctrl._store_guided(4, EdgeFrameFit(36.0, 100.0, 150.0, np.zeros((0, 2)),
                                                         None, 0.9, frozenset(), ""))
    worker = _started(qapp)
    worker.finish()
    state = controller._get_state()
    assert state.fit_source[4] == FitSource.GUIDED and state.radius[4] == 36.0  # an anchor


# --------------------------------------------------------------------------- #
#  End to end: the real solver after a real edit
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def drift_sequence(tmp_path_factory):
    frames, truth = render_seq(drift_spec(20.0))
    folder = tmp_path_factory.mktemp("drift20_retrack")
    for i, frame in enumerate(frames):
        assert cv2.imwrite(str(folder / f"f_{i:03d}.tiff"), frame)
    return folder, truth


def test_real_retrack_after_an_edit_leaves_good_frames_alone(qapp, gui, drift_sequence):
    window, controller = gui
    folder, truth = drift_sequence
    controller.file_ctrl.on_folder_selected(str(folder))
    peak = int(np.nanargmax(truth["R"]))
    controller._state = update_state(controller._get_state(), detection_mode="edge",
                                     reference_frame=0,
                                     edge_seed=(truth["rc"][peak], truth["cc"][peak]))
    window.left_panel.measure_panel.edge_box._track_check.setChecked(False)
    controller.auto_ctrl.on_auto_fit()
    _wait(qapp, lambda: not controller.auto_ctrl.is_busy(), 240.0)
    assert controller.retrack.worker_factory is TrackingWorker
    window.left_panel.measure_panel.edge_box._track_check.setChecked(True)
    controller.retrack.timer.setInterval(20)
    _settle(qapp)

    state = controller._get_state()
    allowed = retrack_frames(state, controller.review.queued_frames())
    good = [i for i in range(len(state.radius)) if i not in allowed and i != 0]
    before = {i: capture(state, i) for i in good}
    controller._on_lock_range(0, 0, locked=True)  # the edit
    _wait(qapp, controller.retrack.is_running, 10.0)
    _wait(qapp, lambda: not controller.retrack.is_running(), 240.0)
    assert "Re-tracked" in window.header._status_label.text()
    assert all(same_result(capture(state, i), before[i]) for i in good)
