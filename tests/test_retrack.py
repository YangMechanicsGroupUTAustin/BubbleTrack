"""Automatic re-tracking after result edits (Auto mode): debounced, quiet (locked
like a run only when it runs long), cancelled by a new edit, a Fit or Stop,
limited to frames that do not look good, and undone in one Ctrl+Z."""

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


def test_edit_starts_a_quiet_retrack_after_the_debounce(qapp, app):
    """A short re-track only says "Updating…": Fit stays usable, no red Stop."""
    window, controller = app
    controller.retrack.escalate_timer.setInterval(60_000)
    _edit(controller)
    header = window.header.status_text()  # what the edit said
    assert not FakeWorker.made  # debounced
    worker = _started(qapp)
    lp = window.left_panel
    assert controller.retrack.is_running()
    assert window.status_bar.activity_text() == "Updating…"
    assert lp.measure_panel.isEnabled() and lp.fit_controls.fit_button.isEnabled()
    assert lp.fit_controls.progress_row.isHidden()
    assert window.header.status_text() == header  # nothing to report yet
    worker.finish()
    assert not controller.retrack.is_running() and window.status_bar.activity_text() == ""


def test_a_long_retrack_locks_editing_and_offers_stop(qapp, app):
    window, controller = app
    controller.retrack.escalate_timer.setInterval(20)
    _edit(controller)
    worker = _started(qapp)
    lp = window.left_panel
    _wait(qapp, lambda: not lp.measure_panel.isEnabled())
    assert controller.review.panel.isEnabled()  # Review goes on: an edit restarts it
    assert lp.fit_controls.stop_button.isVisibleTo(lp) and lp.fit_controls.stop_button.isEnabled()
    assert "Re-tracking" in lp.fit_controls.progress_text()
    worker.finish()
    assert not controller.retrack.is_running() and lp.measure_panel.isEnabled()
    assert lp.fit_controls.progress_row.isHidden()


def test_fit_during_a_quiet_retrack_cancels_it(qapp, app):
    window, controller = app
    controller.retrack.escalate_timer.setInterval(60_000)
    _edit(controller)
    worker = _started(qapp)
    controller.bus.emit("run_state_changed", True)  # what Fit announces when it starts
    assert worker.stop_requested and not controller.retrack.is_running()
    controller.bus.emit("run_state_changed", False)


def test_a_retrack_that_changes_nothing_keeps_the_header(qapp, app):
    window, controller = app
    controller.retrack.escalate_timer.setInterval(60_000)
    window.header.set_status("Frame 2 fixed", "#10b981")
    controller.retrack._on_results_changed()
    worker = _started(qapp)
    controller.retrack._allowed = frozenset()  # the solve may change no frame
    worker.finish()
    assert window.header.status_text() == "Frame 2 fixed"


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
    assert "Tracking updated 2 nearby frames" in text and "Ctrl+Z" in text


def test_ctrl_z_undoes_the_whole_retrack(qapp, app):
    window, controller = app
    state = controller._get_state()
    _edit(controller)
    before = {i: capture(state, i) for i in range(N)}
    _started(qapp).finish()
    controller._on_undo()
    assert all(same_result(capture(state, i), before[i]) for i in range(N))
    assert "Undid the re-track" in window.header._status_label.text()
    top = controller.history.peek()
    _settle(qapp)
    # An undo only recomputes the Review list from the restored results: the
    # solve it starts may change no frame and adds nothing to the history.
    assert len(FakeWorker.made) == 2 and controller.retrack._allowed == frozenset()
    FakeWorker.made[-1].finish()
    assert all(same_result(capture(state, i), before[i]) for i in range(N))
    assert controller.history.peek() is top


def test_undoing_the_edit_during_a_retrack_drops_its_result(qapp, app):
    window, controller = app
    state = controller._get_state()
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
    controller.retrack.escalate_timer.setInterval(20)
    worker = _started(qapp)
    _wait(qapp, lambda: window.left_panel.fit_controls.stop_button.isEnabled())
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
    assert all(same_result(capture(state, i), before[i]) for i in good)


# -- held while a fix is open or a question is asked (2026-11-18 Review bug hunt) ----- #

def test_a_fix_holds_the_countdown_and_esc_releases_it(qapp, app):
    """Fix + Esc used to cancel the pending re-track for good."""
    _window, controller = app
    controller.retrack.timer.setInterval(60_000)  # stays pending while we look
    _edit(controller)
    assert controller.retrack.timer.isActive()
    controller.file_ctrl.on_frame_changed(2)
    assert controller.review.fix.start(2)
    assert not controller.retrack.timer.isActive()  # nothing may land under the points
    controller._on_lock_range(3, 3, locked=True)  # an edit during the fix: still held
    assert not controller.retrack.timer.isActive()
    controller.review.fix.cancel()
    assert controller.retrack.timer.isActive()  # released: it runs after the quiet time


def test_a_hold_abandons_a_running_retrack_and_restarts_it(qapp, app):
    """Looks right (ML assist) or a question: the running solve is stale, so it is
    dropped and started again when the hold ends."""
    _window, controller = app
    _edit(controller)
    worker = _started(qapp)
    controller.retrack.hold()
    assert worker.stop_requested and not controller.retrack.is_running()
    controller.retrack.release_hold()
    _wait(qapp, lambda: len(FakeWorker.made) == 2 and FakeWorker.made[-1].started)


def test_skipped_frames_stay_out_of_a_running_retrack(qapp, app):
    _window, controller = app
    state = controller._get_state()
    _edit(controller)
    worker = _started(qapp)
    assert 4 in controller.retrack._allowed
    r4 = float(state.radius[4])
    controller.retrack.exclude([4])  # Review's Skip on the row of frame 5
    worker.finish()
    assert state.radius[4] == r4  # "left as it is"


def test_a_partial_run_keeps_review_progress(qapp, app):
    """A range refit or Train and apply used to reset "All 3 checked"."""
    from bubbletrack.event_bus import RUN_FINISHED

    _window, controller = app
    resets = []
    controller.review.panel.reset_progress = lambda: resets.append(1)
    controller.bus.emit(RUN_FINISHED, {"frames": (3, 4, 5), "edge": True})
    assert resets == []
    controller.bus.emit(RUN_FINISHED, {"frames": tuple(range(N)), "edge": True, "ml": True})
    assert resets == [1]


def test_an_undo_during_a_hold_updates_the_list_when_released(qapp, app):
    """Ctrl+Z while a fix holds re-tracking: at release only the Review list is
    recomputed (a full re-track could redo the re-track just undone)."""
    _window, controller = app
    rt = controller.retrack
    _edit(controller)
    _started(qapp).finish()  # a re-track lands (one history entry)
    rt.hold()
    controller._on_undo()  # undo the re-track while held
    assert not rt.timer.isActive()
    rt.release_hold()
    _wait(qapp, lambda: len(FakeWorker.made) == 2 and FakeWorker.made[-1].started)
    assert rt._allowed == frozenset()  # list only
    FakeWorker.made[-1].finish()


def test_a_hold_during_the_list_only_solve_keeps_it_list_only(qapp, app):
    """A hold (F, a question) while the post-undo list solve runs must not turn it
    into a full re-track at release: that would redo the re-track just undone."""
    _window, controller = app
    rt = controller.retrack
    _edit(controller)
    _started(qapp).finish()  # a re-track lands
    controller._on_undo()  # ... and is undone: a list-only solve starts
    _wait(qapp, lambda: len(FakeWorker.made) == 2 and FakeWorker.made[-1].started)
    rt.hold()
    rt.release_hold()
    _wait(qapp, lambda: len(FakeWorker.made) == 3 and FakeWorker.made[-1].started)
    assert rt._allowed == frozenset()  # still list-only
