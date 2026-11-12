"""Fit All in Edge mode ends with sequence tracking: repaired frames are
stored as TRACKED, manual/locked frames anchor the track and stay as they
are, and the solver's review list reaches the Review queue."""

from __future__ import annotations

import time

import cv2
import numpy as np
import pytest

from _sequence_synth import drift_spec, render_seq
from bubbletrack.controller.tracking import apply_tracking, review_items
from bubbletrack.event_bus import RESULTS_CHANGED
from bubbletrack.model.provenance import FitSource
from bubbletrack.model.sequence import FrameSolution, Observation, SequenceResult
from bubbletrack.model.sequence_pipeline import (
    Redetector, TrackingStopped, anchors, observations,
)
from bubbletrack.model.state import AppState, update_state


def _state(n=5):
    state = AppState().with_results_initialized(n)
    state.radius[:] = [20.0, 21.0, -1.0, 23.0, 24.0]
    state.circle_fit_par[:] = [[100.0, 150.0]] * n
    state.circle_fit_par[2] = [np.nan, np.nan]
    state.fit_confidence[:] = [0.9, 0.8, 0.2, 0.7, 0.9]
    return state


# --------------------------------------------------------------------------- #
#  Snapshot: results -> solver inputs
# --------------------------------------------------------------------------- #

def test_observations_mark_unfitted_frames_as_missing():
    obs = observations(_state())
    assert [o.ok for o in obs] == [True, True, False, True, True]
    assert obs[2].confidence == pytest.approx(0.2)  # rejected, but scored
    assert obs[1].circle == (100.0, 150.0, 21.0)


def test_manual_and_locked_frames_are_anchors():
    state = _state()
    state.fit_source[1] = FitSource.MANUAL
    state.fit_locked[2] = True  # accepted "no bubble here"
    state.fit_locked[4] = True
    found = anchors(state)
    assert set(found) == {1, 2, 4}
    assert found[1] == (100.0, 150.0, 21.0)
    assert all(np.isnan(v) for v in found[2])


# --------------------------------------------------------------------------- #
#  Re-detection and storing the solver's choice
# --------------------------------------------------------------------------- #

class _FakeFit:
    def __init__(self, radius, rc=100.0, cc=150.0):
        self.radius, self.rc, self.cc = radius, rc, cc
        self.confidence, self.flags = 0.8, frozenset({"non_circular"})
        self.edge_xy = np.array([[1.0, 2.0]] * 3)
        self.metrics = {"r_circle": radius + 1}
        self.mask = None


def _redetector(monkeypatch, fits):
    import bubbletrack.model.sequence_pipeline as module

    calls = []

    def fake_fit_edge(raw, gridx, gridy, job):
        calls.append(job.params)
        return fits.pop(0)
    monkeypatch.setattr(module, "fit_edge", fake_fit_edge)
    from bubbletrack.model.edge_pipeline import EdgeJob, make_params

    job = EdgeJob(make_params(seed=(5.0, 5.0)), None, 500.0)
    return Redetector(lambda i: np.zeros((4, 4)), (1, 4), (1, 4), job), calls


def test_redetector_uses_the_prior_instead_of_the_click(monkeypatch):
    redetect, calls = _redetector(monkeypatch, [_FakeFit(22.0)])
    obs = redetect(2, (100.0, 150.0, 22.0))
    assert obs.ok and obs.radius == 22.0
    assert calls[0].prior == (100.0, 150.0, 22.0) and calls[0].seed is None


def test_redetector_stops_when_asked(monkeypatch):
    redetect, _ = _redetector(monkeypatch, [_FakeFit(22.0)])
    redetect._stop_requested = lambda: True
    with pytest.raises(TrackingStopped):
        redetect(0, (1.0, 1.0, 1.0))


def _outcome(redetect, solutions):
    from bubbletrack.controller.tracking import TrackingOutcome

    frames = tuple(FrameSolution(r, 100.0, 150.0, 0.8, source, frozenset(), 0)
                   for r, source in solutions)
    result = SequenceResult(frames, (), ((0, len(frames) - 1),),
                            ((2, ("lost",)), (3, ("jump", "disagree"))), 0)
    return TrackingOutcome(result, redetect)


def test_only_tracked_frames_are_stored_and_anchors_are_kept(monkeypatch):
    redetect, _ = _redetector(monkeypatch, [_FakeFit(22.0), _FakeFit(23.5)])
    redetect(2, (100.0, 150.0, 22.0))
    redetect(3, (100.0, 150.0, 23.5))
    state = _state()
    state.fit_source[3] = FitSource.MANUAL  # a manual fix the solver must not touch
    outcome = _outcome(redetect, [(20.0, "raw"), (21.0, "raw"), (22.0, "tracked"),
                                  (23.5, "tracked"), (24.0, "lost")])
    assert apply_tracking(state, outcome) == [2]
    assert state.radius[2] == 22.0 and state.fit_source[2] == FitSource.TRACKED
    assert state.fit_metrics["r_circle"][2] == 23.0 and "non_circular" in state.fit_flags[2]
    assert state.radius[3] == 23.0 and state.fit_source[3] == FitSource.MANUAL
    assert state.radius[4] == 24.0  # lost: kept, goes to review instead


def test_review_items_carry_the_solver_reasons(monkeypatch):
    redetect, _ = _redetector(monkeypatch, [])
    items = review_items(_outcome(redetect, [(20.0, "raw")] * 5).result)
    assert [(i.frame, i.reasons) for i in items] == [(2, ("lost",)), (3, ("jump", "disagree"))]


# --------------------------------------------------------------------------- #
#  End to end through the GUI (real engine, real worker threads)
# --------------------------------------------------------------------------- #

def _wait(qapp, predicate, timeout_s: float = 240.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        qapp.processEvents()
        if predicate():
            return
        time.sleep(0.01)
    raise TimeoutError("condition not reached")


@pytest.fixture(scope="module")
def drift_sequence(tmp_path_factory):
    """drift 20 px/frame with satellites: a click helps only near its frame."""
    frames, truth = render_seq(drift_spec(20.0))
    folder = tmp_path_factory.mktemp("drift20")
    for i, frame in enumerate(frames):
        assert cv2.imwrite(str(folder / f"f_{i:03d}.tiff"), frame)
    return folder, truth


def _fit_all(qapp, gui, folder, truth, *, track: bool):
    window, controller = gui
    controller.file_ctrl.on_folder_selected(str(folder))
    peak = int(np.nanargmax(truth["R"]))
    controller._state = update_state(controller._get_state(), detection_mode="edge",
                                     reference_frame=0,
                                     edge_seed=(truth["rc"][peak], truth["cc"][peak]))
    window.left_panel.automatic_tab._track_check.setChecked(track)
    events = []
    controller.bus.subscribe(RESULTS_CHANGED, lambda *a: events.append(a))
    controller.auto_ctrl.on_auto_fit()
    _wait(qapp, lambda: not controller.auto_ctrl.is_busy())
    return controller._get_state(), events


def _correct(state, truth) -> int:
    ok = 0
    for t in range(1, len(state.radius)):
        r = truth["R"][t]
        if not np.isfinite(r) or r < 3:
            continue
        rc, cc = state.circle_fit_par[t]
        if state.radius[t] > 0 and abs(state.radius[t] - r) <= max(1.0, 0.03 * r) \
                and np.hypot(rc - truth["rc"][t], cc - truth["cc"][t]) <= max(2.0, 0.1 * r):
            ok += 1
    return ok


def test_tracking_repairs_frames_a_click_cannot(qapp, gui, drift_sequence):
    folder, truth = drift_sequence
    window, controller = gui
    single, _ = _fit_all(qapp, gui, folder, truth, track=False)
    single_ok = _correct(single, truth)
    assert not (single.fit_source == FitSource.TRACKED).any()

    tracked, events = _fit_all(qapp, gui, folder, truth, track=True)
    assert _correct(tracked, truth) >= single_ok + 4
    assert (tracked.fit_source == FitSource.TRACKED).any()
    assert "tracking re-measured" in window.header._status_label.text()
    assert events  # autosave and views were told


def test_manual_fix_anchors_tracking_and_survives(qapp, gui, drift_sequence):
    folder, truth = drift_sequence
    window, controller = gui
    _fit_all(qapp, gui, folder, truth, track=False)
    state = controller._get_state()
    state.radius[5], state.circle_fit_par[5] = 99.0, [10.0, 10.0]  # deliberately odd
    state.fit_source[5] = FitSource.MANUAL
    controller.auto_ctrl._ask_manual_policy = lambda n: "keep"
    window.left_panel.automatic_tab._track_check.setChecked(True)
    controller.auto_ctrl.on_auto_fit()
    _wait(qapp, lambda: not controller.auto_ctrl.is_busy())
    state = controller._get_state()
    assert state.radius[5] == 99.0 and state.fit_source[5] == FitSource.MANUAL


def test_stop_during_tracking_keeps_single_frame_results(qapp, gui, drift_sequence, monkeypatch):
    folder, truth = drift_sequence
    window, controller = gui
    import bubbletrack.controller.auto_controller as auto

    started = []
    original = auto.AutoController._start_tracking

    def start_then_stop(self):
        original(self)
        started.append(True)
        self.on_auto_stop()
    monkeypatch.setattr(auto.AutoController, "_start_tracking", start_then_stop)
    state, _ = _fit_all(qapp, gui, folder, truth, track=True)
    assert started
    assert not (state.fit_source == FitSource.TRACKED).any()
    assert "tracking stopped" in window.header._status_label.text()


def test_tracking_review_reaches_the_queue_and_expires_on_edit(qapp, gui, drift_sequence):
    from bubbletrack.controller.review_controller import ReviewItem

    folder, truth = drift_sequence
    window, controller = gui
    _fit_all(qapp, gui, folder, truth, track=False)
    rc = controller.review
    state = controller._get_state()
    state.fit_confidence[:] = 0.99  # nothing suspicious of its own
    state.fit_flags[:] = [frozenset()] * len(state.radius)
    controller.bus.emit("tracking_review", (ReviewItem(7, ("lost",), 0),))
    assert 7 in rc.panel.frames()
    controller._on_delete_point(7)  # the user dealt with it
    assert 7 not in rc.panel.frames()
