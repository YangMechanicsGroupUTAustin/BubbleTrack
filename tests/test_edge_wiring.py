"""Wiring of the Edge detection engine into the GUI, worker and persistence.

The engine itself is replaced by a recording fake so these tests check the
data flow only (parameters in, radius/confidence out); the engine has its
own accuracy tests.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pytest

from bubbletrack.controller import worker as worker_module
from bubbletrack.model import edge_pipeline
from bubbletrack.model.provenance import FitSource
from bubbletrack.model.state import update_state


@dataclass
class FakeResult:
    radius: float = 40.0
    rc: float = 125.0
    cc: float = 200.0
    confidence: float = 0.9
    flags: frozenset = frozenset()
    edge_xy: np.ndarray = field(default_factory=lambda: np.array(
        [[85.0, 200.0], [125.0, 240.0], [165.0, 200.0], [125.0, 160.0]]))
    mask: np.ndarray = field(default_factory=lambda: np.zeros((250, 400), bool))
    gate: float = 0.3

    @property
    def accepted(self) -> bool:
        return np.isfinite(self.radius) and self.confidence >= self.gate


class FakeEngine:
    def __init__(self, result: FakeResult | None = None):
        self.result = result or FakeResult()
        self.calls: list[dict] = []

    def __call__(self, raw, gridx, gridy, *, reference=None, params=None):
        self.calls.append({"raw": raw, "reference": reference, "params": params,
                           "gridx": gridx, "gridy": gridy})
        return self.result


@dataclass(frozen=True)
class FakeParams:
    edge_definition: str = "steep"
    polarity: str = "dark"
    seed: tuple | None = None
    prior: tuple | None = None


@pytest.fixture
def fake_engine(monkeypatch):
    engine = FakeEngine()
    monkeypatch.setattr(edge_pipeline, "_engine", lambda: engine)
    monkeypatch.setattr(edge_pipeline, "_params_class", lambda: FakeParams)
    monkeypatch.setattr(edge_pipeline, "_choose_reference", lambda read_raw, n: 1)
    monkeypatch.setattr(worker_module.BatchWorker, "inline", True)
    return engine


def _edge_mode(controller, **extra):
    controller._state = update_state(controller._get_state(), detection_mode="edge", **extra)


def _load(controller, folder):
    controller.file_ctrl.on_folder_selected(str(folder))


def _status(window):
    return window.header._status_label.text()


def _wait(qapp, predicate, timeout_s=60.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        qapp.processEvents()
        if predicate():
            return
        time.sleep(0.01)
    raise TimeoutError


# -- UI ---------------------------------------------------------------------------------

class TestPretuneTab:
    def test_offers_edge_method_and_round_trips(self, qapp):
        from bubbletrack.ui.pretune_tab import PreTuneTab

        tab = PreTuneTab()
        tab.set_filter_params({"detection_mode": "edge", "edge_definition": "outer",
                               "edge_polarity": "bright"})
        params = tab.get_filter_params()
        assert params["detection_mode"] == "edge"
        assert params["edge_definition"] == "outer"
        assert params["edge_polarity"] == "bright"

    def test_threshold_controls_disabled_in_edge_mode(self, qapp):
        from bubbletrack.ui.pretune_tab import PreTuneTab

        tab = PreTuneTab()
        tab.set_filter_params({"detection_mode": "edge"})
        assert not tab._threshold.isEnabled()
        assert not tab._removing_factor.isEnabled()
        tab.set_filter_params({"detection_mode": "intensity"})
        assert tab._threshold.isEnabled()


# -- single-frame paths -------------------------------------------------------------------

class TestSingleFrame:
    def test_fit_current_frame_uses_engine(self, gui, make_sequence, fake_engine):
        window, controller = gui
        _load(controller, make_sequence("A", 6))
        _edge_mode(controller, edge_definition="outer", edge_seed=(120.0, 190.0))
        controller.file_ctrl.on_frame_changed(4)
        controller.pretune_ctrl.on_pretune_fit()

        state = controller._get_state()
        assert state.radius[4] == 40.0
        assert state.fit_confidence[4] == pytest.approx(0.9)
        assert state.fit_source[4] == FitSource.REFIT
        call = fake_engine.calls[-1]
        assert call["params"].edge_definition == "outer"
        assert call["params"].seed == (120.0, 190.0)
        assert call["reference"] is not None  # auto reference (frame 1 via the patched chooser)
        assert call["raw"].dtype == np.uint16  # raw data, not the 8-bit threshold image

    def test_low_confidence_is_rejected_but_recorded(self, gui, make_sequence, fake_engine):
        window, controller = gui
        fake_engine.result = FakeResult(confidence=0.1)
        _load(controller, make_sequence("A", 4))
        _edge_mode(controller)
        controller.pretune_ctrl.on_pretune_fit()
        state = controller._get_state()
        assert state.radius[0] == -1.0
        assert state.fit_confidence[0] == pytest.approx(0.1)
        assert "confidence" in _status(window).lower()

    def test_explicit_reference_frame(self, gui, make_sequence, fake_engine):
        window, controller = gui
        _load(controller, make_sequence("A", 6))
        _edge_mode(controller)
        controller.file_ctrl.on_frame_changed(2)
        controller.pretune_ctrl.on_use_current_as_reference()
        assert controller._get_state().reference_frame == 2
        controller.file_ctrl.on_frame_changed(4)
        controller.pretune_ctrl.on_pretune_fit()
        ref = fake_engine.calls[-1]["reference"]
        expected = edge_pipeline.read_raw(controller._get_state().images[2])
        np.testing.assert_array_equal(ref, expected)

    def test_seed_pick_records_click_and_not_manual_points(self, gui, make_sequence):
        window, controller = gui
        _load(controller, make_sequence("A", 3))
        _edge_mode(controller)
        controller.pretune_ctrl.on_pick_seed()
        window.original_panel.point_clicked.emit(199.5, 124.5)  # scene x, y
        assert controller._get_state().edge_seed == pytest.approx((125.0, 200.0))
        assert controller.manual_ctrl._manual_points == []

    def test_chart_refit_and_neighbor_refit_use_engine(self, gui, make_sequence, fake_engine):
        _, controller = gui
        _load(controller, make_sequence("A", 6))
        _edge_mode(controller)
        controller._on_refit_point(2)
        assert controller._get_state().radius[2] == 40.0

        state = controller._get_state()
        state.radius[1], state.radius[3] = 30.0, 34.0
        state.circle_fit_par[1] = state.circle_fit_par[3] = [120.0, 190.0]
        controller._on_refit_point_with_neighbors(2)
        prior = fake_engine.calls[-1]["params"].prior
        assert prior == pytest.approx((120.0, 190.0, 32.0))
        assert controller._get_state().fit_source[2] == FitSource.NEIGHBOR


# -- Fit All -------------------------------------------------------------------------------

def test_fit_all_edge_mode(qapp, gui, make_sequence, fake_engine):
    window, controller = gui
    _load(controller, make_sequence("A", 8))
    _edge_mode(controller)
    controller.auto_ctrl.on_auto_fit()
    _wait(qapp, lambda: window.left_panel.automatic_tab._fit_btn.isEnabled())
    state = controller._get_state()
    assert (state.radius == 40.0).all(), _status(window)
    np.testing.assert_allclose(state.fit_confidence, 0.9)
    assert (state.fit_source == FitSource.AUTO).all()
    assert all(c["reference"] is not None for c in fake_engine.calls)


def test_chart_scores_are_confidence_in_edge_mode(gui, make_sequence):
    from bubbletrack.controller.display_mixin import refresh_chart

    window, controller = gui
    _load(controller, make_sequence("A", 4))
    _edge_mode(controller)
    state = controller._get_state()
    state.radius[:] = 30.0
    state.fit_confidence[:] = [0.2, 0.5, 0.8, 1.0]
    scores = refresh_chart(state, window)
    np.testing.assert_allclose(scores, [0.2, 0.5, 0.8, 1.0])


# -- persistence ---------------------------------------------------------------------------

def test_session_round_trip_edge_fields(gui, make_sequence, tmp_path):
    _, controller = gui
    _load(controller, make_sequence("A", 4))
    _edge_mode(controller, edge_definition="outer", edge_polarity="bright",
               reference_frame=1, edge_seed=(100.0, 150.0))
    controller._get_state().fit_confidence[:] = [0.1, 0.2, np.nan, 0.4]
    path = tmp_path / "e.brt"
    assert controller.save_session_file(str(path))
    _load(controller, make_sequence("B", 2))
    assert controller.load_session_file(str(path))
    state = controller._get_state()
    assert state.detection_mode == "edge"
    assert (state.edge_definition, state.edge_polarity) == ("outer", "bright")
    assert state.reference_frame == 1
    assert state.edge_seed == (100.0, 150.0)
    np.testing.assert_allclose(state.fit_confidence, [0.1, 0.2, np.nan, 0.4])


def test_config_persists_edge_settings(isolated_config):
    from bubbletrack.model.config import load_config, save_config
    from bubbletrack.model.state import AppState

    save_config(AppState(detection_mode="edge", edge_definition="outer", edge_polarity="bright"))
    cfg = load_config()
    assert cfg["detection_mode"] == "edge"
    assert cfg["edge_definition"] == "outer"
    assert cfg["edge_polarity"] == "bright"
