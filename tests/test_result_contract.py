"""Shared per-frame result contract used by the engine, sequence solver,
review queue, visualisation and export work streams."""

from __future__ import annotations

import numpy as np
import pytest

from bubbletrack.model.provenance import FitSource
from bubbletrack.model.session import load_session, save_session
from bubbletrack.model.state import FIT_METRIC_KEYS, AppState


def test_result_arrays_are_allocated():
    state = AppState().with_results_initialized(4)
    assert set(state.fit_metrics) == set(FIT_METRIC_KEYS)
    for values in state.fit_metrics.values():
        assert values.shape == (4,) and np.isnan(values).all()
    assert state.fit_flags == [frozenset()] * 4
    assert state.fit_locked.dtype == bool and not state.fit_locked.any()


def test_session_round_trip_of_metrics_flags_and_locks(tmp_path):
    state = AppState(folder_path="x").with_results_initialized(3)
    state.fit_metrics["r_circle"][1] = 41.5
    state.fit_metrics["circularity"][2] = 0.93
    state.fit_flags[1] = frozenset({"non_circular", "clipped_completed"})
    state.fit_locked[2] = True
    path = tmp_path / "s.brt"
    save_session(str(path), state)
    data = load_session(str(path))
    assert data["fit_metrics"]["r_circle"][1] == 41.5
    assert np.isnan(data["fit_metrics"]["r_circle"][0])
    assert data["fit_flags"][1] == frozenset({"non_circular", "clipped_completed"})
    assert data["fit_locked"].tolist() == [False, False, True]


def test_restore_results_copies_contract_fields():
    from bubbletrack.controller.session_controller import restore_results

    src = AppState().with_results_initialized(2)
    src.fit_metrics["area_px"][0] = 123.0
    src.fit_flags[0] = frozenset({"fragmented"})
    src.fit_locked[1] = True
    dst = AppState().with_results_initialized(2)
    restore_results(dst, {"radius": np.array([5.0, 6.0]), "fit_metrics": src.fit_metrics,
                          "fit_flags": src.fit_flags, "fit_locked": src.fit_locked})
    assert dst.fit_metrics["area_px"][0] == 123.0
    assert dst.fit_flags[0] == frozenset({"fragmented"})
    assert dst.fit_locked[1]


def test_edge_fit_carries_metrics_and_store_fit_writes_them():
    from bubbletrack.controller import edge_support
    from bubbletrack.model.edge_pipeline import EdgeFrameFit

    state = AppState().with_results_initialized(3)
    fit = EdgeFrameFit(30.0, 10.0, 12.0, np.zeros((3, 2)), None, 0.8,
                       frozenset({"non_circular"}), "",
                       metrics={"r_circle": 31.0, "circularity": 0.9})
    edge_support.store_fit(state, 1, fit, FitSource.REFIT)
    assert state.fit_metrics["r_circle"][1] == 31.0
    assert state.fit_metrics["circularity"][1] == 0.9
    assert np.isnan(state.fit_metrics["area_px"][1])
    assert state.fit_flags[1] == frozenset({"non_circular"})


def test_locked_frames_are_always_protected_from_fit_all(gui, make_sequence):
    _, controller = gui
    controller.file_ctrl.on_folder_selected(str(make_sequence("A", 6)))
    controller._get_state().fit_locked[2] = True
    protected = controller.auto_ctrl._choose_protected_frames(0, 5)
    assert protected == frozenset({2})
