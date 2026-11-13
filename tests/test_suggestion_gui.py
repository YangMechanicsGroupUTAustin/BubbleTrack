"""Auto's rejected guess in the running app (offscreen): drawn dashed on both
panels, display-only, replaced by any new fit, kept by Fit All and by sessions.
Accepting it in Review: ``test_suggestion_review.py``."""

from __future__ import annotations

import dataclasses
import time

import numpy as np
import pytest
from PyQt6.QtCore import Qt

from bubbletrack.controller import edge_support, worker as worker_module
from bubbletrack.controller.suggestion_view import GUESS_COLOUR, GUESS_GAP_COLOUR
from bubbletrack.model import edge_pipeline
from bubbletrack.model.cases.labels import app_result, frame_label
from bubbletrack.model.provenance import FitSource
from bubbletrack.model.sequence_pipeline import observations
from bubbletrack.model.state import update_state
from bubbletrack.model.suggestion import shown_suggestion
from conftest import write_sequence

RADII = [0.0, 0.0, 30.0, 40.0, 50.0, 45.0, 35.0]  # frames 1-2: bubble-free references
DAMPED = 0.2  # below the engine's 0.3 gate


@pytest.fixture
def damp(monkeypatch):
    """The real engine, with the confidence of chosen bubbles lowered below the gate."""
    from bubbletrack.model.edge import detect_edge

    radii: set[float] = {40.0}

    def engine(*args, **kwargs):
        res = detect_edge(*args, **kwargs)
        if any(np.isfinite(res.radius) and abs(res.radius - r) < 3.0 for r in radii):
            return dataclasses.replace(res, confidence=DAMPED)
        return res

    monkeypatch.setattr(edge_pipeline, "_engine", lambda: engine)
    return radii


@pytest.fixture
def app(gui, tmp_path, damp):
    window, controller = gui
    controller.file_ctrl.on_folder_selected(str(write_sequence(tmp_path / "seq", RADII)))
    controller._state = update_state(controller._get_state(), detection_mode="edge",
                                     reference_frame=0)
    window.left_panel.measure_panel.edge_box._track_check.setChecked(False)
    return window, controller


def _state(controller):
    return controller._get_state()


def _reject(controller, idx: int):
    """Fit frame *idx* with the damped engine and store it (as Fit All would)."""
    state = _state(controller)
    fit = edge_support.fit_frame(state, idx, 500.0)
    edge_support.store_fit(state, idx, fit, FitSource.AUTO)
    return fit


def _fit(controller, idx: int):
    state = _state(controller)
    edge_support.store_fit(state, idx, edge_support.fit_frame(state, idx, 500.0), FitSource.AUTO)


def _show(controller, idx: int):
    controller.file_ctrl.on_frame_changed(idx)


def _dashed(panel) -> list:
    return [item for item in panel._overlay_items
            if hasattr(item, "pen") and item.pen().style() == Qt.PenStyle.DashLine]


def _status(window) -> str:
    return window.header._status_label.text()


# -- display ------------------------------------------------------------------------------

def test_rejected_frame_shows_the_guess_dashed_on_both_panels(app):
    window, controller = app
    fit = _reject(controller, 3)
    assert fit.radius == -1.0 and fit.suggestion is not None
    _show(controller, 3)
    for panel in (window.original_panel, window.binary_panel):
        dashed = _dashed(panel)
        assert len(dashed) == 1 and dashed[0].pen().color().name() == GUESS_COLOUR
        # a dark line under the dashes keeps the outline visible on the light Auto view
        under = panel._overlay_items[panel._overlay_items.index(dashed[0]) - 1]
        assert under.pen().style() == Qt.PenStyle.SolidLine
        assert under.pen().color().name() == GUESS_GAP_COLOUR
        assert "Auto's guess (not accepted, confidence 0.20) — see Review" \
            in panel.overlay_tip()
        assert "press A" not in panel.overlay_tip()  # the Review card may advise Fix
        assert "Not a bubble? Press N in Review" in panel.overlay_tip()
        assert "Auto's guess (not accepted)" in panel.legend_texts()  # the corner key names it
        assert "outline (measured)" not in panel.legend_texts()
    assert _status(window).startswith("Frame 4: Auto's guess (not accepted, confidence 0.20)")
    # the Auto view draws it in ROI coordinates: same place as the image panel
    assert window.binary_panel._overlay_items[0].path().boundingRect().center().x() == \
        pytest.approx(window.original_panel._overlay_items[-1].path().boundingRect().center().x()
                      - (_state(controller).gridy[0] - 1), abs=0.5)


def test_other_frames_show_no_guess_and_clear_the_tip(app):
    window, controller = app
    _reject(controller, 3)
    _fit(controller, 4)
    _show(controller, 3)
    _show(controller, 4)
    assert not _dashed(window.original_panel) and not _dashed(window.binary_panel)
    assert window.original_panel.overlay_tip() == "" == window.binary_panel.overlay_tip()


def test_the_guess_is_display_only(app):
    window, controller = app
    _reject(controller, 3)
    state = _state(controller)
    controller._refresh_chart()
    assert state.radius[3] == -1.0 and state.circle_xy[3] is None
    assert np.isnan(state.circle_fit_par[3]).all() and state.fit_source[3] == FitSource.NONE
    assert 4.0 not in window.radius_chart._frames  # not in R(t)
    assert np.isnan(observations(state)[3].radius)  # not seen by tracking
    assert not frame_label(app_result(state, 3)).is_labelled  # not a label


def test_pretune_fit_current_frame_draws_the_guess(app):
    window, controller = app
    _show(controller, 3)
    controller.pretune_ctrl.on_pretune_fit()
    assert shown_suggestion(_state(controller), 3) is not None
    assert len(_dashed(window.original_panel)) == 1 and len(_dashed(window.binary_panel)) == 1


# -- any new fit replaces or clears the guess --------------------------------------------

@pytest.mark.parametrize("action", ["no_bubble", "clear_all", "circle", "manual_points"])
def test_new_results_clear_the_guess(app, damp, action):
    window, controller = app
    _reject(controller, 3)
    _show(controller, 3)
    damp.clear()  # from now on Auto accepts the frame
    if action == "no_bubble":
        controller._on_mark_no_bubble(3, 3)
    elif action == "clear_all":
        controller.auto_ctrl.on_auto_clear()
    elif action == "circle":
        assert controller.review.start_fix(3)  # Review's Fix: drag a circle
        window.original_panel.circle_drawn.emit(203.5, 128.5, 32.0, True)
    else:
        manual = controller.manual_ctrl
        manual.on_manual_select()
        for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            manual.on_point_clicked(200.0 + 40 * np.cos(angle), 125.0 + 40 * np.sin(angle))
        manual.on_manual_done()
    state = _state(controller)
    assert state.fit_suggestion[3] is None, action
    assert not _dashed(window.original_panel)


def test_fit_all_keeps_the_guess_of_rejected_frames(qapp, app, monkeypatch):
    window, controller = app
    monkeypatch.setattr(worker_module.BatchWorker, "inline", True)
    controller.auto_ctrl.on_auto_fit()
    deadline = time.monotonic() + 120
    while controller.auto_ctrl.is_busy() and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    state = _state(controller)
    assert state.radius[3] == -1.0 and shown_suggestion(state, 3) is not None
    assert shown_suggestion(state, 3).radius == pytest.approx(40.0, abs=1.5)
    assert all(state.fit_suggestion[i] is None for i in (2, 4, 5, 6))
    assert (state.radius[[2, 4, 5, 6]] > 0).all()


def test_a_reopened_session_shows_the_guess_again(app, tmp_path):
    window, controller = app
    s = _reject(controller, 3).suggestion
    _fit(controller, 4)  # sessions are autosaved/saved with at least one result
    path = str(tmp_path / "s.brt")
    assert controller.save_session_file(path)
    controller.auto_ctrl.on_auto_clear()
    assert controller.load_session_file(path)
    _show(controller, 3)
    restored = shown_suggestion(_state(controller), 3)
    assert restored is not None and restored.radius == s.radius
    assert len(_dashed(window.original_panel)) == 1
