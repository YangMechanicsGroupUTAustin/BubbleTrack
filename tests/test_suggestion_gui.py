"""Auto's rejected guess in the running app (offscreen): drawn dashed on both
panels, display-only until Accept (A) in Review stores it, undone by Ctrl+Z,
replaced by any new fit, kept by Fit All and by sessions."""

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
from bubbletrack.model.cases.schema import BY_ACCEPTED, LABEL_BUBBLE, LABEL_NO_BUBBLE
from bubbletrack.model.provenance import FitSource
from bubbletrack.model.sequence_pipeline import observations
from bubbletrack.model.state import update_state
from bubbletrack.model.suggestion import shown_suggestion
from bubbletrack.ui import chart_style as cs
from bubbletrack.ui.review_rows import tooltip_html
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
    window.left_panel.pretune_tab.edge_box._track_check.setChecked(False)
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


def _select(controller, frame: int):
    rc = controller.review
    rc.refresh()
    assert rc.panel.select_frame(frame, keep_row=False)
    return rc


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
        assert "Auto's guess (confidence 0.20)" in panel.overlay_tip()
        assert "press A in Review" in panel.overlay_tip()
    assert _status(window).startswith("Frame 4: Auto's guess (confidence 0.20)")
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


def test_rejected_refit_replaces_the_guess_and_says_so(app):
    window, controller = app
    old = _reject(controller, 3).suggestion
    _show(controller, 3)
    controller._on_refit_point(3)
    new = shown_suggestion(_state(controller), 3)
    assert new is not None and new is not old
    assert "Auto's guess is shown dashed" in _status(window)
    assert len(_dashed(window.original_panel)) == 1


def test_pretune_fit_current_frame_draws_the_guess(app):
    window, controller = app
    _show(controller, 3)
    controller.pretune_ctrl.on_pretune_fit()
    assert shown_suggestion(_state(controller), 3) is not None
    assert len(_dashed(window.original_panel)) == 1 and len(_dashed(window.binary_panel)) == 1


# -- review -------------------------------------------------------------------------------

def test_review_row_says_a_guess_is_available(app):
    _, controller = app
    _reject(controller, 3)
    rc = _select(controller, 3)
    segment = rc.panel.current_segment()
    assert segment.label == "No result" and segment.frames == (3,)
    tip = tooltip_html(segment, _state(controller).images)
    assert "Auto&#x27;s guess available" in tip and "Accept (A) stores Auto&#x27;s guess" in tip
    assert "Auto's guess available" in rc.panel._detail.text()


def test_accept_stores_the_guess_as_a_locked_auto_result(app):
    window, controller = app
    s = _reject(controller, 3).suggestion
    _show(controller, 3)
    rc = _select(controller, 3)
    rc.perform("accept")
    state = _state(controller)
    assert state.radius[3] == s.radius and tuple(state.circle_fit_par[3]) == (s.rc, s.cc)
    np.testing.assert_array_equal(state.circle_xy[3], s.edge_xy)
    assert state.fit_confidence[3] == DAMPED and state.fit_flags[3] == s.flags
    for key, value in s.metrics:
        assert state.fit_metrics[key][3] == value
    assert state.fit_source[3] == FitSource.AUTO and state.fit_locked[3]
    assert shown_suggestion(state, 3) is None
    # user-confirmed everywhere: diamond on the chart, an accepted hard-case label
    chart = window.radius_chart
    pos = int(np.flatnonzero(chart._frames == 4)[0])
    assert cs.marker_symbols(chart._sources, chart._locked, len(chart._frames))[pos] == "d"
    label = frame_label(app_result(state, 3))
    assert (label.status, label.by) == (LABEL_BUBBLE, BY_ACCEPTED)
    assert 3 not in rc.panel.frames()
    assert "Auto's guess accepted" in _status(window) and "Ctrl+Z" in _status(window)
    _show(controller, 3)
    assert not _dashed(window.original_panel)  # drawn as a result now


def test_accept_without_a_guess_still_means_no_bubble(app, damp):
    _, controller = app
    damp.clear()
    state = _state(controller)
    state.fit_confidence[3] = 0.1  # rejected with nothing to suggest
    rc = _select(controller, 3)
    rc.perform("accept")
    assert state.radius[3] == -1.0 and state.fit_locked[3]
    assert frame_label(app_result(state, 3)).status == LABEL_NO_BUBBLE


def test_accept_on_a_mixed_row_stores_guesses_and_locks_the_rest(app):
    _, controller = app
    state = _state(controller)
    _reject(controller, 3)
    state.fit_confidence[[2, 4]] = 0.1  # no result, no guess
    rc = _select(controller, 3)
    segment = rc.panel.current_segment()
    assert segment.frames == (2, 3, 4)
    assert "Auto&#x27;s guess on 1 frame; 2 without one are locked as no bubble" in \
        tooltip_html(segment, state.images)
    rc.perform("accept")
    assert state.radius[3] > 0 and state.fit_locked[[2, 3, 4]].all()
    assert (state.radius[[2, 4]] == -1.0).all()


def test_ctrl_z_undoes_the_promotion(app):
    window, controller = app
    s = _reject(controller, 3).suggestion
    _show(controller, 3)
    _select(controller, 3).perform("accept")
    controller._on_undo()
    state = _state(controller)
    assert state.radius[3] == -1.0 and not state.fit_locked[3]
    assert state.fit_source[3] == FitSource.NONE and shown_suggestion(state, 3) is s
    assert "Undid the accept of Auto's guess on frame 4" in _status(window)
    _show(controller, 3)
    assert len(_dashed(window.original_panel)) == 1
    assert 3 in controller.review.panel.frames()


# -- any new fit replaces or clears the guess --------------------------------------------

@pytest.mark.parametrize("action", ["refit", "delete", "no_bubble", "manual_clear", "clear_all",
                                    "circle", "manual_points"])
def test_new_results_clear_the_guess(app, damp, action):
    window, controller = app
    _reject(controller, 3)
    _show(controller, 3)
    damp.clear()  # from now on Auto accepts the frame
    if action == "refit":
        controller._on_refit_point(3)
    elif action == "delete":
        controller._on_delete_point(3)
    elif action == "no_bubble":
        controller._on_mark_no_bubble(3, 3)
    elif action == "manual_clear":
        controller.manual_ctrl.on_manual_clear()
    elif action == "clear_all":
        controller.auto_ctrl.on_auto_clear()
    elif action == "circle":
        window.left_panel.tab_bar.select("Manual")
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
