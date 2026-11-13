""""Bubble may be non-round" outside the engine (owner decisions 2026-11-13): the setting
and its persistence, sessions, the pipeline's acceptance, Review, the legend, the
sequence review, ML assist and the exports' radius definition (see test_export_v31_*).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from bubbletrack.controller.display_mixin import EQUAL_AREA_CIRCLE, FITTED_CIRCLE, circle_legend
from bubbletrack.controller.review_advice import FIX, NONE, advise
from bubbletrack.controller.review_queue import GUESS, build_review_items
from bubbletrack.controller.review_segments import build_segments, category_label, category_of
from bubbletrack.model.config import load_config, save_config, validate_config
from bubbletrack.model.edge import EdgeResult
from bubbletrack.model.edge_pipeline import (
    UNMEASURABLE_REASON, EdgeJob, fit_from_result, make_params, params_from_state,
)
from bubbletrack.model.provenance import AREA_RADIUS_FLAG, FitSource
from bubbletrack.model.sequence import DEFAULT_SEQUENCE_CONFIG, FrameSolution
from bubbletrack.model.sequence.review import build_review
from bubbletrack.model.session import load_session, save_session, session_parameters
from bubbletrack.model.state import AppState
from bubbletrack.model.suggestion import make_suggestion, set_suggestion
from bubbletrack.model.teach import MeasureConfig, ml_region, select_result
from bubbletrack.ui.review_text import problem_sentence, reason_text

NAN = float("nan")


# ------------------------------------------------------------------ the setting

def test_off_by_default_and_passed_to_the_engine():
    assert AppState().bubble_non_round is False
    assert params_from_state(AppState()).non_round is False
    on = AppState(bubble_non_round=True)
    assert params_from_state(on).non_round is True
    assert params_from_state(on, prior=(50.0, 60.0, 20.0)).non_round is True
    assert make_params().non_round is False


def test_persisted_like_the_other_detection_parameters(tmp_path, monkeypatch):
    monkeypatch.setattr("bubbletrack.model.config.CONFIG_DIR", tmp_path)
    monkeypatch.setattr("bubbletrack.model.config.CONFIG_FILE", tmp_path / "config.json")
    save_config(AppState(bubble_non_round=True))
    assert load_config()["bubble_non_round"] is True
    assert validate_config({"bubble_non_round": "yes"}) == {}
    assert validate_config({"bubble_non_round": 0}) == {"bubble_non_round": False}


@pytest.mark.parametrize("on", [False, True])
def test_sessions_round_trip_the_setting(tmp_path, on):
    state = AppState(detection_mode="edge", bubble_non_round=on).with_results_initialized(2)
    assert session_parameters(state)["bubble_non_round"] is on
    path = str(tmp_path / "s.brt")
    save_session(path, state)
    assert load_session(path)["bubble_non_round"] is on


def _legacy_session(tmp_path, mode: str) -> str:
    """A session saved before the setting existed (version 3.1, no key)."""
    state = AppState(detection_mode=mode).with_results_initialized(4)
    state.radius[:] = [30.0, 31.0, 32.0, -1.0]
    state.fit_source[:] = [FitSource.AUTO, FitSource.MANUAL, FitSource.AUTO, FitSource.NONE]
    state.fit_flags[0] = frozenset({"clipped", "clipped_completed"})
    state.fit_flags[2] = frozenset({"circle_fallback"})
    path = tmp_path / f"old_{mode}.brt"
    save_session(str(path), state)
    data = json.loads(path.read_text(encoding="utf-8"))
    del data["bubble_non_round"]
    data["version"] = "3.1"
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


def test_a_session_from_before_the_setting_keeps_its_area_definition(tmp_path):
    data = load_session(_legacy_session(tmp_path, "edge"))
    assert data["bubble_non_round"] is True                 # Auto measured the area then
    flags = data["fit_flags"]
    assert AREA_RADIUS_FLAG in flags[0]                     # Auto's area result
    assert AREA_RADIUS_FLAG not in flags[1]                 # a manual circle fit
    assert AREA_RADIUS_FLAG not in flags[2]                 # the circle fallback
    assert AREA_RADIUS_FLAG not in flags[3]                 # no result
    assert "bubble_non_round" not in load_session(_legacy_session(tmp_path, "intensity"))


# ------------------------------------------------------------------ acceptance

def _result(flags, radius=40.0, confidence=0.9):
    return EdgeResult(radius=radius, rc=50.0, cc=60.0, radius_sigma=0.1,
                      edge_xy=np.array([[10.0, 60.0], [50.0, 100.0], [90.0, 60.0]]),
                      confidence=confidence, flags=frozenset(flags), r_area=radius, outer_radius=NAN,
                      shoulder_width=NAN, arc_coverage=0.6, inlier_fraction=1.0,
                      mask=np.zeros((4, 4), bool))


@pytest.mark.parametrize("min_confidence", [None, 0.15])
def test_an_unmeasurable_frame_has_no_result_but_keeps_its_guess(min_confidence):
    res = _result({"clipped", "clipped_unmeasurable", AREA_RADIUS_FLAG})
    assert not res.accepted
    fit = fit_from_result(res, EdgeJob(make_params(non_round=True), None, 400.0),
                          min_confidence=min_confidence)
    assert fit.radius == -1.0 and fit.rejected_reason == UNMEASURABLE_REASON
    assert fit.suggestion is not None and fit.suggestion.radius == 40.0
    assert "clipped_unmeasurable" in fit.suggestion.flags


# ------------------------------------------------------------------ Review

def _state(n=13) -> AppState:
    state = AppState(detection_mode="edge").with_results_initialized(n)
    state.radius[:] = 30.0
    state.circle_fit_par[:] = [[100.0, 100.0]] * n
    state.fit_source[:] = FitSource.AUTO
    state.fit_confidence[:] = 0.9
    return state


@pytest.fixture
def flagged():
    """Frames 0: circle off its area; 3: the same measured by area (information); 6: clipped,
    visible wall not round (circle); 9: clipped and completed (information); 12: clipped,
    unmeasurable (area mode, no result, a guess)."""
    state = _state()
    state.fit_flags[0] = frozenset({"non_circular"})
    state.fit_flags[3] = frozenset({"non_circular", AREA_RADIUS_FLAG})
    state.fit_flags[6] = frozenset({"clipped", "clipped_not_round"})
    state.fit_flags[9] = frozenset({"clipped", "clipped_completed", AREA_RADIUS_FLAG})
    state.radius[12] = -1.0
    state.fit_source[12] = FitSource.NONE
    state.fit_flags[12] = frozenset({"clipped", "clipped_unmeasurable", AREA_RADIUS_FLAG})
    set_suggestion(state, 12, make_suggestion(30.0, 100.0, 100.0, np.zeros((4, 2)), 0.9,
                                              state.fit_flags[12]))
    return state, {item.frame: item for item in build_review_items(state)}


def test_non_circular_is_a_check_reason_only_for_the_circle(flagged):
    _, items = flagged
    assert set(items) == {0, 6, 12}
    assert items[0].reasons == ("non_circular",) and category_of(items[0]) == "shape"


def test_clipped_frames_have_a_row_of_their_own(flagged):
    _, items = flagged
    rows = {s.start: s for s in build_segments(items.values())}
    assert rows[6].category == "clipped" and rows[6].tier == "medium"     # worth a check
    assert rows[12].category == "clipped" and rows[12].tier == "high"     # no result
    assert {"rejected", GUESS, "clipped_unmeasurable"} <= set(items[12].reasons)
    assert category_label("clipped") == "Cut by the image edge"
    assert problem_sentence("clipped") == "Cut by the image edge here."
    assert reason_text("clipped_unmeasurable", NAN) == "cut by the image edge: its area can't be measured"
    assert reason_text("clipped_not_round", NAN) == "cut by the image edge and the visible wall is not round"


def test_a_well_fitting_clipped_frame_is_information_only():
    state = _state(1)
    state.fit_flags[0] = frozenset({"clipped", "clipped_completed", AREA_RADIUS_FLAG})
    assert build_review_items(state) == []


def test_advice_in_plain_words(flagged):
    state, items = flagged
    # The circle may be off the wall: no button is recommended, the text says
    # when to accept (Accept stays available) and when to change the setting.
    oval = advise(state, 0, items[0])
    assert oval.action == NONE and oval.can_accept
    assert oval.text.startswith("Outline not round here.")
    assert 'tick "Bubble may be non-round"' in oval.text and "Fit again" in oval.text
    cut = advise(state, 6, items[6])
    assert cut.action == NONE and cut.can_accept
    assert cut.text.startswith("Cut by the image edge here.")
    assert "circle may be off" in cut.text
    unmeasurable = advise(state, 12, items[12])
    assert unmeasurable.action == FIX and unmeasurable.guess and unmeasurable.can_accept
    assert unmeasurable.text.startswith("Cut by the image edge here. Its area can't be measured")


def test_the_sequence_review_follows_the_same_rule():
    def frame(flags):
        return FrameSolution(30.0, 100.0, 100.0, 0.9, "raw", frozenset(flags), 0)

    frames = (frame({"non_circular"}), frame({"non_circular", AREA_RADIUS_FLAG}),
              frame({"clipped_not_round"}), frame({"clipped_completed"}))
    listed = dict(build_review(frames, (), DEFAULT_SEQUENCE_CONFIG))
    assert listed == {0: ("non_circular",), 2: ("clipped_not_round",)}


def test_the_drawn_circle_is_named_by_its_definition():
    state = _state(2)
    state.fit_flags[1] = frozenset({AREA_RADIUS_FLAG})
    assert circle_legend(state, 0) == FITTED_CIRCLE
    assert circle_legend(state, 1) == EQUAL_AREA_CIRCLE


# ------------------------------------------------------------------ ML assist

def _soft_ellipse(rc, cc, a, b, shape=(120, 160)):
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    d = np.hypot((yy - rc) / b, (xx - cc) / a)
    return (0.5 * (1.0 + np.tanh((1.0 - d) * min(a, b)))).astype(np.float32)


def test_ml_outline_follows_the_definition():
    p = _soft_ellipse(60, 80, 45, 22)
    cfg = MeasureConfig(refine=False)
    circle, area = ml_region(p, config=cfg), ml_region(p, config=cfg, non_round=True)
    assert area.radius == pytest.approx(np.sqrt(45 * 22), rel=0.02) == area.r_area
    assert circle.r_area == pytest.approx(area.r_area)
    assert abs(circle.radius - area.radius) > 0.05 * area.radius      # the circle, not the area
    assert AREA_RADIUS_FLAG in select_result(area, None, config=cfg, non_round=True).flags
    assert AREA_RADIUS_FLAG not in select_result(circle, None, config=cfg).flags


def test_ml_circle_leaves_out_the_roi_border():
    """A disk cut by the ROI border: the circle through its wall is the disk's."""
    yy, xx = np.mgrid[0:120, 0:160]
    p = (0.5 * (1.0 + np.tanh(30.0 - np.hypot(yy - 10.0, xx - 80.0)))).astype(np.float32)
    reg = ml_region(p, config=MeasureConfig(refine=False))
    assert reg.clipped and reg.radius == pytest.approx(30.0, abs=0.7)
    assert (reg.rc, reg.cc) == pytest.approx((11.0, 81.0), abs=0.7)
