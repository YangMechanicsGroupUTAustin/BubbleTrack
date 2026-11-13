"""Drag a circle: the acceptance rule for a user-drawn circle and the guided
engine call (prior = the circle, no seed click, confidence floor 0.15)."""

from __future__ import annotations

import numpy as np
import pytest

from _golden_cases import SceneSpec, render_scene
from bubbletrack.model import edge_pipeline, guided_fit
from bubbletrack.model.edge_pipeline import EdgeFrameFit, EdgeJob, make_params
from bubbletrack.model.guided_fit import (
    CENTRE_TOLERANCE, CLICK_FALLBACK_RADIUS_PX, MIN_CONFIDENCE, RADIUS_TOLERANCE,
    UserCircle, clicked_circle, drawn_circle, fit_guided, is_consistent, judge,
    nearest_fitted_radius,
)

# --------------------------------------------------------------------------- #
#  Thresholds live in one place
# --------------------------------------------------------------------------- #


def test_thresholds_are_the_study_values():
    assert (CENTRE_TOLERANCE, RADIUS_TOLERANCE, MIN_CONFIDENCE) == (0.5, 0.5, 0.15)
    assert "85 -> 91 %" in guided_fit.__doc__  # the rule cites the study it comes from


# --------------------------------------------------------------------------- #
#  Consistency with the drag
# --------------------------------------------------------------------------- #

DRAG = drawn_circle(100.0, 200.0, 40.0)


@pytest.mark.parametrize("offset, ok", [(0.0, True), (19.9, True), (20.0, True), (20.1, False)])
def test_centre_must_lie_within_half_the_drawn_radius(offset, ok):
    assert is_consistent(DRAG, 100.0 + offset, 200.0, 40.0) is ok


@pytest.mark.parametrize("radius, ok", [(20.0, True), (19.9, False), (60.0, True), (60.1, False)])
def test_radius_must_be_within_half_the_drawn_radius(radius, ok):
    assert is_consistent(DRAG, 100.0, 200.0, radius) is ok


def test_a_click_without_size_checks_the_centre_against_the_measured_radius():
    click = clicked_circle(100.0, 200.0, None)
    assert not click.dragged and not click.sized
    assert click.radius == CLICK_FALLBACK_RADIUS_PX
    assert is_consistent(click, 100.0, 215.0, 40.0)       # 0.375 R off: inside the bubble
    assert not is_consistent(click, 100.0, 225.0, 40.0)   # 0.625 R off
    assert is_consistent(click, 100.0, 200.0, 400.0)      # no size given: any radius


def test_a_click_takes_the_neighbour_radius_and_is_checked_like_a_drag():
    click = clicked_circle(100.0, 200.0, 40.0)
    assert not click.dragged and click.sized and click.radius == 40.0
    assert is_consistent(click, 100.0, 200.0, 55.0)
    assert not is_consistent(click, 100.0, 200.0, 61.0)


def test_prior_is_the_circle():
    assert DRAG.prior == (100.0, 200.0, 40.0)


def test_drawn_circle_needs_a_positive_radius():
    with pytest.raises(ValueError):
        drawn_circle(1.0, 1.0, 0.0)


# --------------------------------------------------------------------------- #
#  Nearest fitted neighbour (prior radius for a click)
# --------------------------------------------------------------------------- #

def test_nearest_fitted_radius_skips_the_frame_itself_and_failed_frames():
    radius = np.array([10.0, -1.0, 99.0, np.nan, -1.0, 30.0])
    assert nearest_fitted_radius(radius, 2) == 10.0      # frames 0 and 5 are 2 / 3 away
    assert nearest_fitted_radius(radius, 4) == 30.0
    assert nearest_fitted_radius(radius, 3) == 99.0


def test_nearest_fitted_radius_prefers_the_earlier_frame_on_a_tie():
    assert nearest_fitted_radius(np.array([10.0, -1.0, 30.0]), 1) == 10.0


def test_nearest_fitted_radius_is_none_without_fits():
    assert nearest_fitted_radius(np.array([-1.0, -1.0]), 0) is None
    assert nearest_fitted_radius(None, 0) is None


# --------------------------------------------------------------------------- #
#  Verdict on an engine fit
# --------------------------------------------------------------------------- #

def _fit(radius=40.0, rc=100.0, cc=200.0, confidence=0.8, reason=""):
    return EdgeFrameFit(radius, rc, cc, np.zeros((0, 2)), None, confidence,
                        frozenset(), reason)


def test_consistent_fit_above_the_floor_is_accepted():
    verdict = judge(DRAG, _fit(confidence=0.16))
    assert verdict.accepted and verdict.reason == ""


def test_a_fit_far_from_the_drag_is_refused():
    verdict = judge(DRAG, _fit(rc=150.0))
    assert not verdict.accepted and "circle" in verdict.reason


def test_no_wall_found_is_refused_with_the_engine_reason():
    verdict = judge(DRAG, _fit(radius=-1.0, confidence=0.1, reason="low confidence (0.10)"))
    assert not verdict.accepted and "0.10" in verdict.reason


# --------------------------------------------------------------------------- #
#  The engine call
# --------------------------------------------------------------------------- #

class _FakeResult:
    def __init__(self, confidence, radius=40.0):
        self.radius, self.rc, self.cc = radius, 100.0, 200.0
        self.confidence = confidence
        self.accepted = confidence >= 0.3
        self.flags = frozenset({"circle_fallback"} if confidence < 0.3 else ())
        self.edge_xy = np.array([[60.0, 200.0], [140.0, 200.0], [100.0, 160.0]])
        self.mask = None


def _fake_engine(monkeypatch, confidence):
    calls = []

    def engine(raw, gridx, gridy, reference=None, params=None):
        calls.append(params)
        return _FakeResult(confidence)
    monkeypatch.setattr(edge_pipeline, "_engine", lambda: engine)
    return calls


JOB = EdgeJob(make_params(seed=(5.0, 5.0)), None, 500.0)


def test_engine_gets_the_circle_as_prior_and_no_seed(monkeypatch):
    calls = _fake_engine(monkeypatch, 0.9)
    fit, verdict = fit_guided(np.zeros((4, 4)), (1, 4), (1, 4), JOB, DRAG)
    assert calls[0].prior == DRAG.prior and calls[0].seed is None
    assert verdict.accepted and fit.radius == 40.0


def test_confidence_between_floor_and_gate_is_kept(monkeypatch):
    _fake_engine(monkeypatch, 0.2)
    fit, verdict = fit_guided(np.zeros((4, 4)), (1, 4), (1, 4), JOB, DRAG)
    assert verdict.accepted and fit.radius == 40.0 and fit.confidence == 0.2


def test_confidence_below_the_floor_is_refused(monkeypatch):
    _fake_engine(monkeypatch, 0.1)
    fit, verdict = fit_guided(np.zeros((4, 4)), (1, 4), (1, 4), JOB, DRAG)
    assert not verdict.accepted and fit.radius == -1.0


def test_fit_edge_default_gate_is_unchanged(monkeypatch):
    _fake_engine(monkeypatch, 0.2)
    assert edge_pipeline.fit_edge(np.zeros((4, 4)), (1, 4), (1, 4), JOB).radius == -1.0
    assert edge_pipeline.fit_edge(np.zeros((4, 4)), (1, 4), (1, 4), JOB,
                                  min_confidence=0.15).radius == 40.0


# --------------------------------------------------------------------------- #
#  Real engine on a synthetic frame
# --------------------------------------------------------------------------- #

REF = render_scene(SceneSpec(radius=0.0, center=(125.0, 200.0), seed=1))
FRAME = render_scene(SceneSpec(radius=40.0, center=(125.0, 200.0), seed=3))
REAL_JOB = EdgeJob(make_params(), REF, 500.0)


def test_rough_drag_measures_the_wall():
    fit, verdict = fit_guided(FRAME, (1, 250), (1, 400), REAL_JOB, drawn_circle(130.0, 195.0, 30.0))
    assert verdict.accepted
    assert fit.radius == pytest.approx(40.0, abs=1.0)
    assert (fit.rc, fit.cc) == pytest.approx((126.0, 201.0), abs=1.0)  # 1-based


def test_drag_on_an_empty_frame_is_refused():
    _, verdict = fit_guided(REF, (1, 250), (1, 400), REAL_JOB, drawn_circle(126.0, 201.0, 40.0))
    assert not verdict.accepted


def test_drag_beside_the_bubble_is_refused():
    _, verdict = fit_guided(FRAME, (1, 250), (1, 400), REAL_JOB, drawn_circle(60.0, 60.0, 30.0))
    assert not verdict.accepted


def test_user_circle_is_frozen():
    with pytest.raises(Exception):
        DRAG.rc = 0.0  # type: ignore[misc]
    assert isinstance(DRAG, UserCircle)
