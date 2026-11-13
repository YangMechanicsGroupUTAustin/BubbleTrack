"""GUIDED provenance (a drawn circle, wall measured by Auto) counts as a
user-confirmed result everywhere MANUAL does, behind one switch."""

from __future__ import annotations

import numpy as np
import pytest

from bubbletrack.controller.review_queue import build_review_items
from bubbletrack.controller.tracking import TrackingOutcome, apply_tracking
from bubbletrack.model import provenance
from bubbletrack.model.export_fit import FIT_SOURCE_NAMES
from bubbletrack.model.provenance import (
    USER_CONFIRMED_SOURCES, FitSource, is_user_confirmed, user_confirmed_mask,
)
from bubbletrack.model.sequence import FrameSolution, SequenceResult
from bubbletrack.model.sequence_pipeline import anchors
from bubbletrack.model.state import AppState
from bubbletrack.ui import chart_style


def test_guided_is_a_new_source_code():
    assert FitSource.GUIDED == 6
    assert FIT_SOURCE_NAMES[FitSource.GUIDED] == "GUIDED"


def test_manual_and_guided_are_user_confirmed():
    assert provenance.GUIDED_COUNTS_AS_CONFIRMED is True  # the default
    assert USER_CONFIRMED_SOURCES == frozenset({FitSource.MANUAL, FitSource.GUIDED})
    for source in FitSource:
        assert is_user_confirmed(source) is (source in USER_CONFIRMED_SOURCES)


def test_user_confirmed_mask():
    codes = np.array([0, 1, 2, 3, 4, 5, 6], dtype=np.int8)
    assert user_confirmed_mask(codes).tolist() == [False, False, True, False, False, False, True]
    assert user_confirmed_mask(None, 3).tolist() == [False, False, False]


def test_chart_draws_guided_as_the_existing_diamond():
    sources = np.array([FitSource.AUTO, FitSource.MANUAL, FitSource.GUIDED, FitSource.TRACKED])
    assert chart_style.marker_symbols(sources, None, 4).tolist() == ["o", "d", "d", "o"]
    assert set(chart_style.SYMBOL_SIZES) == {"o", "d"}  # no new marker shapes
    assert [name for _, name in chart_style.SHAPE_LEGEND] == ["computed", "manual / locked"]


def _state(n=5):
    state = AppState().with_results_initialized(n)
    state.radius[:] = [20.0, 21.0, 22.0, 23.0, 24.0]
    state.circle_fit_par[:] = [[100.0, 150.0]] * n
    state.fit_confidence[:] = 0.9
    state.fit_source[:] = FitSource.AUTO
    return state


def test_guided_frames_anchor_the_track():
    state = _state()
    state.fit_source[2] = FitSource.GUIDED
    assert anchors(state) == {2: (100.0, 150.0, 22.0)}


def test_guided_frames_with_low_confidence_are_not_reviewed():
    state = _state()
    state.fit_source[1] = FitSource.GUIDED
    state.fit_confidence[1] = 0.16
    state.fit_flags[1] = frozenset({"circle_fallback", "non_circular"})
    state.fit_source[3] = FitSource.AUTO
    state.fit_confidence[3] = 0.16
    assert [item.frame for item in build_review_items(state)] == [3]


class _Redetector:
    def fit_for(self, idx, solution):
        from bubbletrack.model.edge_pipeline import EdgeFrameFit
        return EdgeFrameFit(solution.radius, 100.0, 150.0, np.zeros((0, 2)), None, 0.8,
                            frozenset(), "")


def _outcome(radii):
    frames = tuple(FrameSolution(r, 100.0, 150.0, 0.8, "tracked", frozenset(), 0) for r in radii)
    return TrackingOutcome(SequenceResult(frames, (), ((0, len(radii) - 1),), (), 0), _Redetector())


def test_tracking_never_touches_guided_frames():
    state = _state()
    state.fit_source[2] = FitSource.GUIDED
    changed = apply_tracking(state, _outcome([30.0] * 5))
    assert changed == [0, 1, 3, 4]
    assert state.radius[2] == 22.0 and state.fit_source[2] == FitSource.GUIDED


def test_allowed_frames_limit_what_tracking_may_change():
    state = _state()
    before = state.radius.copy()
    changed = apply_tracking(state, _outcome([30.0] * 5), allowed_frames={1, 4})
    assert changed == [1, 4]
    assert state.radius[0] == before[0] and state.radius[2] == before[2]
    assert state.fit_source[0] == FitSource.AUTO


def test_allowed_frames_none_keeps_fit_all_behaviour():
    state = _state()
    assert apply_tracking(state, _outcome([30.0] * 5)) == [0, 1, 2, 3, 4]


@pytest.mark.parametrize("flag", [False, True])
def test_the_switch_is_one_constant(monkeypatch, flag):
    monkeypatch.setattr(provenance, "GUIDED_COUNTS_AS_CONFIRMED", flag)
    sources = provenance.confirmed_sources()
    assert (FitSource.GUIDED in sources) is flag and FitSource.MANUAL in sources
