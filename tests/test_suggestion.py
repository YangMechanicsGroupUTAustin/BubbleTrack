"""Auto's rejected guess as a suggestion (model layer): what ``fit_edge`` keeps
for a rejected frame, where it lives in the state, undo and sessions."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pytest

from bubbletrack.controller import edge_support
from bubbletrack.model import edge_pipeline
from bubbletrack.model.edge_pipeline import EdgeJob, fit_edge
from bubbletrack.model.provenance import FitSource
from bubbletrack.model.result_history import capture, record_edit, restore, same_result
from bubbletrack.model.session import load_session, save_session
from bubbletrack.model.state import AppState, clear_frame_result
from bubbletrack.model.suggestion import (
    Suggestion, from_json, make_suggestion, same_suggestion, shown_suggestion, to_json,
)

OUTLINE = np.array([[85.0, 200.0], [125.0, 240.0], [165.0, 200.0], [125.0, 160.0]])


@dataclass
class FakeResult:
    radius: float = 40.0
    rc: float = 125.0
    cc: float = 200.0
    confidence: float = 0.9
    flags: frozenset = frozenset({"circle_fallback"})
    edge_xy: np.ndarray = field(default_factory=lambda: OUTLINE.copy())
    mask: np.ndarray = field(default_factory=lambda: np.zeros((250, 400), bool))
    r_circle: float = 41.0
    circularity: float = 0.93

    @property
    def accepted(self) -> bool:
        return bool(np.isfinite(self.radius) and self.radius > 0 and self.confidence >= 0.3)


def _fit(monkeypatch, max_radius=500.0, min_confidence=None, **result):
    fake = FakeResult(**result)
    monkeypatch.setattr(edge_pipeline, "_engine", lambda: lambda *a, **k: fake)
    job = EdgeJob(params=None, reference=None, max_radius=max_radius)
    return fit_edge(np.zeros((250, 400)), (1, 250), (1, 400), job, min_confidence=min_confidence)


def _suggestion(radius=31.0, confidence=0.24) -> Suggestion:
    return make_suggestion(radius, 120.0, 190.0, OUTLINE, confidence,
                           frozenset({"circle_fallback"}), {"r_circle": radius + 0.5})


# -- what fit_edge keeps ---------------------------------------------------------------------

class TestFitEdge:
    def test_low_confidence_keeps_the_candidate_as_a_suggestion(self, monkeypatch):
        fit = _fit(monkeypatch, confidence=0.24)
        # the result itself is rejected exactly as before
        assert fit.radius == -1.0 and np.isnan(fit.rc) and np.isnan(fit.cc)
        assert fit.edge_xy.shape == (0, 2) and fit.rejected_reason == "low confidence (0.24)"
        s = fit.suggestion
        assert (s.radius, s.rc, s.cc, s.confidence) == (40.0, 125.0, 200.0, 0.24)
        np.testing.assert_array_equal(s.edge_xy, OUTLINE)
        assert s.flags == frozenset({"circle_fallback"})
        assert s.metrics_dict == {"r_circle": 41.0, "circularity": 0.93}

    def test_suggestion_owns_its_outline(self, monkeypatch):
        fake_xy = OUTLINE.copy()
        fit = _fit(monkeypatch, confidence=0.1, edge_xy=fake_xy)
        fake_xy[0, 0] = -5.0
        assert fit.suggestion.edge_xy[0, 0] == 85.0

    def test_accepted_fit_is_unchanged_and_has_no_suggestion(self, monkeypatch):
        fit = _fit(monkeypatch, confidence=0.9)
        assert (fit.radius, fit.rc, fit.cc, fit.rejected_reason) == (40.0, 125.0, 200.0, "")
        np.testing.assert_array_equal(fit.edge_xy, OUTLINE)
        assert fit.suggestion is None

    @pytest.mark.parametrize("result", [
        {"radius": float("nan"), "confidence": 0.1},   # no bubble found
        {"radius": 0.0, "confidence": 0.1},
        {"rc": float("nan"), "confidence": 0.1},       # no centre
    ])
    def test_no_candidate_no_suggestion(self, monkeypatch, result):
        assert _fit(monkeypatch, **result).suggestion is None

    def test_implausible_radius_is_not_suggested(self, monkeypatch):
        fit = _fit(monkeypatch, max_radius=30.0, confidence=0.1)
        assert fit.radius == -1.0 and fit.suggestion is None

    def test_guided_floor_rejection_is_suggested(self, monkeypatch):
        fit = _fit(monkeypatch, min_confidence=0.15, confidence=0.1)
        assert fit.radius == -1.0 and fit.suggestion is not None
        assert _fit(monkeypatch, min_confidence=0.15, confidence=0.2).suggestion is None


# -- the state ---------------------------------------------------------------------------

def _state(n=4) -> AppState:
    return AppState().with_results_initialized(n)


def test_results_start_without_suggestions():
    assert _state(3).fit_suggestion == [None, None, None]


def test_shown_only_while_the_frame_has_no_result_and_is_unlocked():
    state = _state()
    state.fit_suggestion[1] = _suggestion()
    assert shown_suggestion(state, 1) is state.fit_suggestion[1]
    assert shown_suggestion(state, 0) is None and shown_suggestion(state, 9) is None
    state.radius[1] = 30.0
    assert shown_suggestion(state, 1) is None
    state.radius[1] = -1.0
    state.fit_locked[1] = True
    assert shown_suggestion(state, 1) is None
    assert shown_suggestion(AppState(), 0) is None  # nothing loaded


def test_clear_frame_result_clears_the_suggestion():
    state = _state()
    state.fit_suggestion[2] = _suggestion()
    clear_frame_result(state, 2)
    assert state.fit_suggestion[2] is None


class TestStoreFit:
    def test_rejected_fit_stores_its_suggestion(self, monkeypatch):
        state = _state()
        fit = _fit(monkeypatch, confidence=0.24)
        edge_support.store_fit(state, 1, fit, FitSource.AUTO)
        assert state.radius[1] == -1.0 and state.fit_source[1] == FitSource.NONE
        assert state.fit_suggestion[1] is fit.suggestion

    def test_accepted_fit_clears_an_older_suggestion(self, monkeypatch):
        state = _state()
        state.fit_suggestion[1] = _suggestion()
        edge_support.store_fit(state, 1, _fit(monkeypatch, confidence=0.9), FitSource.REFIT)
        assert state.radius[1] == 40.0 and state.fit_suggestion[1] is None

    def test_meta_without_a_suggestion_clears_it(self):
        state = _state()
        state.fit_suggestion[1] = _suggestion()
        edge_support.store_meta(state, 1, {})
        assert state.fit_suggestion[1] is None

    def test_works_without_the_array(self, monkeypatch):
        from bubbletrack.model.state import update_state

        state = update_state(_state(), fit_suggestion=None)
        edge_support.store_fit(state, 1, _fit(monkeypatch, confidence=0.2), FitSource.AUTO)


class TestPromote:
    def test_accept_stores_the_guess_as_a_locked_auto_result(self):
        state = _state()
        s = _suggestion()
        state.fit_suggestion[2] = s
        state.fit_confidence[2] = s.confidence
        assert edge_support.accept_suggestion(state, 2)
        assert state.radius[2] == 31.0 and tuple(state.circle_fit_par[2]) == (120.0, 190.0)
        np.testing.assert_array_equal(state.circle_xy[2], OUTLINE)
        assert state.fit_confidence[2] == 0.24 and state.fit_flags[2] == s.flags
        assert state.fit_metrics["r_circle"][2] == 31.5
        assert np.isnan(state.fit_metrics["circularity"][2])
        assert state.fit_source[2] == FitSource.AUTO and state.fit_locked[2]
        assert state.fit_suggestion[2] is None

    def test_accept_without_a_shown_suggestion_does_nothing(self):
        state = _state()
        assert not edge_support.accept_suggestion(state, 2)
        state.fit_suggestion[2] = _suggestion()
        state.fit_locked[2] = True  # "no bubble here" wins
        assert not edge_support.accept_suggestion(state, 2)
        assert state.radius[2] == -1.0


# -- review queue ------------------------------------------------------------------------

class TestReviewRows:
    @staticmethod
    def _queue():
        from bubbletrack.controller.review_queue import build_review_items
        from bubbletrack.controller.review_segments import build_segments

        state = _state(6)
        state.fit_confidence[[1, 2, 3]] = [0.1, 0.24, 0.12]  # three rejected frames ...
        state.fit_suggestion[2] = _suggestion()              # ... one with a guess
        return state, build_segments(build_review_items(state))

    def test_guess_describes_a_no_result_frame(self):
        from bubbletrack.controller.review_queue import GUESS

        state, segments = self._queue()
        assert [(s.category, s.frames) for s in segments] == [("no_result", (1, 2, 3))]
        reasons = {item.frame: item.reasons for item in segments[0].items}
        assert reasons[2][:2] == ("rejected", GUESS) and GUESS not in reasons[1]

    def test_guess_never_lists_a_frame_on_its_own(self):
        from bubbletrack.controller.review_queue import build_review_items

        state = _state(3)
        state.fit_suggestion[1] = _suggestion()  # unscored: not an engine rejection
        assert build_review_items(state) == []
        state.fit_confidence[1] = 0.2
        state.fit_locked[1] = True  # accepted "no bubble here"
        assert build_review_items(state) == []

    def test_texts_say_what_accept_does(self):
        from bubbletrack.controller.review_segments import ReviewSegment
        from bubbletrack.ui.review_text import detail_text, tooltip_lines

        _, segments = self._queue()
        row = segments[0]
        lines = tooltip_lines(row, ())
        assert ("Auto's guess on 1 of 3 frames (dashed): Accept (A) stores the one on screen, "
                "Accept all (Shift+A) every one") in lines
        assert "Not a bubble? No bubble (N); Shift+N for the whole row" in lines
        single = ReviewSegment("no_result", (row.items[1],))
        assert "Auto's guess (dashed outline): Accept (A) stores it as the result" \
            in tooltip_lines(single, ())
        guess, bare = row.items[1], row.items[0]
        assert detail_text(guess, 2, (), segment=row).endswith("A accepts the guess shown")
        assert detail_text(bare, 1, (), segment=row) == detail_text(bare, 1, ())  # no guess
        assert "A " not in detail_text(guess, 2, ())  # no row selected: no hint
        plain = ReviewSegment("no_result", (bare,))
        assert not any("Accept" in line for line in tooltip_lines(plain, ()))
        assert detail_text(bare, 1, (), segment=plain) == detail_text(bare, 1, ())


# -- undo history ------------------------------------------------------------------------

def test_history_captures_and_restores_the_suggestion():
    state = _state()
    s = _suggestion()
    state.fit_suggestion[2] = s
    before = capture(state, 2)
    state.fit_suggestion[2] = None
    assert not same_result(capture(state, 2), before)
    assert record_edit(state, "x", {2: before}) is not None
    restore(state, 2, before)
    assert state.fit_suggestion[2] is s and same_result(capture(state, 2), before)


def test_same_suggestion_compares_values():
    a, b = _suggestion(), _suggestion()
    assert same_suggestion(a, b) and same_suggestion(None, None)
    assert not same_suggestion(a, None) and not same_suggestion(a, _suggestion(radius=32.0))
    nan_a = make_suggestion(30.0, 1.0, 2.0, OUTLINE, float("nan"))
    nan_b = make_suggestion(30.0, 1.0, 2.0, OUTLINE, float("nan"))
    assert same_suggestion(nan_a, nan_b)


def test_make_suggestion_rejects_missing_circles():
    assert make_suggestion(float("nan"), 1.0, 2.0, OUTLINE, 0.2) is None
    assert make_suggestion(-1.0, 1.0, 2.0, OUTLINE, 0.2) is None
    assert make_suggestion(30.0, 1.0, float("inf"), OUTLINE, 0.2) is None
    assert make_suggestion(30.0, 1.0, 2.0, None, 0.2).edge_xy.shape == (0, 2)


# -- sessions ----------------------------------------------------------------------------

def test_json_round_trip_with_nan_values():
    s = make_suggestion(30.0, 1.0, 2.0, np.array([[1.0, np.nan], [3.0, 4.0]]), 0.2,
                        frozenset({"b", "a"}), {"r_circle": float("nan"), "area_px": 9.0})
    data = to_json([None, s])
    json.dumps(data, allow_nan=False)  # strict JSON
    back = from_json(json.loads(json.dumps(data)))
    assert back[0] is None and same_suggestion(back[1], s)


def test_malformed_entries_are_dropped():
    good = to_json([_suggestion()])[0]
    back = from_json([good, {"radius": 3.0}, "nonsense", {"radius": None, "center": [1, 2]}])
    assert same_suggestion(back[0], _suggestion()) and back[1:] == [None, None, None]
    assert from_json({"not": "a list"}) is None


def test_session_round_trip(tmp_path):
    state = _state()
    state.fit_suggestion[1] = _suggestion()
    path = tmp_path / "s.brt"
    save_session(str(path), state)
    data = load_session(str(path))
    assert data["fit_suggestion"][0] is None
    assert same_suggestion(data["fit_suggestion"][1], state.fit_suggestion[1])


def test_old_session_without_suggestions_loads(tmp_path):
    state = _state()
    path = tmp_path / "old.brt"
    save_session(str(path), state)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw.pop("fit_suggestion")
    path.write_text(json.dumps(raw), encoding="utf-8")
    assert "fit_suggestion" not in load_session(str(path))


def test_session_restore_puts_suggestions_back():
    from bubbletrack.controller.session_controller import restore_results, split_session

    saved = _state()
    saved.fit_suggestion[3] = _suggestion()
    data = {"radius": saved.radius, "fit_suggestion": list(saved.fit_suggestion)}
    _, params, results = split_session(data)
    assert "fit_suggestion" not in params
    target = _state()
    assert restore_results(target, results) is None
    assert target.fit_suggestion[3] is saved.fit_suggestion[3]
    short = {"radius": saved.radius, "fit_suggestion": [None]}
    restore_results(target, short)  # wrong length: ignored
    assert target.fit_suggestion[3] is saved.fit_suggestion[3]
