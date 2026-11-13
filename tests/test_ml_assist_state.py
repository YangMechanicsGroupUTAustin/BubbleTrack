"""ML assist and the app's results (no scikit-learn): verified examples, the
stored plain-Auto result handed back to ML assist, what is stored per outcome,
suggestions, texts, sessions, the Review category and the new source code."""

from __future__ import annotations

import json

import numpy as np
import pytest

from bubbletrack.controller.review_queue import build_review_items
from bubbletrack.controller.review_segments import build_segments, category_of
from bubbletrack.model import ml_assist as M
from bubbletrack.model.edge import EdgeResult
from bubbletrack.model.edge_pipeline import EdgeFrameFit, EdgeJob, make_params
from bubbletrack.model.export_fit import FIT_SOURCE_NAMES
from bubbletrack.model.ml_pipeline import StoredAuto, storable
from bubbletrack.model.provenance import FitSource
from bubbletrack.model.session import load_session, save_session, session_parameters
from bubbletrack.model.state import AppState, update_state
from bubbletrack.model.teach import BACKGROUND, BUBBLE, MLExamples, MLRegion, MLResult, stroke_pixels
from bubbletrack.ui.chart_style import marker_symbols

N = 8


def _state(n: int = N) -> AppState:
    state = AppState(images=tuple(f"f{i}.tif" for i in range(n))).with_results_initialized(n)
    state.radius[:] = np.linspace(20.0, 34.0, n)
    state.circle_fit_par[:] = [[50.0, 60.0]] * n
    state.circle_xy[:] = [np.array([[30.0, 60.0], [50.0, 80.0], [70.0, 60.0]])] * n
    state.fit_source[:] = FitSource.AUTO
    state.fit_confidence[:] = 0.8
    return update_state(state, gridx=(1, 100), gridy=(1, 120))


# --------------------------------------------------------------------------- #
#  Verified examples
# --------------------------------------------------------------------------- #

def test_verified_frames_are_user_decisions_with_a_bubble():
    state = _state()
    state.fit_source[1] = FitSource.MANUAL
    state.fit_source[2] = FitSource.GUIDED
    state.fit_locked[3] = True                    # accepted Auto result
    state.fit_locked[4] = True                    # "no bubble here": not a bubble example
    state.radius[4] = -1.0
    state.fit_source[5] = FitSource.ML           # computed: not verified
    circles = M.verified_circles(state)
    assert sorted(circles) == [1, 2, 3]
    assert circles[1] == (50.0, 60.0, pytest.approx(22.0))
    assert M.protected_frames(state) == frozenset({1, 2, 3, 4})


def test_examples_for_run_keeps_strokes_and_syncs_verified_frames():
    state = _state()
    state.fit_source[6] = FitSource.MANUAL
    strokes = MLExamples().paint(2, BUBBLE, [(10, 10), (10, 11)]).verify(0, (1.0, 1.0, 5.0))
    state = update_state(state, ml_examples=strokes)
    ex = M.examples_for_run(state)
    assert set(ex.verified) == {6}                # stale circle 0 dropped, frame 6 added
    assert ex.strokes == strokes.strokes
    assert M.examples_for_run(state, MLExamples()).strokes == {}   # pluggable source


def test_readiness_needs_bubble_and_background():
    ready, text = M.readiness(MLExamples())
    assert not ready and text == "0 verified frames, 0 strokes — needs at least one bubble and one background example"
    ex = MLExamples().paint(1, BUBBLE, stroke_pixels([(10, 10), (10, 30)], 2, (100, 120)))
    assert not M.readiness(ex)[0]
    ex = ex.paint(1, BACKGROUND, stroke_pixels([(80, 10), (80, 30)], 2, (100, 120)))
    ex = ex.paint(1, BACKGROUND, stroke_pixels([(90, 90)], 2, (100, 120)))
    assert M.readiness(ex) == (True, "0 verified frames, 3 strokes — ready")
    assert M.readiness(MLExamples().verify(0, (50.0, 60.0, 20.0)))[1] == "1 verified frame, 0 strokes — ready"


# --------------------------------------------------------------------------- #
#  The stored plain-Auto result
# --------------------------------------------------------------------------- #

def test_stored_auto_offers_accepted_and_rejected_auto_results_only():
    state = _state()
    state.radius[1] = -1.0
    state.fit_source[1] = FitSource.NONE          # Auto rejected it (scored)
    state.fit_confidence[1] = 0.12
    state.fit_source[2] = FitSource.ML
    state.fit_flags[3] = frozenset({"ml_guided"})
    state.fit_source[4] = FitSource.MANUAL
    state.radius[5] = -1.0
    state.fit_source[5] = FitSource.NONE
    state.fit_confidence[5] = np.nan              # never measured / deleted
    state.fit_locked[6] = True
    state.fit_source[7] = FitSource.TRACKED
    state.fit_flags[7] = frozenset({"ml_disagrees", "shoulder"})
    stored = M.stored_auto_map(state, range(N))
    assert sorted(stored) == [0, 1, 7]
    assert stored[0].fitted and stored[0].radius == pytest.approx(20.0)
    assert not stored[1].fitted and stored[1].confidence == pytest.approx(0.12)
    assert stored[7].flags == frozenset({"shoulder"})   # ML's own flags are re-decided


def test_stored_auto_reads_like_the_engine_result():
    s = StoredAuto(25.0, 50.0, 60.0, 0.7, frozenset({"shoulder"}), np.zeros((3, 2)), {"area_px": 9.0})
    res = s.as_result(0.3)
    assert res.accepted and res.radius == 25.0 and res.flags == frozenset({"shoulder"})
    fit = s.as_fit()
    assert fit.radius == 25.0 and fit.metrics == {"area_px": 9.0}
    rejected = StoredAuto(np.nan, np.nan, np.nan, 0.1, frozenset())
    assert not rejected.as_result(0.3).accepted and rejected.as_fit().radius == -1.0


# --------------------------------------------------------------------------- #
#  What is stored for each outcome
# --------------------------------------------------------------------------- #

JOB = EdgeJob(params=make_params(), reference=None, max_radius=100.0)


def _edge(radius, conf=0.8, flags=()):
    return EdgeResult(radius=radius, rc=50.0, cc=60.0, radius_sigma=0.1,
                      edge_xy=np.array([[30.0, 60.0], [50.0, 80.0], [70.0, 60.0]]),
                      confidence=conf, flags=frozenset(flags), r_area=radius, outer_radius=np.nan,
                      shoulder_width=np.nan, arc_coverage=1.0, inlier_fraction=1.0,
                      mask=np.zeros((4, 4), bool))


def _region(radius=24.0, conf=0.8):
    contour = np.array([[26.0, 60.0], [50.0, 84.0], [74.0, 60.0], [50.0, 36.0]])
    return MLRegion(radius=radius, rc=50.0, cc=60.0, confidence=conf, mask=np.ones((4, 4), bool),
                    clipped=False, refined=True, terms={"solidity": 0.97}, contour=contour)


def _result(source, radius, flags, guided=None, region=None, conf=0.8):
    return MLResult(radius, 50.0, 60.0, conf, source, frozenset(flags), region, guided, 0.3, 0.02)


PLAIN_REJECTED = EdgeFrameFit(-1.0, np.nan, np.nan, np.empty((0, 2)), None, 0.05,
                              frozenset({"no_candidate"}), "no bubble found")
PLAIN_OK = EdgeFrameFit(22.0, 50.0, 60.0, np.ones((5, 2)), None, 0.9, frozenset(), "")


def test_guided_result_is_an_auto_result_flagged_ml_guided():
    guided = _edge(23.0, flags=("shoulder",))
    out = storable(_result("auto_ml", 23.0, {"ml_guided", "shoulder"}, guided=guided), PLAIN_REJECTED, JOB)
    assert out.source == FitSource.AUTO and out.ml_source == "auto_ml"
    assert out.fit.radius == 23.0 and out.fit.flags == frozenset({"ml_guided", "shoulder"})


def test_ml_outline_is_stored_with_its_contour_and_source_ml():
    out = storable(_result("ml", 24.0, {"ml_outline", "ml_unfamiliar"}, region=_region()), PLAIN_REJECTED, JOB)
    assert out.source == FitSource.ML and out.fit.radius == 24.0
    assert out.fit.edge_xy.shape == (4, 2) and out.fit.confidence == 0.8
    assert out.fit.flags == frozenset({"ml_outline", "ml_unfamiliar"})
    assert out.fit.metrics["solidity"] == 0.97


def test_plain_auto_kept_with_mls_remarks():
    out = storable(_result("auto", 22.0, {"ml_disagrees"}), PLAIN_OK, JOB)
    assert out.source == FitSource.AUTO and out.fit is not PLAIN_OK
    assert out.fit.radius == 22.0 and out.fit.flags == frozenset({"ml_disagrees"})


def test_no_result_looks_like_autos_rejection():
    region = _region(conf=0.1)
    out = storable(_result("ml", 24.0, {"ml_outline", "ml_unfamiliar"}, region=region, conf=0.1),
                   PLAIN_REJECTED, JOB)
    assert out.source == FitSource.NONE and out.fit.radius == -1.0
    assert out.fit.confidence == 0.05                         # plain Auto's own
    assert out.fit.flags == frozenset({"no_candidate", "ml_unfamiliar"})


def test_implausible_ml_outline_is_no_result():
    big = EdgeJob(params=make_params(), reference=None, max_radius=10.0)
    out = storable(_result("ml", 24.0, {"ml_outline"}, region=_region()), PLAIN_REJECTED, big)
    assert out.source == FitSource.NONE


# --------------------------------------------------------------------------- #
#  Suggestions, hints and texts
# --------------------------------------------------------------------------- #

def test_suggestions_leave_out_verified_locked_and_reference_frames():
    state = _state(20)
    state.fit_source[19] = FitSource.MANUAL       # the biggest bubble is verified
    picks = M.suggestions(state, reference=0)
    assert len(picks) == 2 and 19 not in picks and 0 not in picks
    assert max(picks) == 18                       # the biggest left, then a smaller one
    state.radius[:] = -1.0                        # nothing accepted: evenly spaced in time
    assert len(M.suggestions(state, reference=0)) == 2


def test_failure_share_counts_missing_and_reviewed_frames():
    state = _state()
    state.radius[[1, 2]] = -1.0
    state.fit_source[3] = FitSource.MANUAL        # user decisions do not count
    assert M.failure_share(state, range(N), queued={4}) == pytest.approx(3 / 7)
    assert M.failure_share(state, [], queued=()) == 0.0


def test_default_brush_scales_with_the_bubble():
    state = _state()
    state.radius[:] = 50.0
    assert M.default_brush(state) == 6
    state.radius[:] = -1.0
    assert M.default_brush(state) == round(M.BRUSH_FRACTION * M.DEFAULT_BUBBLE_RADIUS)
    state.radius[:] = 15.0                        # Auto measured a small wrong object ...
    state.radius[2] = 75.0
    state.fit_source[2] = FitSource.MANUAL        # ... the user's frame says otherwise
    assert M.default_brush(state) == 9


def test_preview_line_names_the_source_and_warnings():
    from bubbletrack.controller.ml_preview import FramePreview
    from bubbletrack.controller.ml_teach_view import describe
    from bubbletrack.model.ml_pipeline import MLFrameFit

    def line(source, ml_source, flags, frame=4, protected=frozenset()):
        fit = PLAIN_OK.__class__(22.0, 50.0, 60.0, np.ones((5, 2)), None, 0.81, frozenset(flags), "")
        return describe(FramePreview(frame, MLFrameFit(fit, source, ml_source), np.ones((4, 4))), protected)

    assert line(FitSource.AUTO, "auto_ml", {"ml_guided"}) == (
        "Frame 5: Auto (ML-guided), R = 22.0 px (confidence 0.81)", False)
    assert line(FitSource.ML, "ml", {"ml_outline"})[0].endswith("— goes to Review")
    assert "elsewhere" in line(FitSource.AUTO, "auto", {"ml_disagrees"})[0]
    text, warn = line(FitSource.AUTO, "auto", {"ml_unfamiliar"}, protected=frozenset({4}))
    assert warn and "your result here is kept" in text and "unlike the taught ones" in text


def test_run_summary_and_state_line():
    state = _state()
    state.fit_source[1] = FitSource.ML
    state.fit_flags[1] = frozenset({"ml_outline", "ml_unfamiliar"})
    state.fit_flags[2] = frozenset({"ml_guided"})
    state.radius[3] = -1.0
    counts = M.run_counts(state, range(N))
    assert counts == {"guided": 1, "auto": 5, "outline": 1, "none": 1, "unfamiliar": 1}
    assert M.run_summary(counts) == "ML assist: 1 ML outline to review, 1 frame unlike the taught ones"
    ex = MLExamples().verify(0, (1.0, 1.0, 5.0)).verify(4, (1.0, 1.0, 5.0))
    assert M.state_text(True, ex) == "On (2 examples)"
    assert M.state_text(False, ex) == "Off (2 examples)"
    assert M.state_text(False, MLExamples()) == "Off"


# --------------------------------------------------------------------------- #
#  Sessions, the Review queue, provenance
# --------------------------------------------------------------------------- #

def test_session_keeps_examples_and_switch(tmp_path):
    ex = MLExamples().paint(2, BUBBLE, [(10, 10)]).paint(2, BACKGROUND, [(40, 40)]).verify(5, (50.0, 60.0, 20.0))
    state = update_state(_state(), ml_assist=True, ml_examples=ex)
    path = tmp_path / "s.brt"
    save_session(str(path), state)
    data = load_session(str(path))
    assert data["ml_assist"] is True and data["ml_examples"] == ex
    assert session_parameters(state)["ml_examples"] == ex.to_dict()  # hard cases store it too


def test_old_and_damaged_sessions_load_without_ml_assist(tmp_path):
    path = tmp_path / "old.brt"
    save_session(str(path), _state())
    raw = json.loads(path.read_text(encoding="utf-8"))
    for key in ("ml_assist", "ml_examples"):
        raw.pop(key)
    path.write_text(json.dumps(raw), encoding="utf-8")
    data = load_session(str(path))
    assert "ml_examples" not in data                           # AppState defaults apply
    raw.update(ml_assist=True, ml_examples={"version": 99})
    path.write_text(json.dumps(raw), encoding="utf-8")
    data = load_session(str(path))
    assert data["ml_examples"] is None and data["ml_assist"] is False


def test_ml_outlines_are_one_review_category_whatever_else_is_flagged():
    state = _state()
    for i in (2, 3, 4):
        state.fit_source[i] = FitSource.ML
        state.fit_flags[i] = frozenset({"ml_outline"})
    state.fit_confidence[3] = 0.35                             # also low confidence
    state.fit_flags[6] = frozenset({"ml_unfamiliar"})          # context only: not listed alone
    items = build_review_items(state)
    assert [i.frame for i in items] == [3, 2, 4]
    assert {category_of(i) for i in items} == {"ml_outline"}
    segments = build_segments(items)
    assert len(segments) == 1 and segments[0].label == "ML outline" and segments[0].frames == (2, 3, 4)
    state.fit_flags[6] = frozenset({"ml_disagrees"})
    assert category_of(next(i for i in build_review_items(state) if i.frame == 6)) == "ml_disagrees"


def test_ml_source_is_exported_by_name_and_drawn_as_a_computed_dot():
    assert FitSource.ML == 7 and FIT_SOURCE_NAMES[FitSource.ML] == "ML"
    assert list(marker_symbols(np.array([FitSource.ML, FitSource.MANUAL]), None, 2)) == ["o", "d"]
