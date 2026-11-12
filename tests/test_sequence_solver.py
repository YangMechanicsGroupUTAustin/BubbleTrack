"""Sequence solver: API contract, seeding, repair, anchors, review list, cost (scripted detectors)."""

from __future__ import annotations

import logging
import time
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from _sequence_synth import ScriptedDetector, observe, rayleigh_radii, truth_track
from bubbletrack.model.sequence import (
    FrameSolution, Observation, SequenceConfig, SequenceResult, solve_sequence,
)

NAN = float("nan")


def world(n=40, cycles=((-1, 31, 60.0), (30, 14, 25.0)), **kw):
    """A bubble in every frame (two cycles by default), and a detector that finds it near a prior."""
    truth = truth_track(rayleigh_radii(n, cycles, floor=2.0), **kw)
    return truth, ScriptedDetector(truth)


def close(sol: FrameSolution, t, tol=1e-6) -> bool:
    """A verified (not lost) solution on the true circle."""
    return (t is not None and sol.ok and sol.source in ("raw", "tracked", "anchor")
            and abs(sol.radius - t[2]) <= tol and np.hypot(sol.rc - t[0], sol.cc - t[1]) <= tol)


# --------------------------------------------------------------------------- #
#  Contract
# --------------------------------------------------------------------------- #
def test_observation_ok_and_coercion():
    o = Observation(np.float32(12.5), 3, 4.0, 0.8, {"non_circular"})
    assert o.ok and isinstance(o.radius, float) and o.flags == frozenset({"non_circular"})
    assert not Observation(NAN, 1, 1, 0.5).ok
    assert not Observation(-1.0, 1, 1, 0.5).ok
    assert not Observation(5.0, NAN, 1, 0.5).ok
    with pytest.raises(FrozenInstanceError):
        o.radius = 3.0  # type: ignore[misc]
    with pytest.raises(ValueError):
        Observation("x", 1, 1, 0.5)  # type: ignore[arg-type]


def test_frame_solution_rejects_unknown_source():
    with pytest.raises(ValueError):
        FrameSolution(1.0, 1.0, 1.0, 1.0, "guess", frozenset(), 0)


@pytest.mark.parametrize("bad", [dict(min_confidence=1.5), dict(cloud_frames=0), dict(gate_centre_px=-1),
                                 dict(wide_gate_factor=0.5), dict(error_window=2.5), dict(release_ratio=NAN)])
def test_config_validation(bad):
    with pytest.raises(ValueError):
        SequenceConfig(**bad)


def test_config_integer_fields_are_coerced():
    cfg = SequenceConfig(error_window=5.0, max_redetect_coast=0)
    assert isinstance(cfg.error_window, int) and cfg.max_redetect_coast == 0


def test_result_shape_and_types():
    truth, det = world()
    res = solve_sequence(observe(truth), det)
    assert isinstance(res, SequenceResult)
    assert len(res.frames) == len(truth)
    assert all(isinstance(f, FrameSolution) for f in res.frames)
    assert all(isinstance(i, int) and isinstance(r, tuple) for i, r in res.review)
    assert res.cycles and all(a <= b for a, b in res.cycles)


def test_bad_inputs_raise():
    truth, det = world(n=10, cycles=((-1, 12, 30.0),))
    raw = observe(truth)
    with pytest.raises(TypeError):
        solve_sequence(list(raw[:5]) + [(1, 2, 3)], det)
    with pytest.raises(ValueError):
        solve_sequence(raw, det, anchors={12: (1.0, 2.0, 3.0)})
    with pytest.raises(ValueError):
        solve_sequence(raw, det, anchors={3: (1.0, 2.0)})
    with pytest.raises(ValueError):
        solve_sequence(raw, det, anchors={3: (NAN, 2.0, 5.0)})


def test_inputs_are_not_mutated_and_the_solve_is_deterministic():
    truth, det = world()
    raw = list(observe(truth, replace={10: Observation(20.0, 40.0, 300.0, 0.9)}))
    anchors = {5: truth[5]}
    raw_copy, anchors_copy = list(raw), dict(anchors)
    a = solve_sequence(raw, det, anchors=anchors)
    b = solve_sequence(raw, det, anchors=anchors)
    assert raw == raw_copy and anchors == anchors_copy
    assert repr(a) == repr(b)                      # repr: NaN fields compare equal


# --------------------------------------------------------------------------- #
#  Seeding and the redetect budget
# --------------------------------------------------------------------------- #
def test_clean_sequence_needs_no_redetect():
    truth, det = world()
    res = solve_sequence(observe(truth), det)
    assert det.calls == [] and res.redetect_calls == 0
    for i, t in enumerate(truth):
        if t is not None:
            assert res.frames[i].source == "raw" and close(res.frames[i], t)


def test_seed_is_the_most_confident_ok_frame():
    truth, det = world(n=30, cycles=((-1, 32, 50.0),))
    raw = list(observe(truth, confidence=0.6))
    raw[17] = Observation(truth[17][2], truth[17][0], truth[17][1], 0.97)
    assert solve_sequence(raw, det).seed_frame == 17


def test_seed_prefers_a_frame_its_neighbours_agree_with():
    truth, det = world(n=30, cycles=((-1, 32, 50.0),))
    raw = list(observe(truth, confidence=0.8))
    raw[3] = Observation(20.0, 40.0, 320.0, 0.99)          # an isolated confident distractor
    raw[18] = Observation(truth[18][2], truth[18][0], truth[18][1], 0.9)
    res = solve_sequence(raw, det)
    assert res.seed_frame == 18
    assert close(res.frames[3], truth[3]) and res.frames[3].source == "tracked"


def test_wrong_objects_are_repaired_and_only_those_frames_redetected():
    truth, det = world()
    wrong = {8: Observation(30.0, 40.0, 320.0, 0.9),                    # distractor elsewhere
             12: Observation(0.55 * truth[12][2], truth[12][0], truth[12][1], 0.9),   # inner ring
             20: Observation(NAN, NAN, NAN, 0.0)}                        # missed
    res = solve_sequence(observe(truth, replace=wrong), det)
    assert det.frames_called == [8, 12, 20]
    for i in wrong:
        assert res.frames[i].source == "tracked" and close(res.frames[i], truth[i])
        assert "disagree" in res.frames[i].flags


def test_lost_when_nothing_consistent_is_found_then_reacquired():
    truth, _ = world()
    gone = {10, 11, 12}
    raw = observe(truth, replace={i: Observation(25.0, 30.0, 330.0, 0.9) for i in gone})
    det = ScriptedDetector(truth, overrides={i: ScriptedDetector.nothing() for i in gone})
    res = solve_sequence(raw, det, anchors={2: truth[2]})
    for i in gone:
        f = res.frames[i]
        assert f.source == "lost" and "lost" in f.flags and f.confidence == 0.0
        assert f.radius == 25.0                       # the unverified raw value is kept
    assert close(res.frames[13], truth[13]) and "reacquired" in res.frames[13].flags


def test_redetect_exceptions_are_logged_and_count_as_failures(caplog):
    truth, det = world(n=20, cycles=((-1, 22, 40.0),))

    def boom(idx, prior):
        raise OSError("unreadable frame")
    raw = observe(truth, replace={9: Observation(NAN, NAN, NAN, 0.0)})
    with caplog.at_level(logging.WARNING):
        res = solve_sequence(raw, boom)
    assert res.frames[9].source == "lost"
    assert "redetect failed on frame 9" in caplog.text


def test_redetect_must_return_an_observation():
    truth, _ = world(n=20, cycles=((-1, 22, 40.0),))
    raw = observe(truth, replace={9: Observation(NAN, NAN, NAN, 0.0)})
    with pytest.raises(TypeError):
        solve_sequence(raw, lambda idx, prior: (1.0, 2.0, 3.0))


def test_legacy_nan_confidence_never_gates():
    truth, det = world(n=30, cycles=((-1, 32, 50.0),))
    raw = observe(truth, confidence=NAN)
    res = solve_sequence(raw, det)
    assert det.calls == []
    assert all(f.source == "raw" for f, t in zip(res.frames, truth) if t is not None)
    assert not any("low_confidence" in f.flags for f in res.frames)


def test_weak_raw_inside_the_gate_is_kept_or_optionally_refined():
    truth, det = world(n=30, cycles=((-1, 32, 50.0),))
    t = truth[15]
    raw = observe(truth, replace={15: Observation(t[2] + 1.0, t[0], t[1], 0.2)})
    res = solve_sequence(raw, det)
    assert det.calls == [] and res.frames[15].source == "raw"
    assert "low_confidence" in res.frames[15].flags
    res = solve_sequence(raw, det, config=SequenceConfig(refine_below_confidence=0.3))
    assert det.frames_called == [15]
    assert res.frames[15].source == "tracked" and close(res.frames[15], t)


def test_nothing_to_track():
    raw = tuple(Observation(NAN, NAN, NAN, 0.0) for _ in range(6))
    res = solve_sequence(raw, lambda i, p: Observation(NAN, NAN, NAN, 0.0))
    assert res.seed_frame == -1
    assert all(f.source == "none" and f.cycle == -1 for f in res.frames)
    assert res.collapse_frames == () and res.cycles == () and res.review == ()


def test_empty_sequence():
    res = solve_sequence((), lambda i, p: Observation(NAN, NAN, NAN, 0.0))
    assert res.frames == () and res.seed_frame == -1 and res.review == () and res.cycles == ()


# --------------------------------------------------------------------------- #
#  Anchors
# --------------------------------------------------------------------------- #
def test_anchors_on_the_first_last_and_adjacent_frames():
    truth, det = world(n=20, cycles=((-1, 22, 40.0),))
    anchors = {0: truth[0], 1: truth[1], 19: truth[19], 9: (NAN, NAN, 0.0), 10: truth[10]}
    res = solve_sequence(observe(truth), det, anchors=anchors)
    assert [res.frames[i].source for i in anchors] == ["anchor"] * len(anchors)
    assert all(close(res.frames[i], truth[i]) for i in range(20) if i not in anchors)
    assert det.calls == []


def test_anchor_is_copied_and_seeds_the_track():
    truth, det = world()
    raw = observe(truth)
    a = (truth[6][0] + 1.0, truth[6][1], truth[6][2] + 2.0)
    res = solve_sequence(raw, det, anchors={6: a})
    f = res.frames[6]
    assert (f.rc, f.cc, f.radius, f.source, f.confidence) == (a[0], a[1], a[2], "anchor", 1.0)
    assert res.seed_frame == 6
    assert 6 not in [i for i, _ in res.review]


def test_anchor_overrides_a_confident_wrong_track():
    """Raw follows a distractor everywhere (confidently); one anchor on the bubble fixes every frame."""
    truth, det = world(n=30, cycles=((-1, 32, 50.0),))
    raw = tuple(Observation(34.0, 46.0, 301.0, 0.95) for _ in truth)
    res = solve_sequence(raw, det, anchors={15: truth[15]})
    assert all(close(res.frames[i], truth[i]) for i in range(30) if i != 15)
    assert det.frames_called == [i for i in range(30) if i != 15]


def test_anchors_reset_the_track_state():
    """The bubble jumps 60 px between frames 14 and 15: an anchor after the jump re-seeds the track."""
    radii = rayleigh_radii(30, [(-1, 32, 40.0)])
    truth = tuple(None if r <= 0 else (125.0, 200.0 + (60.0 if t >= 15 else 0.0), float(r))
                  for t, r in enumerate(radii))
    det = ScriptedDetector(truth)
    res = solve_sequence(observe(truth), det, anchors={5: truth[5], 20: truth[20]})
    assert all(close(res.frames[i], truth[i]) for i in range(30) if i not in (5, 20))
    assert res.frames[15].source in ("raw", "tracked")


def test_an_empty_anchor_is_a_barrier_without_a_bubble():
    truth, det = world(n=30, cycles=((-1, 32, 50.0),))
    res = solve_sequence(observe(truth), det, anchors={10: (NAN, NAN, -1.0)})
    f = res.frames[10]
    assert f.source == "anchor" and not f.ok
    assert all(close(res.frames[i], truth[i]) for i in range(30) if i != 10)


# --------------------------------------------------------------------------- #
#  Review list and cost
# --------------------------------------------------------------------------- #
def test_review_orders_likely_wrong_first_and_ranks_collapse_neighbours_low():
    truth, _ = world()
    det = ScriptedDetector(truth, overrides={18: ScriptedDetector.nothing()})
    raw = list(observe(truth, replace={18: Observation(25.0, 30.0, 330.0, 0.9),
                                       8: Observation(30.0, 40.0, 320.0, 0.9)}))
    raw[22] = Observation(raw[22].radius, raw[22].rc, raw[22].cc, 0.95, {"non_circular"})
    res = solve_sequence(raw, det)
    order = [i for i, _ in res.review]
    reasons = dict(res.review)
    assert order[0] == 18 and reasons[18][0] == "lost"
    assert reasons[8] == ("disagree",) and reasons[22] == ("non_circular",)
    assert order.index(8) < order.index(22)
    c = res.collapse_frames[0]
    assert "collapse" in res.frames[c].flags and reasons[c] == ("collapse",)
    assert all(order.index(i) < order.index(c) for i in (8, 18, 22))


def test_solver_cost_is_below_two_ms_per_frame():
    n = 2000
    truth = truth_track(rayleigh_radii(n, [(k, 40, 60.0 - k / 50) for k in range(0, n - 40, 40)]),
                        velocity=(0.0, 0.01))
    raw = observe(truth, replace={i: Observation(20.0, 30.0, 330.0, 0.9) for i in range(5, n, 37)})
    det = ScriptedDetector(truth)
    t0 = time.perf_counter()
    res = solve_sequence(raw, det)
    ms = (time.perf_counter() - t0) * 1e3 / n
    assert ms < 2.0, f"{ms:.2f} ms/frame"
    assert len(det.calls) <= 2 * len(range(5, n, 37)) + 2 * len(res.collapse_frames) + 10
