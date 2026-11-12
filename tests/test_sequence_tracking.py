"""Sequence solver tracking rules: motion gates, jumps, collapses, re-acquisition, cloud (scripted)."""

from __future__ import annotations

import numpy as np
import pytest

from _sequence_synth import ScriptedDetector, observe, rayleigh_radii, truth_track
from bubbletrack.model.sequence import Observation, SequenceConfig, solve_sequence

NAN = float("nan")


def close(sol, t, tol=1e-6) -> bool:
    """A verified (not lost) solution on the true circle."""
    return (t is not None and sol.ok and sol.source in ("raw", "tracked", "anchor")
            and abs(sol.radius - t[2]) <= tol and np.hypot(sol.rc - t[0], sol.cc - t[1]) <= tol)


def all_correct(res, truth, skip=()) -> bool:
    return all(close(res.frames[i], t) for i, t in enumerate(truth) if t is not None and i not in skip)


def moving(radii, path):
    return tuple(None if not r > 0 else (float(path[t][0]), float(path[t][1]), float(r))
                 for t, r in enumerate(radii))


# --------------------------------------------------------------------------- #
#  Motion
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("speed", [0.0, 5.0, 10.0, 20.0])
def test_constant_drift_is_followed_without_redetect(speed):
    truth = truth_track(rayleigh_radii(24, [(-1, 26, 30.0)], floor=2.0),
                        centre=(125.0, 200.0 - 11.5 * speed), velocity=(0.0, speed))
    det = ScriptedDetector(truth)
    res = solve_sequence(observe(truth), det)
    assert all_correct(res, truth)
    assert det.calls == []


@pytest.mark.parametrize("speed", [5.0, 20.0])
def test_drift_with_a_distractor_every_third_frame_is_repaired(speed):
    truth = truth_track(rayleigh_radii(24, [(-1, 26, 30.0)], floor=2.0),
                        centre=(125.0, 200.0 - 11.5 * speed), velocity=(0.0, speed))
    wrong = {i: Observation(34.0, 46.0, 301.0, 0.99) for i in range(1, 24, 3)}
    det = ScriptedDetector(truth)
    res = solve_sequence(observe(truth, confidence=0.9, replace=wrong), det)
    assert all_correct(res, truth)
    assert set(det.frames_called) == set(wrong)


def test_accelerating_jet_is_followed():
    """Still during growth, then 2 px/frame^2 (about 24 px/frame at the collapse)."""
    n = 26
    path = [(125.0, 110.0 + (0.5 * 2.0 * max(t - 12, 0) ** 2)) for t in range(n)]
    truth = moving(rayleigh_radii(n, [(-1, 27, 40.0)], floor=2.0), path)
    res = solve_sequence(observe(truth), ScriptedDetector(truth))
    assert all_correct(res, truth)


def test_a_sudden_jump_is_accepted_by_the_wide_gate_and_flagged():
    n = 24
    path = [(125.0, 180.0 + (33.0 if t >= 10 else 0.0)) for t in range(n)]
    truth = moving(rayleigh_radii(n, [(-1, 26, 35.0)], floor=2.0), path)
    res = solve_sequence(observe(truth), ScriptedDetector(truth), anchors={4: truth[4]})
    assert all_correct(res, truth, skip=(4,))
    assert "jump" in res.frames[10].flags
    assert 10 in [i for i, _ in res.review]


def test_a_fixed_gate_would_lose_a_fast_bubble_but_the_adaptive_gate_does_not():
    speed = 20.0
    truth = truth_track(rayleigh_radii(24, [(-1, 26, 30.0)], floor=2.0),
                        centre=(125.0, 200.0 - 11.5 * speed), velocity=(0.0, speed))
    fixed = SequenceConfig(gate_speed_k=0.0, gate_unknown_frac=0.0, gate_error_k=0.0, velocity_damping=0.0,
                           coast_growth=0.0, wide_gate_factor=1.0)
    assert not all_correct(solve_sequence(observe(truth), ScriptedDetector(truth), config=fixed), truth)
    assert all_correct(solve_sequence(observe(truth), ScriptedDetector(truth)), truth)


# --------------------------------------------------------------------------- #
#  Loss and re-acquisition
# --------------------------------------------------------------------------- #
def test_redetect_stops_after_a_long_loss_and_the_track_is_reacquired():
    truth = truth_track(rayleigh_radii(60, [(-1, 62, 60.0)], floor=2.0))
    gone = set(range(20, 45))
    raw = observe(truth, replace={i: Observation(NAN, NAN, NAN, 0.0) for i in gone})
    det = ScriptedDetector(truth, overrides={i: ScriptedDetector.nothing() for i in gone})
    res = solve_sequence(raw, det, anchors={5: truth[5]})
    cfg = SequenceConfig()
    assert len(det.calls) == cfg.max_redetect_coast
    assert all(res.frames[i].source == "lost" for i in gone)
    assert "reacquired" in res.frames[45].flags and close(res.frames[45], truth[45])


def test_reacquisition_after_a_loss_needs_more_confidence():
    truth = truth_track(rayleigh_radii(30, [(-1, 32, 50.0)], floor=2.0))
    t11 = truth[11]
    raw = observe(truth, replace={10: Observation(NAN, NAN, NAN, 0.0),
                                  11: Observation(t11[2], t11[0], t11[1], 0.2)})
    det = ScriptedDetector(truth, overrides={10: ScriptedDetector.nothing(), 11: ScriptedDetector.nothing()})
    res = solve_sequence(raw, det, anchors={2: truth[2]})
    assert res.frames[11].source == "lost"           # 0.2 would pass the tight gate without the loss
    assert "reacquired" in res.frames[12].flags


def test_the_redetect_cache_serves_both_passes_between_two_anchors():
    truth = truth_track(rayleigh_radii(30, [(-1, 32, 50.0)], floor=2.0))
    wrong = {i: Observation(25.0, 40.0, 330.0, 0.9) for i in (12, 13, 14)}
    det = ScriptedDetector(truth)
    res = solve_sequence(observe(truth, replace=wrong), det, anchors={4: truth[4], 24: truth[24]})
    assert all_correct(res, truth, skip=(4, 24))
    assert sorted(i for i, _ in det.calls) == [12, 13, 14]


# --------------------------------------------------------------------------- #
#  Collapses and the cloud
# --------------------------------------------------------------------------- #
def test_sharp_collapse_and_rebound_are_tracked_without_redetect():
    radii = rayleigh_radii(44, [(-1, 27, 50.0), (26, 11, 22.0)], floor=1.8)
    truth = truth_track(radii)
    det = ScriptedDetector(truth)
    res = solve_sequence(observe(truth), det)
    assert all_correct(res, truth)
    assert det.calls == []
    assert res.collapse_frames[0] == 26
    assert res.frames[26].cycle == 0 and res.frames[27].cycle == 1


def _cloud_world(residual=None, fragmented=False):
    """Collapse at frame 26, rebound, second collapse at 37, then 7 frames of fragments."""
    radii = rayleigh_radii(44, [(-1, 27, 50.0), (26, 11, 22.0)], floor=1.8)
    radii[38:] = NAN
    truth = truth_track(radii)
    rng = np.random.default_rng(3)
    junk = {}
    for i in range(38, 44):
        if residual is not None:
            junk[i] = Observation(residual, 125.0 + rng.normal(0, 0.3), 200.0 + rng.normal(0, 0.3), 0.2)
        elif fragmented:
            junk[i] = Observation(12.0, 125.0 + rng.normal(0, 3), 200.0 + rng.normal(0, 3), 0.4, {"fragmented"})
        else:
            a = rng.uniform(0, 2 * np.pi)
            junk[i] = Observation(rng.uniform(1.5, 5.0), 125 + 12 * np.sin(a), 200 + 12 * np.cos(a), 0.1)
    return truth, junk


def test_the_fragment_cloud_stops_tracking():
    truth, junk = _cloud_world()
    det = ScriptedDetector(truth, elsewhere=junk)
    res = solve_sequence(observe(truth, replace=junk), det)
    assert res.collapse_frames == (26, 37)
    assert all(res.frames[i].source == "none" and "cloud" in res.frames[i].flags for i in range(38, 44))
    assert all(res.frames[i].cycle == -1 for i in range(38, 44))
    assert res.review[[i for i, _ in res.review].index(38)][1] == ("cloud",)
    assert max(i for i, _ in det.calls) <= 40               # no redetect once the cloud is declared
    assert all_correct(res, truth)


def test_a_persistent_weak_residual_bubble_is_not_a_cloud():
    truth, junk = _cloud_world(residual=6.0)            # 6 < 0.15 x 50, confidence 0.2, but steady
    res = solve_sequence(observe(truth, replace=junk), ScriptedDetector(truth, elsewhere=junk))
    assert not any("cloud" in f.flags for f in res.frames)


def test_fragmented_observations_after_a_collapse_are_a_cloud():
    truth, junk = _cloud_world(fragmented=True)
    res = solve_sequence(observe(truth, replace=junk), ScriptedDetector(truth, elsewhere=junk))
    assert all(res.frames[i].source == "none" and "cloud" in res.frames[i].flags for i in range(38, 44))


def test_small_weak_frames_without_a_collapse_are_not_a_cloud():
    truth = truth_track(rayleigh_radii(40, [(-1, 42, 60.0)], floor=2.0))
    weak = {i: Observation(3.0, 125.0 + 15 * (i % 2), 200.0, 0.05) for i in range(15, 22)}
    det = ScriptedDetector(truth, overrides={i: ScriptedDetector.nothing() for i in weak})
    res = solve_sequence(observe(truth, replace=weak), det)
    assert not any("cloud" in f.flags for f in res.frames)
    assert all(res.frames[i].source == "lost" for i in weak)


def test_frames_before_nucleation_are_none_and_not_reviewed():
    radii = rayleigh_radii(30, [(3, 30, 60.0)])         # nothing in frames 0..3
    truth = truth_track(radii)
    res = solve_sequence(observe(truth), ScriptedDetector(truth))
    assert all(res.frames[i].source == "none" for i in range(4))
    assert not {0, 1, 2, 3} & {i for i, _ in res.review}
    assert res.cycles[0][0] == 4


def test_a_backward_pass_never_stops_early_so_earlier_cycles_survive():
    """An anchor in the second cycle; between the cycles a small bubble (R 4) is invisible for
    three frames.  Backwards this looks like a cloud, but the pass must go on to cycle one."""
    radii = rayleigh_radii(50, [(-1, 27, 60.0), (31, 20, 40.0)], floor=4.0)
    truth = truth_track(radii)
    blind = {27, 28, 29}
    raw = observe(truth, replace={i: Observation(NAN, NAN, NAN, 0.0) for i in blind})
    det = ScriptedDetector(truth, overrides={i: ScriptedDetector.nothing() for i in blind})
    res = solve_sequence(raw, det, anchors={41: truth[41]})
    assert all(res.frames[i].source == "lost" for i in blind)
    assert all_correct(res, truth, skip=blind | {41})
    assert "reacquired" in res.frames[26].flags
