"""Sequence solver + the real Edge engine on rendered synthetic sequences (exact ground truth).

Scenarios (``_sequence_synth.catalogue``): drift 0-20 px/frame, an accelerating jet, a 35 px
jump, a static target beside a LARGER satellite, and collapse + rebound + fragment cloud.
Three distractor bubbles appear with the target in all but the cloud scenario.

Two set-ups, as in the app:
* ``click``:  Fit All with the user's seed click (on the frame of largest radius), no anchors;
* ``anchor``: Fit All without a click (it picks the larger satellite), one accepted frame.
``redetect`` runs the engine with ``EdgeParams(prior=...)`` (no click).

A frame counts when the true bubble is measurable (R >= 3 px, centre inside the image): it
must be a verified solution with |dR| <= max(1 px, 3 %) and centre error <= max(2 px, 10 % R).
Cloud frames must be ``none`` with the ``cloud`` flag.

Measured (edge-1.0.0): every measurable frame correct in both set-ups; single-frame Fit All
alone gets 12/20 (drift 20, click) and 0-1/20 (no click, satellite picked).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from _sequence_synth import catalogue, render_seq
from bubbletrack.model.edge import EdgeParams, detect_edge
from bubbletrack.model.sequence import Observation, solve_sequence

SPECS = {s.name: s for s in catalogue()}


def _obs(res) -> Observation:
    return Observation(res.radius, res.rc, res.cc, res.confidence, res.flags)


@pytest.fixture(scope="module")
def rendered():
    with ThreadPoolExecutor(4) as ex:
        return dict(zip(SPECS, ex.map(lambda s: render_seq(s), SPECS.values())))


def _run(frames, gt, mode: str):
    h, w = frames.shape[1:]
    ref = frames[0]
    s0 = int(np.nanargmax(gt["R"]))
    click = (gt["rc"][s0], gt["cc"][s0]) if mode == "click" else None
    with ThreadPoolExecutor(4) as ex:
        raw = list(ex.map(lambda f: _obs(detect_edge(f, (1, h), (1, w), reference=ref,
                                                     params=EdgeParams(seed=click))), frames))
    calls = []

    def redetect(i, prior):
        calls.append(i)
        return _obs(detect_edge(frames[i], (1, h), (1, w), reference=ref, params=EdgeParams(prior=prior)))
    anchors = {s0: (gt["rc"][s0], gt["cc"][s0], gt["R"][s0])} if mode == "anchor" else None
    return raw, solve_sequence(raw, redetect, anchors=anchors), calls


@pytest.fixture(scope="module")
def solved(rendered):
    """``solved(name, mode)`` -> (frames, gt, raw, result, redetect calls), computed once."""
    cache = {}

    def get(name: str, mode: str):
        if (name, mode) not in cache:
            frames, gt = rendered[name]
            cache[(name, mode)] = (frames, gt, *_run(frames, gt, mode))
        return cache[(name, mode)]
    return get


def _verdicts(frames, gt, sol_of):
    """{frame: correct?} over the measurable frames, and the cloud frames separately."""
    h, w = frames.shape[1:]
    out, cloud = {}, []
    for t in range(1, len(frames)):
        r, rc, cc = gt["R"][t], gt["rc"][t], gt["cc"][t]
        if not np.isfinite(r):
            cloud.append(t)
            continue
        if r < 3 or not (1 <= rc <= h and 1 <= cc <= w):
            continue
        ok, sr, src, scc = sol_of(t)
        out[t] = bool(ok and abs(sr - r) <= max(1.0, 0.03 * r) and np.hypot(src - rc, scc - cc) <= max(2.0, 0.1 * r))
    return out, cloud


@pytest.mark.parametrize("mode", ["click", "anchor"])
@pytest.mark.parametrize("name", list(SPECS))
def test_every_measurable_frame_is_solved(solved, name, mode):
    frames, gt, raw, res, calls = solved(name, mode)

    def solved(t):
        f = res.frames[t]
        return f.source in ("raw", "tracked", "anchor") and f.ok, f.radius, f.rc, f.cc
    verdict, cloud = _verdicts(frames, gt, solved)
    wrong = sorted(t for t, ok in verdict.items() if not ok)
    assert not wrong, f"{name}/{mode}: wrong frames {wrong}"
    for t in cloud:
        assert res.frames[t].source == "none" and "cloud" in res.frames[t].flags, f"{name}/{mode}: frame {t}"
    assert len(calls) <= len(frames)


def test_the_solver_beats_single_frame_where_raw_fails(solved):
    """drift 20 with a click: the click is far from the bubble on most frames."""
    frames, gt, raw, res, _ = solved("drift20", "click")
    before, _ = _verdicts(frames, gt, lambda t: (raw[t].ok, raw[t].radius, raw[t].rc, raw[t].cc))
    after, _ = _verdicts(frames, gt, lambda t: (res.frames[t].ok, res.frames[t].radius,
                                                res.frames[t].rc, res.frames[t].cc))
    assert sum(after.values()) == len(after) > sum(before.values())


def test_cloud_scenario_finds_both_collapses_and_stops_at_the_cloud(solved):
    _, _, _, res, calls = solved("collapse_rebound_cloud", "click")
    assert res.collapse_frames == (26, 37)
    first_cloud = next(i for i, f in enumerate(res.frames) if "cloud" in f.flags)
    assert first_cloud == 38
    assert not [i for i in calls if i > first_cloud + 2]
    assert len(res.cycles) == 2 and res.cycles[-1][1] == 37
