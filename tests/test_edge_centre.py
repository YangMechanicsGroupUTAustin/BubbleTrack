"""Centre and radius come from the same definition (owner report 2026-11-13).

The owner saw a correct outline with an "equal-area circle" of the right size shifted ~50 px
sideways: the area radius was drawn at the robust circle's centre, which on an elongated bubble
hugs one flank.  It happened whenever the outline touched excluded pixels (a saturated glare
blob of >= 20 px next to the wall), because such a region counts as clipped.

Repro (``centre_offset_demo.py``): a 1.7:1 bubble (area radius 60, long axis vertical) with a
5x5 saturated blob just outside its right wall; before the fix the reported centre was 51.5 px
off with confidence 1.00.

Since edge-1.3.0 the default circle itself is the least-squares circle through the outline
(owner decision 2026-11-13): on this bubble R 62.9 (clean) / 63.0 (glare) centred within 0.5 px,
instead of the robust ray circle R 98.9 hugging the left flank.  It is 4.8 / 5.0 % off the
area radius, within the 5 % tolerance, so no ``non_circular`` flag.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pytest

from _edge_synth import Bubble, Scene, boundary_radius, render, truth
from bubbletrack.controller.display_mixin import FITTED_CIRCLE, circle_legend
from bubbletrack.controller.edge_support import store_fit
from bubbletrack.model.edge import EdgeParams, detect_edge
from bubbletrack.model.edge_pipeline import EdgeJob, fit_from_result
from bubbletrack.model.provenance import FitSource
from bubbletrack.model.state import AppState

H, W = 250, 400
RC, CC = 125.0, 200.0
ON, OFF = EdgeParams(non_round=True), EdgeParams()


@lru_cache(maxsize=None)
def _frame(aspect: float, glare: bool):
    b = Bubble(rc=RC, cc=CC, r=60.0, aspect=aspect, angle=np.pi / 2)
    img, meta = render(Scene(bubbles=(b,)))
    if glare:  # a 5x5 saturated blob just outside the right wall
        c0 = int(round(CC + float(boundary_radius(b, np.array([0.0]))[0]))) + 1
        img = img.copy()
        img[int(RC) - 2:int(RC) + 3, c0:c0 + 5] = img.max()
    return img, meta["ref"], truth(b)["r_eq"]


def _measure(aspect: float, glare: bool, params: EdgeParams):
    img, ref, r_eq = _frame(aspect, glare)
    return detect_edge(img, (1, H), (1, W), reference=ref, params=params), r_eq


def test_the_repro_touches_excluded_pixels():
    res, _ = _measure(1.7, True, ON)
    assert {"clipped", "saturation"} <= res.flags


@pytest.mark.parametrize("glare", [False, True], ids=["clean", "glare"])
def test_area_mode_centre_is_the_regions_centroid(glare):
    res, r_eq = _measure(1.7, glare, ON)
    assert res.accepted and res.radius == pytest.approx(r_eq, rel=0.01)
    assert (res.rc, res.cc) == pytest.approx((RC + 1, CC + 1), abs=1.0)       # 1-based truth
    assert (res.rc, res.cc) == pytest.approx(res.centroid, abs=1e-9)
    assert abs(res.provenance["steep_r_circle"] - r_eq) > 0.3 * r_eq           # the ray circle hugs a flank
    assert res.r_circle == pytest.approx(r_eq, rel=0.055)                      # the outline circle does not


@pytest.mark.parametrize("glare", [False, True], ids=["clean", "glare"])
def test_circle_mode_reports_the_centred_outline_circle(glare):
    res, r_eq = _measure(1.7, glare, OFF)
    assert res.radius == res.r_circle and 0 < res.radius - r_eq < 0.055 * r_eq
    assert (res.rc, res.cc) == pytest.approx(res.provenance["circle"]["centre"], abs=1e-9)
    assert (res.rc, res.cc) == pytest.approx((RC + 1, CC + 1), abs=1.0)
    off = abs(res.radius - res.r_area) / res.r_area
    assert ("non_circular" in res.flags) == (off > 0.05) and "clipped_not_round" not in res.flags
    state = AppState(detection_mode="edge").with_results_initialized(1)
    store_fit(state, 0, fit_from_result(res, EdgeJob(OFF, None, 400.0)), FitSource.AUTO)
    assert circle_legend(state, 0) == FITTED_CIRCLE                            # drawn as what it is
    assert state.circle_fit_par[0] == pytest.approx((res.rc, res.cc))


@pytest.mark.parametrize("params", [OFF, ON], ids=["circle", "area"])
@pytest.mark.parametrize("glare", [False, True], ids=["clean", "glare"])
def test_a_round_bubble_stays_exact(glare, params):
    res, r_eq = _measure(1.0, glare, params)
    assert res.accepted and res.radius == pytest.approx(r_eq, rel=0.002)
    assert (res.rc, res.cc) == pytest.approx((RC + 1, CC + 1), abs=0.1)
    assert not {"non_circular", "clipped_not_round", "clipped_unmeasurable"} & res.flags


def test_an_area_mode_guess_is_drawn_where_its_outline_is():
    """Below the gate too, the area radius sits at the region's centroid."""
    img, _, _ = _frame(1.7, True)
    res = detect_edge(img, (1, H), (1, W), reference=None,
                      params=EdgeParams(non_round=True, confidence_gate=1.0))
    assert not res.accepted
    assert (res.rc, res.cc) == pytest.approx(res.centroid, abs=1e-9)
