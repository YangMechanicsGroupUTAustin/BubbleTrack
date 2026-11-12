"""Clipped bubbles (owner decision 2) and fragments (decision 3) through ``detect_edge``.

A bubble cut by the ROI border is measured as its visible area (outline closed along the
border) plus the part of its robust circle fit outside the ROI: flag ``clipped_completed``.

Measured (edge-1.1.0, R = 60 synthetic bubble with rim, ring and saturated spot, with
reference): |R - 60| / 60 <= 0.15 % and |R - r_circle| / r_circle <= 0.01 % for clip
fractions 10-60 % of the diameter (hidden 5-62 % of the area), in a corner and against an
ROI edge; confidence 1.00 up to 50 % clipped, 0.75 at 60 %, 0.39 at 75 %.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pytest

from _edge_shapes import H, W, disk, render_masks
from _edge_synth import Bubble, Scene, render
from bubbletrack.model.edge import EdgeParams, detect_edge

FULL = ((1, H), (1, W))
R = 60.0


@lru_cache(maxsize=None)
def _clipped(rc: float, cc: float):
    img, meta = render(Scene(bubbles=(Bubble(rc=rc, cc=cc, r=R),)))
    return img, meta["ref"]


def _check_completed(res, rc0: float, cc0: float, roi_rows=(1, H), roi_cols=(1, W)):
    assert "clipped_completed" in res.flags and "clipped" in res.flags
    assert res.radius_kind == "area+circle_completion"
    assert res.radius == pytest.approx(R, rel=0.005)
    assert res.radius == pytest.approx(res.r_circle, rel=0.005)
    assert (res.rc, res.cc) == pytest.approx((rc0 + 1, cc0 + 1), abs=0.5)    # circle-fit centre
    assert res.centroid == pytest.approx((rc0 + 1, cc0 + 1), abs=1.0)         # completed region
    assert res.area_px == pytest.approx(np.pi * res.radius ** 2)
    e = res.edge_xy                                                           # no point on the border
    assert e[:, 0].min() > roi_rows[0] - 0.5 + 1e-6 and e[:, 0].max() < roi_rows[1] + 0.5 - 1e-6
    assert e[:, 1].min() > roi_cols[0] - 0.5 + 1e-6 and e[:, 1].max() < roi_cols[1] + 0.5 - 1e-6
    assert res.solidity > 0.98 and "non_circular" not in res.flags


@pytest.mark.parametrize("frac", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
def test_round_bubble_clipped_at_the_top_equals_its_circle(frac):
    rc0, cc0 = R * (1 - 2 * frac), 200.0
    img, ref = _clipped(rc0, cc0)
    res = detect_edge(img, *FULL, reference=ref)
    _check_completed(res, rc0, cc0)
    assert res.accepted


@pytest.mark.parametrize("rc0, cc0", [(24.0, 24.0), (220.0, 372.0)], ids=["top_left", "bottom_right"])
def test_round_bubble_clipped_in_a_corner(rc0, cc0):
    img, ref = _clipped(rc0, cc0)
    _check_completed(detect_edge(img, *FULL, reference=ref), rc0, cc0)


def test_bubble_clipped_by_the_roi_edge_rather_than_the_image():
    img, ref = _clipped(100.0, 200.0)
    rows, cols = (71, 250), (1, 400)                    # ROI starts 30 px below the top of the bubble
    res = detect_edge(img, rows, cols, reference=ref)
    _check_completed(res, 100.0, 200.0, rows, cols)
    assert res.mask.shape == (180, 400)


def test_confidence_falls_as_more_of_the_bubble_is_extrapolated():
    conf = []
    for frac in (0.3, 0.6, 0.75):
        img, ref = _clipped(R * (1 - 2 * frac), 200.0)
        res = detect_edge(img, *FULL, reference=ref)
        conf.append(res.confidence)
        assert res.provenance["confidence_terms"]["completion"] <= 1.0
    assert conf[0] >= 0.9 and conf[0] > conf[1] > conf[2]


def test_unclipped_bubble_is_not_completed():
    img, ref = _clipped(125.0, 200.0)
    res = detect_edge(img, *FULL, reference=ref)
    assert res.radius_kind == "area" and not {"clipped", "clipped_completed"} & res.flags
    assert res.radius == pytest.approx(R, rel=0.003)


# ------------------------------------------------------------------ fragments

MAIN = (125.0, 180.0, 40.0)
SATELLITES = ((70.0, 230.0, 6.0), (180.0, 140.0, 5.0), (125.0, 245.0, 7.0))


def _disk(cy, cx, r):
    return lambda yy, xx: disk(yy, xx, cy, cx, r)


@lru_cache(maxsize=None)
def _scene(bodies: tuple):
    img, radii = render_masks([(_disk(*b), 0.06) for b in bodies])
    return img, radii, render_masks([], seed=99)[0]


def test_main_body_with_three_small_satellites():
    img, radii, ref = _scene((MAIN, *SATELLITES))
    res = detect_edge(img, *FULL, reference=ref)
    assert res.radius == pytest.approx(radii[0], rel=0.01)
    assert res.fragment_count == 3 and "fragmented" in res.flags
    assert res.fragment_area_px == pytest.approx(np.pi * sum(r ** 2 for r in radii[1:]), rel=0.15)
    assert res.accepted


@pytest.mark.parametrize("target", [0, 1], ids=["prior_on_main", "prior_on_satellite"])
def test_large_satellite_and_a_prior_choose_the_main_body(target):
    bodies = (MAIN, (125.0, 268.0, 32.0))
    img, radii, ref = _scene(bodies)
    rc, cc, r = bodies[target]
    res = detect_edge(img, *FULL, reference=ref, params=EdgeParams(prior=(rc + 1, cc + 1, r)))
    assert res.radius == pytest.approx(radii[target], rel=0.01)
    assert res.fragment_count == 1
    assert res.fragment_area_px == pytest.approx(np.pi * radii[1 - target] ** 2, rel=0.05)
    assert (res.rc, res.cc) == pytest.approx((rc + 1, cc + 1), abs=0.3)


def test_thin_crescent_whose_centroid_lies_in_the_bite():
    """A crescent's centroid is outside it: it is reported as the centre but never used to find the
    body, so a prior centred there (tracking) still finds the crescent."""
    def crescent(yy, xx):
        return disk(yy, xx, 125, 200, 60) & ~disk(yy, xx, 125, 216, 54)

    img, radii = render_masks([(crescent, 0.06)])
    ref = render_masks([], seed=99)[0]
    first = detect_edge(img, *FULL, reference=ref)
    assert first.radius == pytest.approx(radii[0], rel=0.01) and first.accepted
    assert not first.mask[int(round(first.rc - 1)), int(round(first.cc - 1))]
    res = detect_edge(img, *FULL, reference=ref, params=EdgeParams(prior=(first.rc, first.cc, first.radius)))
    assert res.radius == pytest.approx(radii[0], rel=0.01) and res.accepted
