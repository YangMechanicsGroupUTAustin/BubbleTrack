"""Refinement of ML's outline to the half-contrast wall of the engine view."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import ndimage

from bubbletrack.model.teach import MeasureConfig
from bubbletrack.model.teach.refine import refine_outline

H, W = 120, 160


def _disk(rc, cc, r):
    yy, xx = np.mgrid[0:H, 0:W]
    return np.hypot(yy - rc, xx - cc) < r


def _radius(mask):
    return math.sqrt(mask.sum() / math.pi)


def test_skewed_speckle_noise_does_not_push_the_wall_outward():
    """Moving speckle makes frame / reference dark at random background pixels;
    averaging along the outline keeps the 50 % crossing where the wall is."""
    rng = np.random.default_rng(0)
    view = np.where(_disk(60, 80, 32), 0.2, 1.0)
    view = np.where(rng.random((H, W)) < 0.3, 0.2, view) + rng.normal(0, 0.05, (H, W))
    out = refine_outline(_disk(60, 80, 28), view, MeasureConfig(refine_band=0.15))
    assert out is not None
    assert _radius(out.mask) == pytest.approx(32.0, abs=1.0)
    assert out.diagnostics["shift"] > 0 and out.diagnostics["view_cnr"] > 1


def test_sectors_follow_an_outline_that_is_off_on_one_side():
    view = np.where(_disk(60, 84, 30), 0.2, 1.0)         # the wall is 4 px further right than ML's outline
    ml = _disk(60, 80, 30)
    eight = refine_outline(ml, view, MeasureConfig(refine_sectors=8))
    one = refine_outline(ml, view, MeasureConfig(refine_sectors=1))
    assert ndimage.center_of_mass(eight.mask)[1] == pytest.approx(84.0, abs=1.0)
    assert ndimage.center_of_mass(one.mask)[1] == pytest.approx(80.0, abs=1.0)
    assert len(eight.offsets) == 8 and len(one.offsets) == 1


def test_no_contrast_means_no_refinement():
    assert refine_outline(_disk(60, 80, 28), np.ones((H, W)), MeasureConfig()) is None


def test_no_wall_near_the_outline_means_no_refinement():
    view = np.where(_disk(60, 80, 45), 0.2, 1.0)          # wall far outside: both plateaus are inside
    assert refine_outline(_disk(60, 80, 28), view, MeasureConfig(refine_band=0.15)) is None


def test_the_shift_is_limited_to_the_window():
    view = np.where(_disk(60, 80, 33), 0.2, 1.0)          # wall 5 px out, window +-0.1 R = 3 px
    out = refine_outline(_disk(60, 80, 28), view, MeasureConfig(refine_band=0.1))
    assert out is not None and 30.0 <= _radius(out.mask) <= 31.5


def test_a_second_pass_recentres_with_a_narrower_window():
    view = np.where(_disk(60, 80, 33), 0.2, 1.0)
    one = refine_outline(_disk(60, 80, 28), view, MeasureConfig(refine_band=0.2, refine_passes=1))
    two = refine_outline(_disk(60, 80, 28), view, MeasureConfig(refine_band=0.2, refine_passes=2))
    assert _radius(one.mask) == pytest.approx(33.0, abs=1.0)
    assert _radius(two.mask) == pytest.approx(33.0, abs=1.0)
