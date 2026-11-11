"""Stages B and E of the Edge engine: coarse mask, hole filling, candidate selection."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage

from bubbletrack.model.edge.config import FitConfig, NormalizeConfig, SegmentConfig
from bubbletrack.model.edge.segment import (
    boundary_points,
    coarse_segment,
    fill_holes_border_aware,
    select_candidate,
)

SEG, NORM, FIT = SegmentConfig(), NormalizeConfig(), FitConfig()
SHAPE = (160, 260)


def _coverage(shape, rc, cc, r):
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    return np.clip(r - np.hypot(yy - rc, xx - cc) + 0.5, 0.0, 1.0)


def _scene(disks, noise=0.01, seed=0, shape=SHAPE):
    """Normalised image (background 1) with dark disks ``(rc, cc, r, depth)``."""
    n = np.ones(shape)
    for rc, cc, r, depth in disks:
        n -= depth * _coverage(shape, rc, cc, r)
    n = ndimage.gaussian_filter(n, 1.0)
    n += noise * np.random.default_rng(seed).standard_normal(shape)
    return n.astype(np.float32)


def _segment(img, exclude=None):
    exclude = np.zeros(img.shape, bool) if exclude is None else exclude
    return coarse_segment(img, exclude, SEG, NORM)


def test_coarse_mask_finds_a_disk():
    coarse = _segment(_scene([(80, 130, 30, 0.9)]))
    truth = _coverage(SHAPE, 80, 130, 30) > 0.5
    assert coarse.contrast == pytest.approx(0.9, abs=0.05)
    assert (coarse.mask ^ truth).sum() < 0.05 * truth.sum()
    assert coarse.mask.dtype == bool and coarse.mask.shape == SHAPE


def test_texture_raises_the_smoothing_scale():
    clean = _segment(_scene([(80, 130, 30, 0.5)], noise=0.005))
    noisy = _segment(_scene([(80, 130, 30, 0.5)], noise=0.3))
    assert noisy.smooth_sigma > clean.smooth_sigma
    assert noisy.noise <= noisy.contrast * SEG.texture_target


def test_hole_open_towards_the_border_is_filled():
    # a bubble clipped by the top edge whose bright centre spot is open towards that edge
    yy, xx = np.mgrid[0:40, 0:60]
    disk = (np.hypot(yy - 5, xx - 30) < 20) & (np.hypot(yy - 2, xx - 30) >= 6)
    filled = fill_holes_border_aware(disk, (True, False, False, False))
    assert filled[2, 30] and filled[0, 30]
    assert filled.sum() == (np.hypot(yy - 5, xx - 30) < 20).sum()
    assert not fill_holes_border_aware(disk, (False, False, False, False))[2, 30]


def test_wall_band_does_not_swallow_the_background():
    band = np.zeros((40, 60), bool)
    band[:, :6] = True                            # vertical wall touching top, bottom and left
    filled = fill_holes_border_aware(band, (True, True, True, False))
    assert filled.sum() == band.sum()


def test_boundary_points_skip_roi_border_pixels():
    crop = np.zeros((10, 10), bool)
    crop[0:6, 2:8] = True                         # touches the top border
    pts = boundary_points(crop, 100, 200, (True, False, False, False))
    assert len(pts) > 0 and pts[:, 0].min() > 100      # nothing on row 0 of the crop


def test_score_prefers_the_round_dark_bubble():
    img = _scene([(80, 80, 25, 0.9), (80, 200, 12, 0.9)])
    sel = select_candidate(_segment(img), np.zeros(SHAPE, bool), SEG, FIT)
    assert sel.best.circle[2] == pytest.approx(25, abs=1.5)
    assert len(sel.candidates) == 2 and 0 < sel.runner_up < 1
    assert sel.method == "score"


def test_seed_selects_the_clicked_bubble():
    img = _scene([(80, 80, 25, 0.9), (80, 200, 12, 0.9)])
    sel = select_candidate(_segment(img), np.zeros(SHAPE, bool), SEG, FIT, seed=(82.0, 198.0))
    assert sel.best.circle[2] == pytest.approx(12, abs=1.0)
    assert sel.best.seed_distance == 0.0 and sel.runner_up == 0.0
    assert sel.method == "seed"


def test_seed_outside_every_component_takes_the_nearest():
    img = _scene([(80, 80, 25, 0.9), (80, 200, 12, 0.9)])
    sel = select_candidate(_segment(img), np.zeros(SHAPE, bool), SEG, FIT, seed=(80.0, 170.0))
    assert sel.best.circle[1] == pytest.approx(200, abs=1.0)
    assert 0 < sel.runner_up < 1


def test_prior_favours_the_expected_circle():
    img = _scene([(80, 80, 25, 0.9), (80, 200, 20, 0.9)])
    sel = select_candidate(_segment(img), np.zeros(SHAPE, bool), SEG, FIT, prior=(80.0, 200.0, 20.0))
    assert sel.best.circle[1] == pytest.approx(200, abs=1.0)
    assert sel.method == "score+prior"


def test_dark_aperture_corners_fail_the_polarity_test():
    yy, xx = np.mgrid[0:SHAPE[0], 0:SHAPE[1]]
    outside = np.hypot(yy - 80, xx - 130) > 95     # circular field of view: black corners
    img = _scene([(80, 130, 15, 0.9)])
    img[outside] = 0.05
    sel = select_candidate(_segment(img), np.zeros(SHAPE, bool), SEG, FIT)
    assert sel.best.circle[2] == pytest.approx(15, abs=1.0)
    assert all(c.inside_fraction >= SEG.polarity_min_inside for c in sel.candidates)


def test_no_dark_component_gives_no_candidate():
    img = np.ones(SHAPE, np.float32)
    sel = select_candidate(_segment(img), np.zeros(SHAPE, bool), SEG, FIT)
    assert sel.best is None and sel.candidates == ()
