"""Stage C of the Edge engine: radial sub-pixel edges and the outer extent."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage

from bubbletrack.model.edge.config import RayConfig
from bubbletrack.model.edge.rays import (
    STATUS_CLIPPED,
    STATUS_EDGE,
    _row_percentile,
    outer_extent,
    ray_edges,
)

CFG = RayConfig()
SHAPE = (200, 300)


def _profile_image(levels, shape=SHAPE, centre=(100.3, 150.6), blur=1.0, ss=4):
    """Radially stepped normalised image: ``levels`` = ((r_outer, value), ...) from the inside out."""
    h, w = shape
    yy = (np.arange(h * ss) + 0.5) / ss - 0.5
    xx = (np.arange(w * ss) + 0.5) / ss - 0.5
    d = np.hypot(*np.meshgrid(yy - centre[0], xx - centre[1], indexing="ij"))
    img = np.ones_like(d)
    for r_out, value in reversed(levels):
        img[d < r_out] = value
    img = img.reshape(h, ss, w, ss).mean(axis=(1, 3))
    return ndimage.gaussian_filter(img, blur).astype(np.float32)


def _radii(pts, centre=(100.3, 150.6)):
    return np.hypot(pts[:, 0] - centre[0], pts[:, 1] - centre[1])


def test_edges_of_a_clean_disk_are_sub_pixel_accurate():
    img = _profile_image(((40.0, 0.1),))
    out = ray_edges(img, (101.0, 150.0, 37.0), 0.9, 0.01, CFG)
    assert out.n_rays == round(2 * np.pi * 37)
    assert np.all(out.status == STATUS_EDGE)
    assert np.median(_radii(out.points)) == pytest.approx(40.0, abs=0.1)
    assert out.multi_fraction == 0.0 and out.n_clipped == 0


def test_rays_leaving_the_roi_are_marked_clipped():
    img = _profile_image(((40.0, 0.1),), centre=(30.0, 150.6))      # top of the bubble outside
    out = ray_edges(img, (30.0, 150.6, 40.0), 0.9, 0.01, CFG)
    assert out.n_clipped > 0 and out.n_clipped == int(np.sum(out.status == STATUS_CLIPPED))
    assert np.median(_radii(out.points, (30.0, 150.6))) == pytest.approx(40.0, abs=0.15)
    assert out.points[:, 0].min() >= 0


def test_invalid_pixels_stop_the_rays():
    img = _profile_image(((40.0, 0.1),))
    invalid = np.zeros(SHAPE, bool)
    invalid[:, 175:] = True                                         # the right side has no data
    out = ray_edges(img, (100.3, 150.6, 40.0), 0.9, 0.01, CFG, invalid=invalid)
    assert out.n_clipped > 0
    assert out.points[:, 1].max() < 175


def test_edge_choice_outer_and_inner_on_a_double_ring():
    # dark core (0.1) to r=30, a mid-grey ring (0.45) to r=40, then the background
    img = _profile_image(((30.0, 0.1), (40.0, 0.45)))
    outer = ray_edges(img, (100.3, 150.6, 35.0), 0.9, 0.01, CFG, edge_choice="outer")
    inner = ray_edges(img, (100.3, 150.6, 35.0), 0.9, 0.01, CFG, edge_choice="inner")
    assert np.median(_radii(outer.points)) == pytest.approx(40.0, abs=0.3)
    assert np.median(_radii(inner.points)) == pytest.approx(30.0, abs=0.3)
    assert inner.multi_fraction > 0.5          # both steps are seen from the core
    assert outer.multi_fraction == 0.0         # the inner step does not straddle the mid level


def test_faint_halo_step_is_not_taken_for_the_wall():
    # bubble to r=40 (0.07), faint halo (0.6) to r=46: only the steep step straddles the mid level
    img = _profile_image(((40.0, 0.07), (46.0, 0.6)))
    out = ray_edges(img, (100.3, 150.6, 42.0), 0.93, 0.01, CFG)
    assert np.median(_radii(out.points)) == pytest.approx(40.0, abs=0.3)


def test_outer_extent_measures_the_shoulder():
    img = _profile_image(((40.0, 0.07), (46.0, 0.6)))
    pts = outer_extent(img, np.zeros(SHAPE, bool), (100.3, 150.6, 40.0), CFG)
    assert len(pts) > 300
    assert np.median(_radii(pts)) == pytest.approx(46.5, abs=1.0)


def test_outer_extent_of_a_clean_edge_is_close_to_the_steep_edge():
    img = _profile_image(((40.0, 0.1),))
    pts = outer_extent(img, np.zeros(SHAPE, bool), (100.3, 150.6, 40.0), CFG)
    assert 0.8 < np.median(_radii(pts)) - 40.0 < 3.0


def test_outer_extent_needs_the_background_in_view():
    img = _profile_image(((40.0, 0.1),))
    exclude = np.zeros(SHAPE, bool)
    exclude[np.hypot(*np.mgrid[0:200, 0:300] - np.array([100.3, 150.6])[:, None, None]) > 45] = True
    assert len(outer_extent(img, exclude, (100.3, 150.6, 40.0), CFG)) == 0


def test_row_percentile_matches_numpy():
    rng = np.random.default_rng(0)
    p = rng.standard_normal((50, 40))
    p[rng.random(p.shape) < 0.3] = np.nan
    p[0, :] = np.nan
    cols = np.arange(5, 30)
    got, rows = _row_percentile(p, cols, 20.0, 4)
    want = np.array([np.nanpercentile(r[cols], 20.0) if np.isfinite(r[cols]).sum() >= 4 else np.nan
                     for r in p])
    np.testing.assert_allclose(got, want, equal_nan=True)
    assert not rows[0]
