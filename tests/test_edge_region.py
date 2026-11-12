"""Stage G of the Edge engine: the bubble as a region (iso-contour), completion, necks, fragments."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage

from bubbletrack.model.edge.config import RegionConfig
from bubbletrack.model.edge.geometry import (
    disk_box_moments,
    hull_area,
    outward_normals,
    polygon_moments,
    vertex_lengths,
)
from bubbletrack.model.edge.lobes import split_necks
from bubbletrack.model.edge.body import filled_components
from bubbletrack.model.edge.region import find_region, trace

CFG = RegionConfig()
SHAPE = (200, 300)
YY, XX = np.mgrid[0:SHAPE[0], 0:SHAPE[1]].astype(float)


def _coverage(rc, cc, r, ss=4):
    """Exact-ish area coverage of each pixel by a disk (4x super-sampling)."""
    yy = (np.arange(SHAPE[0] * ss) + 0.5) / ss - 0.5
    xx = (np.arange(SHAPE[1] * ss) + 0.5) / ss - 0.5
    d2 = (yy[:, None] - rc) ** 2 + (xx[None, :] - cc) ** 2
    return (d2 <= r * r).reshape(SHAPE[0], ss, SHAPE[1], ss).mean(axis=(1, 3))


def _image(cover, depth=0.9, blur=1.0):
    """Normalised image: background 1, a dark body of the given coverage."""
    return ndimage.gaussian_filter(1.0 - depth * cover, blur).astype(np.float32)


def _ring(n=720, rc=100.0, cc=150.0, r=40.0):
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([rc + r * np.sin(th), cc + r * np.cos(th)])


# ------------------------------------------------------------------ geometry

def test_polygon_moments_of_a_dense_circle():
    m = polygon_moments(_ring())
    assert m.area == pytest.approx(np.pi * 40 ** 2, rel=1e-4)
    assert m.centroid == pytest.approx((100.0, 150.0), abs=1e-9)
    assert polygon_moments(_ring()[::-1]).area == pytest.approx(m.area)
    assert vertex_lengths(_ring()).sum() == pytest.approx(2 * np.pi * 40, rel=1e-4)
    assert hull_area(_ring()) == pytest.approx(m.area, rel=1e-4)


@pytest.mark.parametrize("reverse", [False, True])
def test_outward_normals_point_away_from_the_interior(reverse):
    p = _ring()[::-1] if reverse else _ring()
    n = outward_normals(p)
    radial = (p - [100.0, 150.0]) / 40.0
    assert np.min(np.sum(n * radial, axis=1)) > 0.999


@pytest.mark.parametrize("box, frac", [
    ((-0.5, 1e4, -0.5, 1e4), 1.0),            # whole disk
    ((50.3, 1e4, -0.5, 1e4), 0.5),            # cut through the centre row
    ((50.3, 1e4, 80.7, 1e4), 0.25),           # a quadrant
])
def test_disk_box_moments_are_exact(box, frac):
    m = disk_box_moments((50.3, 80.7, 30.0), box, CFG.disk_nodes)
    assert m.area == pytest.approx(frac * np.pi * 900, rel=1e-6)


# ------------------------------------------------------------------ topology

def test_filled_components_fill_enclosed_holes_only():
    dark = (np.hypot(YY - 100, XX - 150) < 40) & (np.hypot(YY - 100, XX - 150) >= 10)   # ring
    dark |= (np.hypot(YY - 2, XX - 40) < 20) & (np.hypot(YY - 0, XX - 40) >= 6)         # open notch
    _, lab = filled_components(dark)
    assert lab[100, 150] > 0 and lab[100, 150] == lab[100, 175]
    assert lab[0, 40] == 0


def test_two_touching_disks_are_split_and_the_anchor_keeps_its_lobe():
    body = (np.hypot(YY - 100, XX - 100) <= 45) | (np.hypot(YY - 100, XX - 170) <= 28)
    big, n_big = split_necks(body, (100.0, 100.0), CFG)
    small, n_small = split_necks(body, (100.0, 175.0), CFG)
    assert n_big == n_small == 2
    assert big.sum() == pytest.approx(np.pi * 45 ** 2, rel=0.02)
    assert small.sum() == pytest.approx(np.pi * 28 ** 2, rel=0.05)
    assert not (big & small).any()


@pytest.mark.parametrize("name, body", [
    ("peanut", (np.hypot(YY - 100, XX - 116) <= 42) | (np.hypot(YY - 100, XX - 184) <= 42)),
    ("disk", np.hypot(YY - 100, XX - 150) <= 50),
    ("ellipse", ((YY - 100) / 30) ** 2 + ((XX - 150) / 70) ** 2 <= 1),
])
def test_a_wide_neck_or_none_keeps_the_body_whole(name, body):
    main, n = split_necks(body, (100.0, 150.0), CFG)
    assert n == 1 and (main == body).all(), name


# ------------------------------------------------------------------ tracing and completion

def test_trace_encloses_the_exact_area_of_a_blurred_disk():
    img = _image(_coverage(100.3, 150.6, 40.0))
    body = img < 0.55
    c = trace(img, 0.55, body, CFG.crop_margin_px)
    assert polygon_moments(c).area == pytest.approx(np.pi * 40 ** 2, rel=0.002)
    assert polygon_moments(c).centroid == pytest.approx((100.3, 150.6), abs=0.02)


def test_bright_centre_spot_and_inner_ring_are_filled():
    cover = _coverage(100.3, 150.6, 40.0)
    img = 1.0 - 0.9 * cover
    d = np.hypot(YY - 100.3, XX - 150.6)
    img = np.where(d < 9, 2.0, img)                        # saturated centre spot
    img = np.where(np.abs(d - 30) < 2, 0.9, img)           # bright inner ring
    img = ndimage.gaussian_filter(img, 1.0).astype(np.float32)
    reg = find_region(img, 0.55, np.zeros(SHAPE, bool), (100.0, 130.0), None, CFG, search_px=10,
                      preferred=[("seed", (100.0, 150.0))])
    assert reg.radius == pytest.approx(40.0, rel=0.002)
    assert reg.mask[100, 150] and not reg.clipped and reg.fragment_count == 0


@pytest.mark.parametrize("rc, cc", [(10.0, 150.0), (-15.0, 150.0), (20.0, 25.0)],
                         ids=["top_25%", "top_69%", "corner"])
def test_clipped_disk_is_completed_with_its_circle(rc, cc):
    img = _image(_coverage(rc, cc, 40.0))
    reg = find_region(img, 0.55, np.zeros(SHAPE, bool), (max(rc, 0) + 10, cc),
                      (rc, cc, 40.0), CFG, search_px=20)
    assert reg.clipped and reg.completed and 0 < reg.hidden_fraction < 1
    assert reg.radius == pytest.approx(40.0, rel=0.002)
    assert reg.centroid == pytest.approx((rc, cc), abs=0.1)
    assert reg.edge[:, 0].min() > -0.5 and reg.solidity > 0.99
    assert reg.circularity == pytest.approx(1.0, abs=0.02)


def test_excluded_pixels_inside_the_disk_are_completed_too():
    img = _image(_coverage(100.0, 150.0, 40.0))
    exclude = (YY > 95) & (YY < 120) & (XX > 175) & (XX < 200)       # dead patch over the right edge
    img = np.where(exclude, 1.0, img).astype(np.float32)
    reg = find_region(img, 0.55, exclude, (100.0, 140.0), (100.0, 150.0, 40.0), CFG,
                      search_px=10)
    assert reg.clipped and reg.completed
    assert reg.radius == pytest.approx(40.0, rel=0.004)


def test_fragments_near_the_main_body_are_counted():
    cover = np.clip(_coverage(100, 120, 35) + _coverage(100, 175, 6) + _coverage(40, 120, 5)
                    + _coverage(190, 290, 5), 0, 1)                   # last one is beyond 2 R
    img = _image(cover)
    reg = find_region(img, 0.55, np.zeros(SHAPE, bool), (100.0, 120.0), None, CFG, search_px=10)
    assert reg.radius == pytest.approx(35.0, rel=0.003)
    assert reg.fragment_count == 2
    assert reg.fragment_area == pytest.approx(np.pi * (36 + 25), rel=0.25)


def test_no_region_at_the_anchors_gives_none():
    img = _image(_coverage(100, 60, 20))
    assert find_region(img, 0.55, np.zeros(SHAPE, bool), (100.0, 250.0), None, CFG,
                       search_px=10) is None


def _ring_bubble(rc, cc, r, core):
    """A dark wall of radius r around a bright core of radius *core* (LIC 01-like), plus a dark island."""
    wall = np.clip(_coverage(rc, cc, r) - _coverage(rc, cc, core), 0, 1)
    return np.clip(wall + _coverage(rc + 3, cc, 4), 0, 1)


def test_bright_core_open_to_the_border_is_filled_when_the_circle_confirms_it():
    img = _image(_ring_bubble(180.0, 150.0, 60.0, 35.0))           # the core reaches the bottom border
    circle = (180.0, 150.0, 60.0)
    reg = find_region(img, 0.55, np.zeros(SHAPE, bool), (140.0, 150.0), circle, CFG, search_px=10,
                      preferred=[("seed", (183.0, 150.0))])          # click on the island in the core
    assert reg.anchor == "seed" and reg.completed
    assert reg.radius == pytest.approx(60.0, rel=0.003)
    bare = find_region(img, 0.55, np.zeros(SHAPE, bool), (140.0, 150.0), None, CFG, search_px=10)
    assert bare.area < 0.75 * np.pi * 60 ** 2                     # no circle: the open core stays out


def test_interior_closed_by_excluded_pixels_is_filled():
    img = _image(_ring_bubble(100.0, 150.0, 40.0, 25.0))
    exclude = (YY > 120) & (YY < 145) & (np.abs(XX - 150) < 6)    # glare cuts the wall at the bottom
    img = np.where(exclude, 1.0, img).astype(np.float32)
    reg = find_region(img, 0.55, exclude, (100.0, 118.0), (100.0, 150.0, 40.0), CFG, search_px=10)
    assert reg.mask[100, 150] and reg.clipped
    assert reg.radius == pytest.approx(40.0, rel=0.005)
