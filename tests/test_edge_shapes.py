"""Bubbles of any shape: the area-equivalent radius of the outer iso-contour (owner decision 1).

Shapes from ``_edge_shapes`` (exact area truth): circle, shallow bite, deep bite (kidney),
jet slot, peanut, 5-lobe, crescent, mushroom with a re-entrant underside (rays cross the
boundary 3+ times), and a disk with a saturated centre spot and a bright inner ring.

Measured (edge-1.1.0, with reference; same with and without a seed click):

=============  ======  ======  ========  ====  ==================
shape          truth   R       r_circle  conf  flags
=============  ======  ======  ========  ====  ==================
circle         60.00   59.98   59.99     1.00  -
shallow_bite   59.01   59.00   59.97     1.00  -
kidney         54.42   54.43   60.00     1.00  non_circular
jet_slot       57.94   57.92   59.99     1.00  non_circular
peanut         57.95   57.90   41.97     1.00  non_circular
five_lobe      55.88   55.70   45.08     1.00  non_circular
crescent       39.81   39.83   60.05     1.00  non_circular
mushroom       46.89   47.04   50.01     1.00  non_circular
spot_and_ring  60.00   59.98   59.99     0.98  saturation
=============  ======  ======  ========  ====  ==================

Before the region layer: kidney 59.9 (confidence 0.99, no flag), jet slot 59.9, crescent
60.1, mushroom 50.0, peanut and 5-lobe not found (``no_candidate``).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pytest

from _edge_shapes import H, SHAPES, W, Shape, reference, render_shape
from bubbletrack.model.edge import EdgeParams, detect_edge
from bubbletrack.model.edge.geometry import polygon_moments

FULL = ((1, H), (1, W))
ROUND = {"circle", "spot_and_ring"}


@lru_cache(maxsize=None)
def _frame(shape: Shape):
    img, r_true = render_shape(shape)
    img.flags.writeable = False
    return img, r_true


@lru_cache(maxsize=1)
def _ref():
    ref = reference()
    ref.flags.writeable = False
    return ref


def _measure(shape: Shape, *, seeded: bool):
    img, r_true = _frame(shape)
    seed = (shape.seed[0] + 1, shape.seed[1] + 1) if seeded else None
    return detect_edge(img, *FULL, reference=_ref(), params=EdgeParams(seed=seed)), r_true


@pytest.mark.parametrize("seeded", [True, False], ids=["seed", "no_seed"])
@pytest.mark.parametrize("shape", SHAPES, ids=[s.name for s in SHAPES])
def test_area_radius_of_every_shape_is_within_1_percent(shape, seeded):
    res, r_true = _measure(shape, seeded=seeded)
    assert res.radius == pytest.approx(r_true, rel=0.01), (res.flags, res.provenance.get("failure"))
    assert res.r_area == res.radius and res.radius_kind == "area"
    assert res.accepted and res.confidence >= 0.9          # a clean outline is confident, whatever its shape


@pytest.mark.parametrize("shape", SHAPES, ids=[s.name for s in SHAPES])
def test_shape_descriptors_and_the_non_circular_flag(shape):
    res, _ = _measure(shape, seeded=True)
    assert res.area_px == pytest.approx(np.pi * res.radius ** 2)
    assert 0 < res.solidity <= 1 and 0 < res.circularity <= 1.01
    if shape.name in ROUND:
        assert "non_circular" not in res.flags
        assert res.r_circle == pytest.approx(res.radius, abs=0.3)
        assert res.solidity > 0.99 and res.circularity == pytest.approx(1.0, abs=0.03)
    elif shape.name != "shallow_bite":
        assert "non_circular" in res.flags


@pytest.mark.parametrize("shape", SHAPES, ids=[s.name for s in SHAPES])
def test_outline_mask_and_centre_describe_the_region(shape):
    res, _ = _measure(shape, seeded=True)
    contour0 = res.contour - 1.0                            # back to 0-based pixel centres
    assert polygon_moments(contour0).area == pytest.approx(res.area_px, rel=1e-9)
    assert res.mask.shape == (H, W) and res.mask.sum() == pytest.approx(res.area_px, rel=0.01)
    assert len(res.edge_xy) == len(res.contour)             # unclipped: the whole outline is edge
    assert (res.rc, res.cc) == pytest.approx(res.centroid, abs=1e-9)
    sr, sc = int(shape.seed[0]), int(shape.seed[1])
    assert res.mask[sr, sc]


@pytest.mark.parametrize("name", ["kidney", "peanut"])
def test_outer_definition_measures_the_same_region_at_the_outer_level(name):
    shape = next(s for s in SHAPES if s.name == name)
    img, _ = _frame(shape)
    steep = detect_edge(img, *FULL, reference=_ref())
    outer = detect_edge(img, *FULL, reference=_ref(), params=EdgeParams(edge_definition="outer"))
    assert outer.accepted and outer.radius_kind == "area" and "non_circular" in outer.flags
    assert 0.5 < outer.radius - steep.radius < 3.0          # the blurred edge's outer extent
    assert outer.shoulder_width == pytest.approx(outer.radius - steep.radius)
