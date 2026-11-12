"""ML assist measurement: probability map -> bubble region, its confidence, and the
precedence that decides the reported circle:

1. ML proposes the region, Auto measures the 50 % wall inside it ("auto_ml");
2. only if Auto cannot measure a clean wall there, ML's own outline ("ml", always reviewed);
3. if ML finds nothing either, no result.

Region, confidence and policy tests use synthetic probability maps and stub engine
results (no scikit-learn); the end-to-end test at the bottom needs scikit-learn.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from bubbletrack.model.edge import EdgeResult
from bubbletrack.model.teach import MeasureConfig, MLRegion, ml_region, select_result

H, W = 120, 160
GATE = 0.3


def _soft_disk(rc, cc, r, width=1.0, shape=(H, W)):
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    return (0.5 * (1.0 + np.tanh((r - np.hypot(yy - rc, xx - cc)) / width))).astype(np.float32)


def _auto(radius, rc, cc, conf):
    return EdgeResult(radius=radius, rc=rc, cc=cc, radius_sigma=0.1, edge_xy=np.empty((0, 2)),
                      confidence=conf, flags=frozenset(), r_area=radius, outer_radius=np.nan,
                      shoulder_width=np.nan, arc_coverage=1.0, inlier_fraction=1.0,
                      mask=np.zeros((H, W), bool))


def _region(radius, rc, cc, conf):
    return MLRegion(radius=radius, rc=rc, cc=cc, confidence=conf, mask=np.zeros((H, W), bool),
                    clipped=False, refined=False, terms={})


# --------------------------------------------------------------------------- #
#  Region from a probability map
# --------------------------------------------------------------------------- #

def test_sharp_disk_gives_its_radius_centre_and_a_high_confidence():
    reg = ml_region(_soft_disk(60, 80, 30), config=MeasureConfig(refine=False))
    assert reg.radius == pytest.approx(30.0, abs=0.8)
    assert (reg.rc, reg.cc) == pytest.approx((61.0, 81.0), abs=0.5)       # 1-based full image
    assert reg.confidence >= 0.7 and not reg.clipped
    assert reg.mask.shape == (H, W) and reg.mask[60, 80]


def test_origin_shifts_coordinates_only():
    reg = ml_region(_soft_disk(60, 80, 30), origin=(11, 21), config=MeasureConfig(refine=False))
    assert (reg.rc, reg.cc) == pytest.approx((71.0, 101.0), abs=0.5)


def test_seed_and_prior_choose_between_two_blobs():
    p = np.maximum(_soft_disk(60, 45, 25), _soft_disk(60, 125, 15))
    cfg = MeasureConfig(refine=False)
    assert ml_region(p, seed=(61.0, 126.0), config=cfg).radius == pytest.approx(15, abs=1)
    assert ml_region(p, prior=(61.0, 46.0, 25.0), config=cfg).radius == pytest.approx(25, abs=1)
    assert ml_region(p, config=cfg).radius == pytest.approx(25, abs=1)      # the bigger one


def test_noise_is_not_a_confident_bubble():
    p = np.random.default_rng(0).random((H, W)).astype(np.float32)
    reg = ml_region(p, config=MeasureConfig(refine=False))
    assert reg is None or reg.confidence < GATE


def test_empty_map_gives_no_region():
    assert ml_region(np.zeros((H, W), np.float32)) is None


def test_a_fuzzy_outline_is_less_confident_than_a_sharp_one():
    cfg = MeasureConfig(refine=False)
    sharp = ml_region(_soft_disk(60, 80, 30, width=1.0), config=cfg)
    fuzzy = ml_region(_soft_disk(60, 80, 30, width=12.0), config=cfg)
    assert fuzzy.confidence < sharp.confidence - 0.2


def test_region_cut_by_the_roi_border_is_completed_with_its_circle():
    reg = ml_region(_soft_disk(12, 80, 30), config=MeasureConfig(refine=False))
    assert reg.clipped
    assert reg.radius == pytest.approx(30.0, rel=0.04)
    assert reg.rc == pytest.approx(13.0, abs=1.5)


def test_refinement_moves_the_outline_to_the_half_contrast_wall():
    p = _soft_disk(60, 80, 28)                                   # the classifier's outline is 4 px inside
    view = 1.0 - 0.8 * _soft_disk(60, 80, 32, width=2.0)         # the wall (50 % crossing) is at 32
    assert ml_region(p, view, config=MeasureConfig(refine=True)).radius == pytest.approx(32, abs=1.0)
    assert ml_region(p, view, config=MeasureConfig(refine=False)).radius == pytest.approx(28, abs=1.0)


def test_refinement_is_skipped_without_contrast():
    p = _soft_disk(60, 80, 28)
    reg = ml_region(p, np.ones((H, W), np.float32), config=MeasureConfig(refine=True))
    assert reg.radius == pytest.approx(28, abs=1.0) and not reg.refined


# --------------------------------------------------------------------------- #
#  Precedence: Auto (ML-guided) -> ML outline -> nothing
# --------------------------------------------------------------------------- #

def test_auto_measures_the_wall_inside_the_ml_region():
    res = select_result(_region(30.5, 61.0, 81.0, 0.9), _auto(30.0, 61.2, 80.8, 0.8))
    assert res.source == "auto_ml" and res.radius == 30.0 and res.confidence == 0.8
    assert res.accepted and "ml_guided" in res.flags and not res.needs_review


def test_ml_outline_steps_in_where_auto_cannot_measure_a_clean_wall():
    region = _region(30.5, 61.0, 81.0, 0.8)
    for guided in (None, _auto(np.nan, np.nan, np.nan, 0.0), _auto(30.0, 61.0, 81.0, 0.1)):
        res = select_result(region, guided)
        assert res.source == "ml" and res.radius == 30.5 and res.accepted
        assert "ml_outline" in res.flags and res.needs_review


def test_auto_wall_on_another_object_is_not_used():
    res = select_result(_region(30.5, 61.0, 81.0, 0.8), _auto(12.0, 20.0, 20.0, 0.9))
    assert res.source == "ml" and res.radius == 30.5


def test_weak_ml_outline_is_not_accepted():
    res = select_result(_region(30.5, 61.0, 81.0, 0.1), None)
    assert not res.accepted and res.source == "ml" and res.radius == 30.5 and res.needs_review


def test_nothing_found_anywhere():
    res = select_result(None, None)
    assert not res.accepted and res.source == "none" and np.isnan(res.radius)
    res = select_result(None, _auto(30.0, 61.0, 81.0, 0.9))       # no ML region: Auto was never guided
    assert res.source == "none"


def test_single_source_policies():
    region = _region(30.5, 61.0, 81.0, 0.8)
    guided = _auto(29.0, 61.0, 81.0, 0.6)
    assert select_result(region, guided, config=MeasureConfig(policy="prior")).radius == 29.0
    assert select_result(region, guided, config=MeasureConfig(policy="contour")).radius == 30.5
    with pytest.raises(ValueError):
        MeasureConfig(policy="vote")


# --------------------------------------------------------------------------- #
#  End to end (scikit-learn)
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(importlib.util.find_spec("sklearn") is None, reason="scikit-learn not installed")
def test_measure_frame_end_to_end():
    from _edge_synth import Bubble, Scene, render
    from bubbletrack.model.edge import EdgeParams
    from bubbletrack.model.teach import MLExamples, measure_frame, train_model

    circles = [(60.0, 80.0, 30.0), (58.0, 84.0, 18.0), (62.0, 78.0, 24.0)]
    frames, ref = [], None
    for i, (rc, cc, r) in enumerate(circles):
        img, info = render(Scene(h=H, w=W, bubbles=(Bubble(rc=rc, cc=cc, r=r),), speckle=0.3, seed=20 + i))
        frames.append(img)
        ref = info["ref"] if ref is None else ref
    ex = MLExamples().verify(0, (61.0, 81.0, 30.0)).verify(1, (59.0, 85.0, 18.0))
    model = train_model(ex, frames.__getitem__, (1, H), (1, W), reference=ref)
    res = measure_frame(model, frames[2], params=EdgeParams(seed=(63.0, 79.0)))
    assert res.accepted and res.source in ("auto_ml", "ml")
    assert res.radius == pytest.approx(24.0, rel=0.05)
    assert (res.rc, res.cc) == pytest.approx((63.0, 79.0), abs=1.5)
    assert res.region is not None and res.guided is not None
