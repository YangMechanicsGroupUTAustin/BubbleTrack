"""Stage D of the Edge engine: robust circle fitting and arc statistics."""

from __future__ import annotations

import numpy as np
import pytest

from bubbletrack.model.edge.config import FitConfig
from bubbletrack.model.edge.fit import arc_stats, fit_circle, polygon_area, ransac_circle

CFG = FitConfig()


def _circle(n, rc=50.0, cc=80.0, r=30.0, a0=0.0, a1=2 * np.pi, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    th = np.linspace(a0, a1, n, endpoint=False)
    rr = r + noise * rng.standard_normal(n)
    return np.column_stack([rc + rr * np.sin(th), cc + rr * np.cos(th)])


def test_exact_points_give_the_exact_circle():
    fit = fit_circle(_circle(200), np.random.default_rng(1), CFG)
    assert fit.circle == pytest.approx((50.0, 80.0, 30.0), abs=1e-6)
    assert fit.method == "ransac/taubin+tukey+lsq"
    assert np.all(fit.weights > CFG.inlier_weight)


def test_outliers_are_rejected():
    pts = _circle(300, noise=0.2)
    rng = np.random.default_rng(3)
    bad = np.column_stack([rng.uniform(10, 90, 120), rng.uniform(40, 120, 120)])  # 29 % clutter
    fit = fit_circle(np.vstack([pts, bad]), np.random.default_rng(1), CFG)
    assert fit.radius == pytest.approx(30.0, abs=0.1)
    assert (fit.rc, fit.cc) == pytest.approx((50.0, 80.0), abs=0.1)
    inl = fit.weights > CFG.inlier_weight
    assert inl[:300].mean() > 0.95 and inl[300:].mean() < 0.3
    assert 0 < fit.radius_sigma < 0.05


def test_partial_arc_is_fitted():
    fit = fit_circle(_circle(120, a0=0.2, a1=2.2, noise=0.1), np.random.default_rng(1), CFG)
    assert fit.radius == pytest.approx(30.0, abs=0.5)


def test_few_points_use_plain_taubin_and_too_few_give_none():
    fit = fit_circle(_circle(6), np.random.default_rng(1), CFG)
    assert fit.method == "taubin" and fit.radius == pytest.approx(30.0, abs=1e-6)
    assert fit_circle(_circle(2), np.random.default_rng(1), CFG) is None
    assert fit_circle(_circle(50), np.random.default_rng(1), CFG, robust=False).method == "taubin"


def test_fit_is_deterministic_for_a_given_seed():
    pts = np.vstack([_circle(100, noise=0.5), np.random.default_rng(5).uniform(0, 100, (40, 2))])
    a = fit_circle(pts, np.random.default_rng(12345), CFG)
    b = fit_circle(pts, np.random.default_rng(12345), CFG)
    assert a.circle == b.circle
    np.testing.assert_array_equal(a.weights, b.weights)


def test_ransac_finds_the_dominant_circle():
    pts = np.vstack([_circle(100), _circle(30, rc=10.0, cc=10.0, r=5.0)])
    rc, cc, r = ransac_circle(pts, np.random.default_rng(0), CFG)
    assert (rc, cc, r) == pytest.approx((50.0, 80.0, 30.0), abs=0.5)


def test_arc_stats_full_and_half_circle():
    cov, gap = arc_stats(_circle(360), 50.0, 80.0)
    assert cov == 1.0 and gap == pytest.approx(1.0)
    cov, gap = arc_stats(_circle(180, a1=np.pi), 50.0, 80.0)
    assert cov == pytest.approx(0.5) and gap == pytest.approx(181.0, abs=0.5)
    assert arc_stats(np.empty((0, 2)), 0.0, 0.0) == (0.0, 360.0)


def test_polygon_area_of_a_dense_circle():
    assert polygon_area(_circle(720), 50.0, 80.0) == pytest.approx(np.pi * 30.0 ** 2, rel=1e-3)
    assert polygon_area(_circle(2), 50.0, 80.0) == 0.0
