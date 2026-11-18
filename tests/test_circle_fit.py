"""Tests for the Taubin circle fitting algorithm."""

import numpy as np

from bubbletrack.model.circle_fit import circle_fit_taubin


def _make_circle_points(rc, cc, r, n=100, noise=0.0):
    """Generate *n* points on a circle with optional noise."""
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    rows = rc + r * np.cos(theta) + noise * np.random.randn(n)
    cols = cc + r * np.sin(theta) + noise * np.random.randn(n)
    return np.column_stack([rows, cols])


class TestCircleFitTaubin:
    def test_perfect_circle(self):
        xy = _make_circle_points(100, 150, 50)
        rc, cc, radius = circle_fit_taubin(xy)
        assert abs(rc - 100) < 0.01
        assert abs(cc - 150) < 0.01
        assert abs(radius - 50) < 0.01

    def test_noisy_circle(self):
        rng = np.random.default_rng(42)
        theta = np.linspace(0, 2 * np.pi, 200, endpoint=False)
        rows = 200 + 80 * np.cos(theta) + rng.normal(0, 0.5, 200)
        cols = 300 + 80 * np.sin(theta) + rng.normal(0, 0.5, 200)
        xy = np.column_stack([rows, cols])
        rc, cc, radius = circle_fit_taubin(xy)
        assert abs(rc - 200) < 1.0
        assert abs(cc - 300) < 1.0
        assert abs(radius - 80) < 1.0

    def test_small_circle(self):
        xy = _make_circle_points(10, 10, 5, n=50)
        rc, cc, radius = circle_fit_taubin(xy)
        assert abs(rc - 10) < 0.1
        assert abs(cc - 10) < 0.1
        assert abs(radius - 5) < 0.1

    def test_large_circle(self):
        xy = _make_circle_points(500, 500, 400, n=500)
        rc, cc, radius = circle_fit_taubin(xy)
        assert abs(rc - 500) < 0.1
        assert abs(cc - 500) < 0.1
        assert abs(radius - 400) < 0.1

    def test_fewer_than_3_points(self):
        xy = np.array([[1.0, 2.0], [3.0, 4.0]])
        rc, cc, radius = circle_fit_taubin(xy)
        assert np.isnan(rc)
        assert np.isnan(cc)
        assert np.isnan(radius)

    def test_single_point(self):
        xy = np.array([[5.0, 5.0]])
        rc, cc, radius = circle_fit_taubin(xy)
        assert np.isnan(rc)

    def test_exactly_3_points(self):
        # 3 points on a circle of radius 10 centred at (0, 0)
        xy = np.array([[10.0, 0.0], [-5.0, 8.66025], [-5.0, -8.66025]])
        rc, cc, radius = circle_fit_taubin(xy)
        assert abs(rc) < 0.5
        assert abs(cc) < 0.5
        assert abs(radius - 10) < 0.5

    def test_partial_arc(self):
        # Only a 90-degree arc
        theta = np.linspace(0, np.pi / 2, 50)
        xy = np.column_stack([60 + 30 * np.cos(theta), 60 + 30 * np.sin(theta)])
        rc, cc, radius = circle_fit_taubin(xy)
        assert abs(rc - 60) < 1.0
        assert abs(cc - 60) < 1.0
        assert abs(radius - 30) < 1.0

    def test_error_less_than_0_1_pct(self):
        """Verify < 0.1% error for a perfect circle."""
        xy = _make_circle_points(256, 256, 128, n=360)
        rc, cc, radius = circle_fit_taubin(xy)
        assert abs(radius - 128) / 128 < 0.001
        assert abs(rc - 256) / 256 < 0.001
        assert abs(cc - 256) / 256 < 0.001
