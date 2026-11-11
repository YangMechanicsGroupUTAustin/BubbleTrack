"""Tests for model/constants.py — verify all constants are importable and sane."""

from bubbletrack.model.constants import (
    AUTO_DISPLAY_THROTTLE_MS,
    DEFAULT_FPS,
    DEFAULT_REMOVING_FACTOR,
    DEFAULT_RMAX_FIT_LENGTH,
    DEFAULT_UM2PX,
    DISPLAY_DEBOUNCE_MS,
    MAX_AXIS_RATIO,
    MAX_ECCENTRICITY,
    MIN_HOLE_AREA,
    PREVIEW_DEBOUNCE_MS,
    RF_MAX_AREA_FRACTION,
    RF_MIN_AREA,
    RMAX_OUTLIER_NEIGHBOURS,
    RMAX_OUTLIER_TOLERANCE,
    TAUBIN_EPSILON,
    TAUBIN_MAX_ITER,
    TAUBIN_MIN_POINTS,
)


class TestDetectionConstants:
    def test_max_axis_ratio_positive(self):
        assert MAX_AXIS_RATIO > 0

    def test_max_eccentricity_positive(self):
        assert MAX_ECCENTRICITY > 0

    def test_min_hole_area_positive(self):
        assert MIN_HOLE_AREA > 0

    def test_max_eccentricity_greater_than_one(self):
        """Eccentricity ratio must exceed 1 (circle has ratio 1)."""
        assert MAX_ECCENTRICITY > 1.0


class TestRemovingFactorConstants:
    def test_rf_min_area_positive(self):
        assert RF_MIN_AREA > 0

    def test_rf_max_area_fraction_in_range(self):
        assert 0 < RF_MAX_AREA_FRACTION <= 1.0


class TestTaubinConstants:
    def test_epsilon_positive_and_tiny(self):
        assert 0 < TAUBIN_EPSILON < 1e-6

    def test_max_iter_positive(self):
        assert TAUBIN_MAX_ITER > 0

    def test_min_points_at_least_three(self):
        assert TAUBIN_MIN_POINTS >= 3


class TestExportDefaults:
    def test_fps_positive(self):
        assert DEFAULT_FPS > 0

    def test_um2px_positive(self):
        assert DEFAULT_UM2PX > 0

    def test_rmax_fit_length_positive_and_odd(self):
        assert DEFAULT_RMAX_FIT_LENGTH > 0
        assert DEFAULT_RMAX_FIT_LENGTH % 2 == 1

    def test_rmax_outlier_tolerance_is_a_fraction(self):
        assert 0 < RMAX_OUTLIER_TOLERANCE < 1

    def test_rmax_outlier_neighbours_positive(self):
        assert RMAX_OUTLIER_NEIGHBOURS >= 1


class TestStateDefaults:
    def test_removing_factor_in_slider_range(self):
        assert 0 <= DEFAULT_REMOVING_FACTOR <= 100


class TestUIConstants:
    def test_debounce_values_positive(self):
        assert DISPLAY_DEBOUNCE_MS > 0
        assert PREVIEW_DEBOUNCE_MS > 0
        assert AUTO_DISPLAY_THROTTLE_MS > 0

    def test_throttle_greater_than_debounce(self):
        """Throttle for batch mode should be >= debounce for interactive mode."""
        assert AUTO_DISPLAY_THROTTLE_MS >= DISPLAY_DEBOUNCE_MS
