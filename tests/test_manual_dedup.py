"""Tests for manual point deduplication logic."""

import numpy as np

from bubbletrack.controller.manual_controller import is_duplicate_point
from bubbletrack.model.constants import MIN_POINT_DISTANCE_PX


class TestPointDeduplication:
    def test_distant_points_accepted(self):
        """Two points > 5px apart should both be kept."""
        existing = [(10.0, 10.0)]
        new_pt = np.array([20.0, 20.0])
        assert not is_duplicate_point(new_pt, existing)

    def test_duplicate_point_rejected(self):
        """Same coordinates should be rejected."""
        existing = [(10.0, 10.0)]
        new_pt = np.array([10.0, 10.0])
        assert is_duplicate_point(new_pt, existing)

    def test_near_point_rejected(self):
        """Point within 5px of existing should be rejected."""
        existing = [(10.0, 10.0)]
        # Distance = sqrt(9 + 9) = ~4.24 < 5
        new_pt = np.array([13.0, 13.0])
        assert is_duplicate_point(new_pt, existing)

    def test_point_just_outside_threshold_accepted(self):
        """Point at exactly 5.01px should be accepted."""
        existing = [(10.0, 10.0)]
        # Place point at exactly 5.01px away along one axis
        new_pt = np.array([10.0, 15.01])
        assert not is_duplicate_point(new_pt, existing)

    def test_empty_list_always_accepts(self):
        """With no existing points, any point should be accepted."""
        new_pt = np.array([50.0, 50.0])
        assert not is_duplicate_point(new_pt, [])

    def test_near_any_existing_point(self):
        """Point near any one of multiple existing points should be rejected."""
        existing = [(10.0, 10.0), (50.0, 50.0), (100.0, 100.0)]
        # Close to the second point
        new_pt = np.array([52.0, 52.0])
        assert is_duplicate_point(new_pt, existing)

    def test_threshold_is_strict_less_than(self):
        """Point at exactly threshold distance should NOT be rejected (< not <=)."""
        existing = [(0.0, 0.0)]
        new_pt = np.array([MIN_POINT_DISTANCE_PX, 0.0])
        # Distance == MIN_POINT_DISTANCE_PX, condition is <, so not duplicate
        assert not is_duplicate_point(new_pt, existing)
