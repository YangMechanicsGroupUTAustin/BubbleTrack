"""Tests for the frozen AppState dataclass and update_state helper."""

import dataclasses

import numpy as np
import pytest

from bubbletrack.model.state import AppState, update_state


class TestAppStateImmutability:
    """Verify AppState is frozen and update_state produces new instances."""

    def test_frozen_prevents_direct_assignment(self):
        s = AppState()
        with pytest.raises(dataclasses.FrozenInstanceError):
            s.image_no = 5

    def test_frozen_prevents_list_field_assignment(self):
        s = AppState()
        with pytest.raises(dataclasses.FrozenInstanceError):
            s.folder_path = "/new/path"

    def test_update_returns_new_instance(self):
        s1 = AppState()
        s2 = update_state(s1, image_no=5)
        assert s1 is not s2
        assert s1.image_no == 0
        assert s2.image_no == 5

    def test_update_preserves_other_fields(self):
        s1 = AppState(folder_path="/some/path", image_no=3)
        s2 = update_state(s1, img_thr=0.7)
        assert s2.img_thr == 0.7
        assert s2.folder_path == "/some/path"
        assert s2.image_no == 3

    def test_update_multiple_fields(self):
        s1 = AppState()
        s2 = update_state(s1, image_no=10, img_thr=0.3, folder_path="/data")
        assert s2.image_no == 10
        assert s2.img_thr == 0.3
        assert s2.folder_path == "/data"
        # Original unchanged
        assert s1.image_no == 0
        assert s1.img_thr == 0.5
        assert s1.folder_path == ""

    def test_update_with_no_changes_returns_new_instance(self):
        s1 = AppState()
        s2 = update_state(s1)
        assert s1 is not s2
        assert s1 == s2


class TestAppStateDefaults:
    """Verify default field values."""

    def test_default_images_is_empty_tuple(self):
        s = AppState()
        assert s.images == ()
        assert isinstance(s.images, tuple)

    def test_default_bubble_cross_edges_is_tuple(self):
        s = AppState()
        assert s.bubble_cross_edges == (False, False, False, False)
        assert isinstance(s.bubble_cross_edges, tuple)

    def test_default_numeric_values(self):
        s = AppState()
        assert s.image_no == 0
        assert s.img_thr == 0.5
        assert s.removing_obj_radius == 0
        assert s.gaussian_sigma == 0.0
        assert s.clahe_clip == 0.0
        assert s.closing_radius == 0
        assert s.opening_radius == 0

    def test_default_numpy_arrays_are_none(self):
        s = AppState()
        assert s.radius is None
        assert s.circle_fit_par is None
        assert s.circle_xy is None

    def test_default_display_state(self):
        s = AppState()
        assert s.cur_img is None
        assert s.cur_img_binary_roi is None


class TestTotalFrames:
    """Verify the total_frames computed property."""

    def test_total_frames_empty(self):
        s = AppState()
        assert s.total_frames == 0

    def test_total_frames_with_images(self):
        s = update_state(AppState(), images=("a.tif", "b.tif", "c.tif"))
        assert s.total_frames == 3


class TestWithResultsInitialized:
    """Verify with_results_initialized returns new state with allocated arrays."""

    def test_allocates_arrays(self):
        s1 = AppState()
        s2 = s1.with_results_initialized(10)
        assert s1 is not s2
        assert s1.radius is None  # original unchanged
        assert s2.radius is not None
        assert s2.radius.shape == (10,)
        assert np.all(s2.radius == -1.0)

    def test_circle_fit_par_shape(self):
        s = AppState().with_results_initialized(5)
        assert s.circle_fit_par.shape == (5, 2)
        assert np.all(np.isnan(s.circle_fit_par))

    def test_circle_xy_length(self):
        s = AppState().with_results_initialized(7)
        assert len(s.circle_xy) == 7
        assert all(x is None for x in s.circle_xy)

    def test_preserves_other_state(self):
        s1 = update_state(AppState(), folder_path="/data", image_no=3)
        s2 = s1.with_results_initialized(5)
        assert s2.folder_path == "/data"
        assert s2.image_no == 3


class TestNumpyArrayMutation:
    """Verify that NumPy array element-level mutation still works.

    This is the pragmatic compromise: frozen prevents field reassignment,
    but array contents remain mutable for performance.
    """

    def test_radius_element_update(self):
        s = AppState().with_results_initialized(5)
        s.radius[2] = 42.0
        assert s.radius[2] == 42.0

    def test_circle_fit_par_element_update(self):
        s = AppState().with_results_initialized(5)
        s.circle_fit_par[1] = [100.0, 200.0]
        assert s.circle_fit_par[1, 0] == 100.0
        assert s.circle_fit_par[1, 1] == 200.0

    def test_circle_xy_list_element_update(self):
        """circle_xy is a list (mutable container), element assignment works."""
        s = AppState().with_results_initialized(5)
        pts = np.array([[1.0, 2.0], [3.0, 4.0]])
        s.circle_xy[0] = pts
        assert np.array_equal(s.circle_xy[0], pts)
