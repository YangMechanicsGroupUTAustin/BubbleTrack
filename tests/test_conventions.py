"""Tests for model/conventions.py — index convention helpers."""

from bubbletrack.model.conventions import (
    clamp_roi,
    display_to_frame,
    frame_to_display,
    image_to_scene,
    roi_to_slice,
    scene_to_image,
)


def test_scene_and_image_coordinates_are_half_a_pixel_apart():
    # 1-based pixel k spans scene [k - 1, k): its centre is k in the image, k - 0.5 in the scene
    assert scene_to_image(9.5) == 10.0 and image_to_scene(10.0) == 9.5
    assert image_to_scene(scene_to_image(3.25)) == 3.25


class TestClampRoi:
    def test_clamp_roi_within_bounds(self):
        """Valid ROI is returned unchanged."""
        rx, ry = clamp_roi((10, 50), (20, 80), 100, 200)
        assert rx == (10, 50)
        assert ry == (20, 80)

    def test_clamp_roi_exceeds_dims(self):
        """ROI exceeding image dimensions is clamped to image size."""
        rx, ry = clamp_roi((10, 150), (20, 300), 100, 200)
        assert rx == (10, 100)
        assert ry == (20, 200)

    def test_clamp_roi_negative_values(self):
        """Negative or zero values are clamped to 1."""
        rx, ry = clamp_roi((-5, 50), (0, 80), 100, 200)
        assert rx == (1, 50)
        assert ry == (1, 80)

    def test_clamp_roi_swapped_min_max(self):
        """Swapped min/max values are corrected."""
        rx, ry = clamp_roi((50, 10), (80, 20), 100, 200)
        assert rx == (10, 50)
        assert ry == (20, 80)


class TestRoiToSlice:
    def test_basic_full_image(self):
        """(1, 100), (1, 200) maps to slice(0, 100), slice(0, 200)."""
        rs, cs = roi_to_slice((1, 100), (1, 200))
        assert rs == slice(0, 100)
        assert cs == slice(0, 200)

    def test_offset_region(self):
        """(50, 150), (30, 120) maps to slice(49, 150), slice(29, 120)."""
        rs, cs = roi_to_slice((50, 150), (30, 120))
        assert rs == slice(49, 150)
        assert cs == slice(29, 120)

    def test_single_pixel_roi(self):
        """(5, 5), (10, 10) maps to slice(4, 5), slice(9, 10) — one pixel."""
        rs, cs = roi_to_slice((5, 5), (10, 10))
        assert rs == slice(4, 5)
        assert cs == slice(9, 10)


class TestFrameDisplay:
    def test_frame_to_display_is_1_based(self):
        assert frame_to_display(0) == 1
        assert frame_to_display(99) == 100

    def test_display_to_frame_is_0_based(self):
        assert display_to_frame(1) == 0
        assert display_to_frame(100) == 99

    def test_roundtrip(self):
        """display_to_frame(frame_to_display(i)) == i for all i."""
        for i in range(10):
            assert display_to_frame(frame_to_display(i)) == i

    def test_reverse_roundtrip(self):
        """frame_to_display(display_to_frame(d)) == d for all d >= 1."""
        for d in range(1, 11):
            assert frame_to_display(display_to_frame(d)) == d
