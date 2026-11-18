"""Tests for bubble detection pipeline."""

import numpy as np

from bubbletrack.model.detection import (
    crop_expanded_image,
    detect_bubble,
    expand_binary_image,
)


def _make_disk_image(h, w, cr, cc, r):
    """Create a bool image with a dark disk (False) on white (True) background."""
    yy, xx = np.ogrid[:h, :w]
    mask = (yy - cr) ** 2 + (xx - cc) ** 2 <= r ** 2
    img = np.ones((h, w), dtype=bool)
    img[mask] = False
    return img


class TestExpandCrop:
    def test_expand_top(self):
        bw = np.zeros((10, 20), dtype=bool)
        expanded = expand_binary_image(bw, [False, True, True, True])
        assert expanded.shape == (20, 20)  # doubled rows
        # Original in bottom half
        np.testing.assert_array_equal(expanded[10:, :], bw)
        # Top half is ones
        assert expanded[:10, :].all()

    def test_expand_right(self):
        bw = np.zeros((10, 20), dtype=bool)
        expanded = expand_binary_image(bw, [True, False, True, True])
        assert expanded.shape == (10, 40)
        np.testing.assert_array_equal(expanded[:, :20], bw)

    def test_expand_down(self):
        bw = np.zeros((10, 20), dtype=bool)
        expanded = expand_binary_image(bw, [True, True, False, True])
        assert expanded.shape == (20, 20)
        np.testing.assert_array_equal(expanded[:10, :], bw)

    def test_expand_left(self):
        bw = np.zeros((10, 20), dtype=bool)
        expanded = expand_binary_image(bw, [True, True, True, False])
        assert expanded.shape == (10, 40)
        np.testing.assert_array_equal(expanded[:, 20:], bw)

    def test_roundtrip(self):
        bw = np.random.default_rng(0).integers(0, 2, (30, 40)).astype(bool)
        edges = [True, False, True, True]
        expanded = expand_binary_image(bw, edges)
        cropped = crop_expanded_image(expanded, 30, 40, edges)
        np.testing.assert_array_equal(cropped, bw)


class TestDetectBubble:
    def test_centered_disk(self):
        """A centred dark disk on white should be detected."""
        img = _make_disk_image(200, 200, 100, 100, 40)
        edges = [False, False, False, False]
        processed, edge_xy = detect_bubble(img, edges, 10, (1, 200), (1, 200))
        assert processed.shape == (200, 200)
        assert edge_xy.shape[0] > 10  # should have boundary points
        assert edge_xy.shape[1] == 2

    def test_edge_coords_in_full_image_space(self):
        img = _make_disk_image(100, 100, 50, 50, 20)
        # ROI starts at row=51, col=51 in full image
        _, edge_xy = detect_bubble(img, [False]*4, 10, (51, 150), (51, 150))
        # All coords should be offset by gridx/gridy start
        assert edge_xy[:, 0].min() >= 51
        assert edge_xy[:, 1].min() >= 51

    def test_small_noise_removed(self):
        """Small speckles should be removed by removing_factor."""
        img = np.ones((100, 100), dtype=bool)
        # Add a small speckle (5 pixels)
        img[10:12, 10:13] = False
        # Add a real bubble
        yy, xx = np.ogrid[:100, :100]
        img[(yy - 50)**2 + (xx - 50)**2 <= 20**2] = False
        _, edge_xy = detect_bubble(img, [False]*4, 50, (1, 100), (1, 100))
        # Should detect the large bubble, not the speckle
        assert edge_xy.shape[0] > 0

    def test_empty_returns_empty(self):
        """All-white image should return no edges."""
        img = np.ones((50, 50), dtype=bool)
        processed, edge_xy = detect_bubble(img, [False]*4, 10, (1, 50), (1, 50))
        assert edge_xy.shape[0] == 0

    def test_opening_removes_spurs(self):
        """Morphological opening should remove thin protrusions."""
        img = _make_disk_image(200, 200, 100, 100, 40)
        # Add a thin spur (1px wide) on the bubble edge
        img[60, 95:106] = False  # horizontal spur
        edges = [False] * 4
        _, edge_no_open = detect_bubble(img, edges, 10, (1, 200), (1, 200))
        _, edge_open = detect_bubble(
            img, edges, 10, (1, 200), (1, 200), opening_radius=3,
        )
        # Opening should produce a cleaner (fewer edge points) result
        assert edge_open.shape[0] <= edge_no_open.shape[0]

    def test_closing_smooths_boundary(self):
        """Morphological closing should smooth boundary irregularities."""
        # Bubble crosses top edge so it's not fully enclosed after expansion
        img = _make_disk_image(200, 200, 0, 100, 40)
        # Add jagged notches on the bubble boundary
        img[30:35, 95:100] = True
        img[25:30, 105:110] = True
        edges = [True, False, False, False]
        proc_no_close, e1 = detect_bubble(img, edges, 10, (1, 200), (1, 200))
        proc_close, e2 = detect_bubble(
            img, edges, 10, (1, 200), (1, 200), closing_radius=5,
        )
        # Both should detect the bubble
        assert e1.shape[0] > 0
        assert e2.shape[0] > 0

    def test_filters_backward_compatible(self):
        """Default filter params should not change output."""
        img = _make_disk_image(200, 200, 100, 100, 40)
        edges = [False] * 4
        p1, e1 = detect_bubble(img, edges, 10, (1, 200), (1, 200))
        p2, e2 = detect_bubble(
            img, edges, 10, (1, 200), (1, 200),
            opening_radius=0, closing_radius=0,
        )
        np.testing.assert_array_equal(p1, p2)
        np.testing.assert_array_equal(e1, e2)
