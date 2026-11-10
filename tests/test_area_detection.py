"""Regression tests for shape-independent dark-region measurements."""

import unittest
from pathlib import Path

import numpy as np

from bubbletrack.model.detection import (
    detect_bubble_with_mode, detect_dark_regions, measure_detection,
)


class AreaDetectionTests(unittest.TestCase):
    def setUp(self):
        self.binary = np.ones((100, 120), dtype=bool)
        self.binary[20:45, 20:55] = False
        self.binary[30:40, 55:65] = False  # irregular extension
        self.binary[25:28, 25:28] = True  # reflection inside the bubble
        self.binary[65:75, 80:95] = False  # second dark region
        self.binary[0:6, :] = False  # image border, not a bubble

    def test_largest_region_uses_area_not_circle_fit(self):
        mask, edge = detect_dark_regions(
            self.binary, 10, (11, 110), (21, 140), max_regions=1,
        )
        radius, rc, cc, count = measure_detection(
            mask, edge, "area", (11, 110), (21, 140),
        )
        self.assertEqual(count, 1)
        self.assertEqual(int(mask.sum()), 975)
        self.assertAlmostEqual(radius, np.sqrt(975 / np.pi))
        self.assertGreater(rc, 11)
        self.assertGreater(cc, 21)

    def test_multiple_regions_sum_areas(self):
        mask, edge = detect_dark_regions(
            self.binary, 10, (1, 100), (1, 120), max_regions=2,
        )
        radius, _, _, count = measure_detection(
            mask, edge, "area", (1, 100), (1, 120),
        )
        self.assertEqual(count, 2)
        self.assertEqual(int(mask.sum()), 1125)
        self.assertAlmostEqual(radius, np.sqrt(1125 / np.pi))

    def test_empty_mask_has_no_measurement(self):
        empty = np.zeros((10, 10), dtype=bool)
        radius, _, _, count = measure_detection(
            empty, np.empty((0, 2)), "area", (1, 10), (1, 10),
        )
        self.assertEqual((radius, count), (-1.0, 0))

    def test_grayscale_recovers_interior_when_adaptive_mask_is_only_a_ring(self):
        yy, xx = np.ogrid[:120, :120]
        distance_sq = (yy - 60) ** 2 + (xx - 60) ** 2
        disk = distance_sq <= 30 ** 2
        gray = np.full((120, 120), 0.7)
        gray[disk] = 0.1
        gray[distance_sq <= 8 ** 2] = 0.85
        gray[10:16, 10:16] = 0.1
        adaptive = np.ones((120, 120), dtype=bool)
        adaptive[(distance_sq >= 25 ** 2) & disk] = False

        mask, edge = detect_dark_regions(
            adaptive, 10000, (1, 120), (1, 120),
            grayscale_roi=gray, max_regions=2,
        )
        radius, _, _, count = measure_detection(
            mask, edge, "area", (1, 120), (1, 120),
        )
        self.assertEqual(count, 1)
        self.assertEqual(int(mask.sum()), int(disk.sum()))
        self.assertAlmostEqual(radius, np.sqrt(disk.sum() / np.pi))

    def test_area_mode_dispatch_uses_grayscale(self):
        gray = np.full((80, 80), 0.8)
        gray[20:60, 20:60] = 0.1
        adaptive = np.ones((80, 80), dtype=bool)
        adaptive[20:60, 20:25] = False
        mask, edge = detect_bubble_with_mode(
            adaptive, gray, "area", 0.5, [False] * 4,
            100, (1, 80), (1, 80),
        )
        radius, _, _, count = measure_detection(
            mask, edge, "area", (1, 80), (1, 80),
        )
        self.assertEqual(int(mask.sum()), 1600)
        self.assertEqual(count, 1)
        self.assertAlmostEqual(radius, np.sqrt(1600 / np.pi))

    def test_large_single_edge_region_beats_interior_speck(self):
        yy, xx = np.ogrid[:100, :140]
        disk = (yy - 70) ** 2 + (xx - 70) ** 2 <= 45 ** 2
        gray = np.full((100, 140), 0.8)
        gray[disk] = 0.1
        gray[62:70, 66:74] = 0.9
        gray[0:5, :] = 0.1  # broad border band must stay excluded
        gray[10:12, 10:12] = 0.1  # small interior speck
        adaptive = np.ones_like(gray, dtype=bool)

        mask, edge = detect_dark_regions(
            adaptive, 10000, (1, 100), (1, 140), grayscale_roi=gray,
        )
        radius, _, _, count = measure_detection(
            mask, edge, "area", (1, 100), (1, 140),
        )
        self.assertEqual(count, 1)
        self.assertEqual(int(mask.sum()), int(disk.sum()))
        self.assertTrue(mask[-1].any())
        self.assertAlmostEqual(radius, np.sqrt(disk.sum() / np.pi))

    def test_provided_frames_28_to_38(self):
        folder = Path(__file__).parent / "Jin_13_23_34"
        if not (folder / "Jin_13_23_34_035.tiff").exists():
            self.skipTest("optional TIFF sequence is not present")
        from bubbletrack.controller.worker import _process_single_frame

        radii = []
        for frame in range(28, 39):
            path = folder / f"Jin_13_23_34_{frame:03d}.tiff"
            _, radius, _, mask, rc, cc, count = _process_single_frame((
                frame - 1, str(path), 0.5, (1, 250), (1, 400),
                [False] * 4, 50, 90, 0, 0, 0, 0, "area", 0, 1,
                float("inf"),
            ))
            self.assertEqual(count, 1, frame)
            self.assertTrue(mask[-1].any(), frame)
            self.assertGreater(radius, 110, frame)
            self.assertLess(radius, 115, frame)
            self.assertLess(abs(rc - 141), 2, frame)
            self.assertLess(abs(cc - 157), 2, frame)
            radii.append(radius)
        self.assertLess(max(abs(np.diff(radii))), 1.0)


if __name__ == "__main__":
    unittest.main()
