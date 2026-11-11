"""Bit depth, sample type and channel handling before adaptive thresholding.

One synthetic bubble frame (16-bit, as from the camera) is re-encoded as
12-bit and 10-bit data stored in uint16, as floats, and as RGB/RGBA.  Every
variant must detect the same bubble as the 16-bit original (within 1 px).
"""

from __future__ import annotations

import numpy as np
import pytest

from _golden_cases import SceneSpec, render_scene
from bubbletrack.model.detection import detect_bubble_with_mode, measure_detection
from bubbletrack.model.image_io import (
    ImageLoadError,
    normalize_frame,
    to_grayscale,
    to_threshold_uint8,
)
from bubbletrack.model.removing_factor import compute_removing_factor

TRUE_RADIUS = 60.0
FRAME16 = render_scene(SceneSpec(radius=TRUE_RADIUS, center=(125.0, 200.0), seed=21))


def _radius(raw: np.ndarray, threshold: float, sigma: float) -> float:
    """Run the Fit All model pipeline (intensity mode) on an in-memory frame."""
    h, w = raw.shape[:2]
    gridx, gridy = (1, h), (1, w)
    _, _, gray_roi, binary_roi = normalize_frame(
        raw, threshold, gridx, gridy, gaussian_sigma=sigma,
    )
    rf = compute_removing_factor(85.0, gridx, gridy)
    processed, edge_xy = detect_bubble_with_mode(
        binary_roi, gray_roi, "intensity", threshold, [False] * 4, rf, gridx, gridy,
        gaussian_sigma=sigma,
    )
    radius, _, _, _ = measure_detection(processed, edge_xy, "intensity", gridx, gridy)
    return radius


def _alpha_like(img: np.ndarray) -> np.ndarray:
    return np.full(img.shape, np.iinfo(img.dtype).max, img.dtype)


_FRAME8 = np.clip(np.round(FRAME16 / 257.0), 0, 255).astype(np.uint8)

VARIANTS = {
    "12bit_in_uint16": (FRAME16 >> 4).astype(np.uint16),
    "10bit_in_uint16": (FRAME16 >> 6).astype(np.uint16),
    "float32_unit_range": (FRAME16 / 65535.0).astype(np.float32),
    "float64_raw_counts": FRAME16.astype(np.float64),
    "rgb16": np.dstack([FRAME16] * 3),
    "rgba16": np.dstack([FRAME16] * 3 + [_alpha_like(FRAME16)]),
    "rgb8": np.dstack([_FRAME8] * 3),
    "rgba8": np.dstack([_FRAME8] * 3 + [_alpha_like(_FRAME8)]),
    "gray_alpha8": np.dstack([_FRAME8, _alpha_like(_FRAME8)]),
    "single_channel_3d": FRAME16[:, :, np.newaxis],
}


# Typical settings: no blur, and the owner's everyday threshold + blur.
@pytest.mark.parametrize(("threshold", "sigma"), [(0.55, 0.0), (0.58, 3.0)])
@pytest.mark.parametrize("name", list(VARIANTS))
def test_variant_detects_same_radius_as_16bit(name, threshold, sigma):
    reference = _radius(FRAME16, threshold, sigma)
    assert reference == pytest.approx(TRUE_RADIUS, abs=2.0), "16-bit reference must work"
    radius = _radius(VARIANTS[name], threshold, sigma)
    assert radius == pytest.approx(reference, abs=1.0), name


class TestToThresholdUint8:
    def test_16bit_data_keeps_the_divide_by_256_path(self):
        raw = np.array([[0, 255, 4096, 30000, 65535]], np.uint16)
        np.testing.assert_array_equal(to_threshold_uint8(raw), (raw / 256).astype(np.uint8))

    def test_12bit_data_is_scaled_by_16(self):
        raw = np.array([[0, 16, 2000, 4095]], np.uint16)
        np.testing.assert_array_equal(to_threshold_uint8(raw), [[0, 1, 125, 255]])

    def test_10bit_data_is_scaled_by_4(self):
        raw = np.array([[0, 4, 500, 1023]], np.uint16)
        np.testing.assert_array_equal(to_threshold_uint8(raw), [[0, 1, 125, 255]])

    def test_uint8_is_unchanged(self):
        raw = np.array([[0, 7, 255]], np.uint8)
        np.testing.assert_array_equal(to_threshold_uint8(raw), raw)

    def test_unit_range_float_is_scaled_by_255(self):
        raw = np.array([[0.0, 0.5, 1.0]], np.float32)
        np.testing.assert_array_equal(to_threshold_uint8(raw), [[0, 127, 255]])

    @pytest.mark.parametrize("shift", [0, 4, 6])  # 16-, 12-, 10-bit counts
    def test_float_camera_counts_match_the_uint16_path(self, shift):
        counts = (FRAME16 >> shift).astype(np.uint16)
        for dtype in (np.float32, np.float64, np.int32, np.uint32):
            np.testing.assert_array_equal(
                to_threshold_uint8(counts.astype(dtype)), to_threshold_uint8(counts),
            )

    def test_negative_float_range_is_min_max_scaled(self):
        raw = np.array([[-2.0, 0.0, 2.0]], np.float64)
        np.testing.assert_array_equal(to_threshold_uint8(raw), [[0, 127, 255]])

    def test_float_range_beyond_uint16_is_min_max_scaled(self):
        raw = np.array([[0.0, 5e4, 1e5]], np.float32)
        np.testing.assert_array_equal(to_threshold_uint8(raw), [[0, 127, 255]])

    def test_non_finite_floats_do_not_poison_the_frame(self):
        raw = np.array([[np.nan, 0.0, 0.5, 1.0, np.inf, -np.inf]], np.float32)
        np.testing.assert_array_equal(to_threshold_uint8(raw), [[0, 0, 127, 255, 255, 0]])

    def test_all_non_finite_is_zero(self):
        out = to_threshold_uint8(np.full((2, 2), np.nan))
        assert out.dtype == np.uint8
        assert not out.any()

    def test_signed_integers_are_min_max_scaled(self):
        raw = np.array([[-100, 0, 100]], np.int16)
        np.testing.assert_array_equal(to_threshold_uint8(raw), [[0, 127, 255]])

    def test_does_not_mutate_input(self):
        raw = np.array([[np.nan, 1.0]], np.float64)
        to_threshold_uint8(raw)
        assert np.isnan(raw[0, 0])


class TestToGrayscale:
    def test_2d_is_returned_as_is(self):
        assert to_grayscale(FRAME16) is FRAME16

    @pytest.mark.parametrize("channels", [3, 4])
    def test_colour_8_and_16_bit(self, channels):
        for img in (_FRAME8, FRAME16):
            stacked = np.dstack([img] * channels)
            gray = to_grayscale(stacked)
            assert gray.shape == img.shape
            assert gray.dtype == img.dtype
            np.testing.assert_allclose(gray, img, atol=1)

    def test_float64_colour(self):
        img = FRAME16.astype(np.float64)
        np.testing.assert_allclose(to_grayscale(np.dstack([img] * 3)), img, rtol=1e-6)

    def test_unsupported_layout_raises(self):
        with pytest.raises(ImageLoadError):
            to_grayscale(np.zeros((4, 4, 5), np.uint8))
        with pytest.raises(ImageLoadError):
            to_grayscale(np.zeros((2, 4, 4, 3), np.uint8))


def test_normalised_image_ignores_nan(tmp_path):
    raw = FRAME16.astype(np.float32)
    raw[0, 0] = np.nan
    cur_img, binary, _, _ = normalize_frame(raw, 0.5, (1, 250), (1, 400))
    assert np.isfinite(cur_img).all()
    assert binary.any() and not binary.all()
