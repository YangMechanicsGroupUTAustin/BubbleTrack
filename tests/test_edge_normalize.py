"""Stage A of the Edge engine: grey conversion, saturation, background surface, normalisation."""

from __future__ import annotations

import numpy as np
import pytest

from bubbletrack.model.edge.config import NormalizeConfig
from bubbletrack.model.edge.normalize import (
    estimate_background,
    full_scale,
    noise_sigma,
    normalize_roi,
    saturation_mask,
    to_gray_frame,
)

CFG = NormalizeConfig()


def _disk(shape, rc, cc, r):
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    return np.hypot(yy - rc, xx - cc) < r


# ------------------------------------------------------------------ grey conversion

def test_uint16_keeps_full_bit_depth():
    img = np.array([[40000, 40001], [3, 65535]], np.uint16)
    g = to_gray_frame(img)
    assert g.values.dtype == np.float32
    np.testing.assert_array_equal(g.values, img.astype(np.float32))
    assert g.full_scale == 65535.0


@pytest.mark.parametrize("channels", [1, 2, 3, 4])
def test_multichannel_frames_become_grey(channels):
    base = np.full((5, 6), 1234, np.uint16)
    img = np.repeat(base[:, :, None], channels, axis=2)
    g = to_gray_frame(img)
    assert g.values.shape == (5, 6)
    np.testing.assert_allclose(g.values, 1234.0, atol=1.0)


def test_input_is_not_modified_and_not_aliased():
    img = np.full((4, 4), 7.0, np.float32)
    img.flags.writeable = False
    g = to_gray_frame(img)
    assert not np.shares_memory(g.values, img)
    g.values[0, 0] = 99.0
    assert img[0, 0] == 7.0


def test_non_finite_pixels_are_masked_and_filled():
    img = np.full((4, 4), 0.5, np.float64)
    img[1, 2] = np.nan
    img[3, 3] = np.inf
    g = to_gray_frame(img)
    assert g.non_finite.sum() == 2 and g.non_finite[1, 2] and g.non_finite[3, 3]
    assert np.all(np.isfinite(g.values)) and g.values[1, 2] == pytest.approx(0.5)
    assert np.isnan(img[1, 2])                       # caller's array untouched


@pytest.mark.parametrize("bad", [np.zeros((0, 3)), np.zeros((2, 2), bool), np.zeros((2, 2), complex),
                                 np.zeros((2, 2, 2, 2)), np.zeros((3, 3, 5))])
def test_unsupported_frames_raise_value_error(bad):
    with pytest.raises(ValueError):
        to_gray_frame(bad)


@pytest.mark.parametrize("img, expected", [
    (np.array([[0, 255]], np.uint8), 255.0),
    (np.array([[0, 60000]], np.uint16), 65535.0),
    (np.array([[0, 4000]], np.uint16), 4095.0),       # packed 12-bit camera data
    (np.array([[0, 900]], np.uint16), 1023.0),        # packed 10-bit
    (np.array([[0.0, 0.8]], np.float32), 1.0),
    (np.array([[0.0, 30000.0]], np.float64), 65535.0),
])
def test_full_scale(img, expected):
    assert full_scale(img) == expected


# ------------------------------------------------------------------ saturation

def test_saturated_blob_is_kept_and_speckle_grains_are_not():
    img = np.full((80, 80), 30000.0, np.float32)
    img[_disk(img.shape, 40, 40, 4)] = 65535.0           # ~50 px glare spot
    rng = np.random.default_rng(0)
    for r, c in rng.integers(0, 80, size=(40, 2)):
        if np.hypot(r - 40, c - 40) > 8:
            img[r, c] = 65535.0                          # isolated saturated speckle grains
    sat = saturation_mask(img, 65535.0, CFG)
    assert sat[40, 40]
    assert sat.sum() == _disk(img.shape, 40, 40, 4).sum()


def test_no_saturation_when_the_peak_is_far_below_full_scale():
    img = np.full((40, 40), 1000.0, np.float32)
    img[10:20, 10:20] = 2000.0
    assert not saturation_mask(img, 65535.0, CFG).any()


# ------------------------------------------------------------------ background and normalisation

def test_background_surface_flattens_vignette_and_ramp():
    h, w = 120, 200
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    illum = 30000 * (1 - 0.3 * (((yy - h / 2) / h) ** 2 + ((xx - w / 2) / w) ** 2)) * (1 + 0.2 * (xx - w / 2) / w)
    disk = _disk((h, w), 60, 100, 25)
    img = (illum * np.where(disk, 0.1, 1.0)).astype(np.float32)
    surface = estimate_background(img, np.zeros((h, w), bool), CFG)
    n = img / surface
    far = ~_disk((h, w), 60, 100, 35)
    assert np.median(np.abs(n[far] - 1.0)) < 0.01
    assert np.median(n[_disk((h, w), 60, 100, 20)]) < 0.2


def test_reference_normalisation_corrects_gain_and_marks_dead_pixels():
    h, w = 100, 160
    rng = np.random.default_rng(1)
    ref = (20000 + 500 * rng.standard_normal((h, w))).astype(np.float32)
    ref[:, :20] = 50.0                                   # black wall: no light in the reference
    frame = 1.2 * ref                                    # flicker: 20 % brighter
    disk = _disk((h, w), 50, 90, 20)
    frame = np.where(disk, 0.1 * frame, frame).astype(np.float32)
    out = normalize_roi(frame, np.zeros((h, w), bool), CFG, reference=ref)
    assert out.gain == pytest.approx(1.2, rel=0.02)
    assert out.invalid[:, :20].all()
    assert np.all(out.image[out.invalid] == 1.0)
    bg = ~out.invalid & ~_disk((h, w), 50, 90, 26)
    assert abs(np.median(out.image[bg]) - 1.0) < 0.01
    assert np.median(out.image[_disk((h, w), 50, 90, 15)]) < 0.2
    assert out.method.startswith("reference")


def test_reference_free_normalisation_reports_its_method():
    img = np.full((60, 80), 1000.0, np.float32)
    out = normalize_roi(img, np.zeros(img.shape, bool), CFG)
    assert out.method.startswith("surface") and out.gain == 1.0
    np.testing.assert_allclose(out.image, 1.0, atol=1e-3)


def test_noise_sigma_matches_white_noise():
    rng = np.random.default_rng(2)
    img = (1.0 + 0.02 * rng.standard_normal((200, 200))).astype(np.float32)
    assert noise_sigma(img, np.zeros(img.shape, bool), CFG) == pytest.approx(0.02, rel=0.1)
