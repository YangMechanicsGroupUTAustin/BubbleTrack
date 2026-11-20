"""Stage A of the Edge engine: grey conversion, saturation, background surface, normalisation."""

from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np
import pytest

from bubbletrack.model.edge import EdgeParams, detect_edge
from bubbletrack.model.edge.config import DEFAULT_CONFIG, NormalizeConfig
from bubbletrack.model.edge.engine import detect_edge_with_config
from bubbletrack.model.edge.normalize import (
    background_level,
    bright_level,
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


def _speckle(shape, grain, rng):
    """Fully developed speckle (exponential intensities, median 1), grain ~ *grain* px."""
    from scipy.ndimage import gaussian_filter

    re = gaussian_filter(rng.standard_normal(shape), grain)
    im = gaussian_filter(rng.standard_normal(shape), grain)
    inten = re ** 2 + im ** 2
    return inten / np.median(inten)


def test_background_surface_keeps_a_big_dark_bubble_out_of_dense_speckle():
    # edge-1.5.0: in dense speckle the residual spread is so large that the later fits used to
    # re-admit a big dark bubble as "background"; the surface then ran through it and its inside
    # normalised to the speckle level (lic_hard/02 frames 6-41, owner decision 2026-11-15)
    h, w = 250, 400
    img = 20000 * _speckle((h, w), 1.2, np.random.default_rng(0))
    bubble = _disk((h, w), 125, 200, 110)
    img = np.where(bubble, 0.03 * img.mean(), img).astype(np.float32)
    n = img / estimate_background(img, np.zeros((h, w), bool), CFG)
    inside = np.median(n[_disk((h, w), 125, 200, 88)])
    outside = np.median(n[~_disk((h, w), 125, 200, 120)])
    assert inside < 0.2 * outside


def test_reference_normalisation_corrects_gain_and_keeps_a_dark_wall():
    # edge-1.4.0: a black wall in the reference is divided like every other pixel (nothing
    # is excluded); the same wall in the frame then normalises to the background
    h, w = 100, 160
    rng = np.random.default_rng(1)
    ref = (20000 + 500 * rng.standard_normal((h, w))).astype(np.float32)
    ref[:, :20] = 50.0                                   # black wall: almost no light in the reference
    frame = 1.2 * ref                                    # flicker: 20 % brighter
    disk = _disk((h, w), 50, 90, 20)
    frame = np.where(disk, 0.1 * frame, frame).astype(np.float32)
    out = normalize_roi(frame, np.zeros((h, w), bool), CFG, reference=ref)
    assert out.gain == pytest.approx(1.2, rel=0.02)
    assert not out.invalid.any()
    assert np.all(np.isfinite(out.image))
    assert np.median(out.image[:, :20]) == pytest.approx(1.0, abs=0.05)
    bg = ~_disk((h, w), 50, 90, 26)
    assert abs(np.median(out.image[bg]) - 1.0) < 0.01
    assert np.median(out.image[_disk((h, w), 50, 90, 15)]) < 0.2
    assert out.method.startswith("reference")


# ------------------------------------------------------------------ background level (edge-1.6.0)

def test_bright_level_ignores_a_dark_majority():
    rng = np.random.default_rng(3)
    y = np.concatenate([0.10 + 0.01 * rng.standard_normal(8000),     # a uniform dark bubble, 80 %
                        1.00 + 0.05 * rng.standard_normal(2000)])    # the background, 20 %
    assert np.median(y) < 0.2                                       # the median is inside the bubble
    assert bright_level(y, CFG) == pytest.approx(1.0, abs=0.02)
    level, shadowed = background_level(y, CFG)
    assert shadowed and level == pytest.approx(1.0, abs=0.02)


def test_background_level_is_exactly_the_median_without_a_shadow():
    rng = np.random.default_rng(4)
    y = np.concatenate([0.10 + 0.01 * rng.standard_normal(3000),     # a bubble, 30 %
                        1.00 + 0.05 * rng.standard_normal(7000)])
    assert background_level(y, CFG) == (float(np.median(y)), False)


def test_gain_is_exactly_the_median_ratio_when_the_bubble_is_small():
    h, w = 100, 160
    rng = np.random.default_rng(5)
    ref = (20000 * _speckle((h, w), 1.2, rng)).astype(np.float32)
    frame = np.where(_disk((h, w), 50, 80, 25), 0.05, 0.8) * ref
    out = normalize_roi(frame.astype(np.float32), np.zeros((h, w), bool), CFG, reference=ref)
    ratio = frame.astype(np.float32) / np.maximum(cv2.GaussianBlur(ref, (0, 0), CFG.ref_blur_sigma), 1e-6)
    assert out.gain == float(np.median(ratio))                       # unchanged since edge-1.4.0
    assert "bright-side" not in out.method


def _big_bubble_frame(seed: int, *, moved: float):
    """Dense speckle seen through a dark bubble (transmission 0.08) covering ~61 % of a 120 x 160
    ROI and cut by its bottom border; the frame is 0.8 x as bright as the reference and *moved*
    of its speckle is decorrelated from the reference's."""
    h, w = 120, 160
    rng = np.random.default_rng(seed)
    ref_tex = _speckle((h, w), 1.2, rng)
    tex = (1 - moved) * ref_tex + moved * _speckle((h, w), 1.2, rng)
    disk = _disk((h, w), 66, 80, 62)
    frame = 0.8 * 20000 * tex * np.where(disk, 0.08, 1.0)
    return frame.astype(np.float32), (20000 * ref_tex).astype(np.float32), disk


def test_reference_normalisation_stays_out_of_a_bubble_filling_most_of_the_roi():
    # edge-1.6.0: the median ratio lay inside the bubble (gain ~0.09 instead of ~0.8), so the
    # residual surface took the bubble in and lifted its inside towards the background
    frame, ref, disk = _big_bubble_frame(0, moved=0.0)
    assert disk.mean() > 0.6
    out = normalize_roi(frame, np.zeros(frame.shape, bool), CFG, reference=ref)
    assert "bright-side" in out.method and out.gain > 0.6
    assert np.median(out.image[_disk(frame.shape, 66, 80, 55)]) < 0.15
    assert np.median(out.image[~_disk(frame.shape, 66, 80, 68)]) == pytest.approx(1.0, abs=0.1)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_a_bubble_filling_most_of_the_roi_is_measured(seed):
    # lic_hard/02 files 12-35 (a bubble of R ~ 110-124 px filling most of a 250 x 400 frame, cut
    # by its border, in moving dense speckle) gave no result up to edge-1.5.1
    frame, ref, _ = _big_bubble_frame(seed, moved=0.3)
    res = detect_edge(frame, (1, 120), (1, 160), reference=ref, params=EdgeParams())
    assert res.accepted, (res.confidence, res.provenance.get("failure"))
    assert res.radius == pytest.approx(62.0, abs=1.5)
    assert np.hypot(res.rc - 67.0, res.cc - 81.0) < 1.5             # 1-based centre
    old = replace(DEFAULT_CONFIG, normalize=replace(CFG, level_shadow_median=0.0))  # the median, always
    before = detect_edge_with_config(frame, (1, 120), (1, 160), reference=ref, params=EdgeParams(), config=old)
    assert not before.accepted


def _bubble_beside_a_bright_band(seed: int, *, band: bool):
    """A large dark bubble (R 50) in a 120 x 200 ROI of moving speckle (30 % decorrelated from the
    reference); with *band*, columns 0-34 of the frame are 3 x as bright as in the reference (an
    air interface beside the bubble, lic_hard/02)."""
    h, w = 120, 200
    rng = np.random.default_rng(seed)
    ref_tex = _speckle((h, w), 1.2, rng)
    tex = 0.7 * ref_tex + 0.3 * _speckle((h, w), 1.2, rng)
    light = np.ones((h, w))
    if band:
        light[:, :35] = 3.0
    frame = 0.8 * 20000 * tex * np.where(_disk((h, w), 60, 110, 50), 0.08, 1.0) * light
    return frame.astype(np.float32), (20000 * ref_tex).astype(np.float32)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_a_bright_band_beside_a_large_bubble_keeps_its_snr(seed):
    # edge-1.6.1: the band (2-4 x the background in lic_hard/02) read as noise over the ROI and
    # cost the snr term ~30 %; the noise is now also measured just outside the wall
    results = {}
    for band in (False, True):
        frame, ref = _bubble_beside_a_bright_band(seed, band=band)
        results[band] = detect_edge(frame, (1, 120), (1, 200), reference=ref, params=EdgeParams())
    plain, banded = results[False], results[True]
    assert banded.provenance["noise"] > 1.1 * plain.provenance["noise"]       # the ROI texture is inflated
    snr = banded.provenance["confidence_terms"]["snr"]
    assert snr >= 0.85 and snr >= plain.provenance["confidence_terms"]["snr"] - 0.15
    assert banded.accepted and banded.radius == pytest.approx(50.0, abs=1.5)
    ring_off = replace(DEFAULT_CONFIG, segment=replace(DEFAULT_CONFIG.segment, wall_texture_min_pixels=10 ** 9))
    frame, ref = _bubble_beside_a_bright_band(seed, band=True)
    before = detect_edge_with_config(frame, (1, 120), (1, 200), reference=ref, params=EdgeParams(), config=ring_off)
    assert before.provenance["confidence_terms"]["snr"] < snr - 0.15         # edge-1.6.0's ROI texture only


def test_only_non_finite_pixels_are_invalid():
    h, w = 60, 80
    ref = np.full((h, w), 20000.0, np.float32)
    ref[:, :10] = 0.0                                    # no light at all: the divisor is floored
    ref[40:50, 50:70] = 65535.0                          # a clipped block
    frame = ref.copy()
    invalid = np.zeros((h, w), bool)
    invalid[5:8, 30:35] = True                           # e.g. NaN in the frame
    out = normalize_roi(frame, np.zeros((h, w), bool), CFG, reference=ref, invalid=invalid)
    np.testing.assert_array_equal(out.invalid, invalid)
    assert np.all(np.isfinite(out.image))
    assert np.all(out.image[invalid] == 1.0)
    assert np.all(out.image[:, :5] == 0.0)               # 0 / 0 reads 0 (no light in either), not NaN
    assert out.gain == pytest.approx(1.0, rel=1e-3)
    assert np.median(out.image[42:48, 52:68]) == pytest.approx(1.0, abs=0.02)  # clipped block: real ratio
    assert np.median(out.image[:, 15:]) == pytest.approx(1.0, abs=0.01)


def test_reference_free_normalisation_reports_its_method():
    img = np.full((60, 80), 1000.0, np.float32)
    out = normalize_roi(img, np.zeros(img.shape, bool), CFG)
    assert out.method.startswith("surface") and out.gain == 1.0
    np.testing.assert_allclose(out.image, 1.0, atol=1e-3)


def test_roi_without_any_background_pixel_is_noted():
    img = np.full((30, 40), 1000.0, np.float32)
    everything = np.ones(img.shape, bool)
    for ref in (None, img):
        out = normalize_roi(img, everything, CFG, reference=ref)
        assert "no usable background" in out.method
        assert np.all(np.isfinite(out.image))
    assert "no usable background" not in normalize_roi(img, ~everything, CFG).method


def test_noise_sigma_matches_white_noise():
    rng = np.random.default_rng(2)
    img = (1.0 + 0.02 * rng.standard_normal((200, 200))).astype(np.float32)
    assert noise_sigma(img, np.zeros(img.shape, bool), CFG) == pytest.approx(0.02, rel=0.1)
