"""No reference pixel is excluded (edge-1.4.0, owner decision 2026-11-14).

Until edge-1.3.0 stage A marked the reference's dark pixels (blurred reference below 0.12 x its
median) and saturated blobs, widened by 5 px, as invalid: the frame / reference image held a
flat 1.0 there and later stages ignored them.  Now a dark band or a clipped block of the
reference excludes nothing: the image holds the real, finite ratio there, and a bubble seen
through the band is measured whole.  Only non-finite (NaN / inf) pixels of the frame or the
reference have no value.  Pixels the old rule never touched give bit-identical results.
"""

from __future__ import annotations

import dataclasses

import cv2
import numpy as np
import pytest

from _golden_cases import SceneSpec, render_scene
from bubbletrack.model.edge import engine
from bubbletrack.model.edge import normalize as norm_mod
from bubbletrack.model.edge.config import NormalizeConfig
from bubbletrack.model.edge.engine import EdgeParams, EdgeResult, detect_edge, engine_view
from bubbletrack.model.edge.normalize import Normalized, normalize_frame_roi, saturation_mask, to_gray_frame

CFG = NormalizeConfig()
SHAPE = (250, 400)
CENTRE = (125.0, 200.0)
FULL_ROI = (slice(0, SHAPE[0]), slice(0, SHAPE[1]))
WALL_COLS = slice(0, 40)                   # a dark wall at the left (5 % light)
BAND_ROWS = slice(110, 140)                # a dark band through the bubble (8 % light)
CLIP = (slice(20, 50), slice(300, 340))    # clipped in the reference only
OLD_DARK_LEVEL, OLD_DILATE_PX = 0.12, 5    # the removed rule's constants (frozen copy below)


def _scene(radius: float, seed: int, *, wall=False, band=False, clip=False) -> np.ndarray:
    """A uint16 frame.  Wall and band attenuate the light (reference and frames alike);
    *clip* saturates a block (used in the reference: the frame there is not clipped)."""
    img = render_scene(SceneSpec(radius=radius, center=CENTRE, shape=SHAPE, seed=seed)).astype(np.float64)
    if wall:
        img[:, WALL_COLS] *= 0.05
    if band:
        img[BAND_ROWS, :] *= 0.08
    if clip:
        img[CLIP] = 65535.0
    return np.clip(np.round(img), 0, 65535).astype(np.uint16)


def _stage_a(raw, ref_raw, *, dark=True):
    return normalize_frame_roi(to_gray_frame(raw), to_gray_frame(ref_raw), FULL_ROI, dark=dark, cfg=CFG)


def _old_rule(ref_raw) -> np.ndarray:
    """The pixels the removed rule excluded (to check the test scenes exercise it)."""
    ref = to_gray_frame(ref_raw)
    blurred = cv2.GaussianBlur(ref.values, (0, 0), CFG.ref_blur_sigma)
    dead = (blurred < OLD_DARK_LEVEL * float(np.median(blurred))) | saturation_mask(
        ref.values, ref.full_scale, CFG) | ref.non_finite
    return cv2.dilate(dead.astype(np.uint8), np.ones((OLD_DILATE_PX,) * 2, np.uint8)) > 0


# ------------------------------------------------------------------ nothing is excluded

@pytest.mark.parametrize("kind", ["wall", "band", "clip"])
def test_dark_or_clipped_reference_areas_exclude_nothing(kind):
    ref = _scene(0.0, 1, **{kind: True})
    raw = _scene(45.0, 2, wall=kind == "wall", band=kind == "band")   # the frame is not clipped
    area = _old_rule(ref)
    assert area.sum() > 1000                                          # the old rule hit this scene
    norm, sat, image, excluded = _stage_a(raw, ref)
    assert not norm.invalid.any() and not excluded.any() and not sat.any()
    assert np.all(np.isfinite(image))
    assert np.std(image[area]) > 0.005                                # real values, not one flat fill
    assert not np.any(image[area] == 1.0)


@pytest.mark.parametrize("kind", ["wall", "band"])
def test_a_static_dark_area_divides_out_to_the_background(kind):
    ref = _scene(0.0, 1, **{kind: True})
    raw = _scene(45.0, 2, **{kind: True})
    _, _, image, _ = _stage_a(raw, ref)
    rows, cols = (slice(None), slice(5, 35)) if kind == "wall" else (slice(115, 135), slice(5, 100))
    assert np.median(image[rows, cols]) == pytest.approx(1.0, abs=0.03)
    assert np.median(image[:, 300:]) == pytest.approx(1.0, abs=0.01)  # background elsewhere


def test_a_bubble_seen_through_a_dark_band_is_measured_whole():
    # until edge-1.3.0 the band (8 % light) was excluded and the bubble's middle completed from
    # the circle; now its wall is found inside the band too
    res = detect_edge(_scene(45.0, 2, band=True), (1, 250), (1, 400), reference=_scene(0.0, 1, band=True))
    assert res.radius == pytest.approx(45.0, abs=1.0)
    assert (res.rc, res.cc) == pytest.approx((CENTRE[0] + 1, CENTRE[1] + 1), abs=1.0)
    assert res.provenance["region"]["hidden_fraction"] == 0.0
    assert not res.provenance["region"]["clipped"]
    in_band = (res.contour[:, 0] >= BAND_ROWS.start + 1) & (res.contour[:, 0] <= BAND_ROWS.stop)
    assert in_band.sum() > 10                                         # outline points inside the band


def test_the_ratio_stays_finite_where_the_reference_has_no_light():
    ref = _scene(0.0, 1).astype(np.float64)
    ref[:, WALL_COLS] = 0.0                                           # not a single count
    raw = _scene(45.0, 2).astype(np.float64)
    raw[:, WALL_COLS] = 0.0
    norm, _, image, excluded = _stage_a(raw.astype(np.uint16), ref.astype(np.uint16))
    assert not norm.invalid.any() and not excluded.any()
    assert np.all(np.isfinite(image))
    assert np.all(image[:, 5:35] == 0.0)                              # 0 / 0 reads 0, not NaN
    assert np.median(image[:, 300:]) == pytest.approx(1.0, abs=0.01)
    assert norm.gain == pytest.approx(1.0, abs=0.01)


# ------------------------------------------------------------------ only what has no value

def _with_nan(img: np.ndarray, rows: slice, cols: slice) -> np.ndarray:
    out = img.astype(np.float64)
    out[rows, cols] = np.nan
    return out


def test_non_finite_pixels_are_the_only_invalid_ones():
    ref = _with_nan(_scene(0.0, 1, wall=True), slice(60, 64), slice(250, 256))
    raw = _with_nan(_scene(45.0, 2, wall=True), slice(200, 204), slice(100, 110))
    norm, _, image, excluded = _stage_a(raw, ref)
    expected = np.zeros(SHAPE, bool)
    expected[60:64, 250:256] = expected[200:204, 100:110] = True      # exactly, not widened
    np.testing.assert_array_equal(norm.invalid, expected)
    np.testing.assert_array_equal(excluded, expected)
    assert np.all(image[expected] == 1.0) and np.all(np.isfinite(image))
    res = detect_edge(raw, (1, 250), (1, 400), reference=ref)
    assert res.radius == pytest.approx(45.0, abs=1.0)


def test_the_frames_own_saturation_is_still_excluded_for_a_dark_bubble():
    ref = _scene(0.0, 1, clip=True)
    raw = _scene(45.0, 2, clip=True)                                  # glare in the frame too
    norm, sat, _, excluded = _stage_a(raw, ref)
    assert not norm.invalid.any()
    assert sat[CLIP].all() and excluded[CLIP].all()                   # the frame's rule, unchanged
    _, sat_b, _, excluded_b = _stage_a(raw, ref, dark=False)
    assert sat_b[CLIP].all() and not excluded_b.any()                 # a bright bubble: signal


def test_engine_view_holds_the_real_ratio_in_the_dark_band():
    ref, raw = _scene(0.0, 1, band=True, clip=True), _scene(45.0, 2, band=True)
    view = engine_view(raw, (1, 250), (1, 400), reference=ref)
    _, _, image, _ = _stage_a(raw, ref)
    np.testing.assert_array_equal(view, image)
    band = view[115:135, 5:100]
    assert np.median(band) == pytest.approx(1.0, abs=0.03) and np.std(band) > 0.003


# ------------------------------------------------------------------ untouched pixels: bit-identical

def _stage_a_1_3(frame, ref, roi, *, dark, cfg):
    """Stage A as it was up to edge-1.3.0 (frozen copy), for the scenes the rule did not touch."""
    values = frame.values[roi]
    sat = saturation_mask(values, frame.full_scale, cfg)
    sat_excl = sat if dark else np.zeros_like(sat)
    invalid = frame.non_finite[roi].copy()
    if ref is None:
        surface = norm_mod.estimate_background(values, sat_excl | invalid, cfg)
        image = np.where(invalid, 1.0, values / np.maximum(surface, norm_mod._EPS)).astype(np.float32)
        method = f"surface(cubic+normconv {cfg.bg_sigma:g})" + norm_mod._no_background_note(sat_excl | invalid)
        norm = Normalized(image, sat_excl, invalid, method, 1.0)
    else:
        eps, scale = norm_mod._EPS, norm_mod._SURFACE_SCALE
        reference = ref.values[roi]
        blurred = cv2.GaussianBlur(reference, (0, 0), cfg.ref_blur_sigma)
        dead = ((blurred < OLD_DARK_LEVEL * float(np.median(blurred)))
                | saturation_mask(reference, ref.full_scale, cfg) | ref.non_finite[roi])
        invalid |= cv2.dilate(dead.astype(np.uint8), np.ones((OLD_DILATE_PX,) * 2, np.uint8)) > 0
        ratio = values / np.maximum(blurred, eps)
        ok = ~invalid & ~sat_excl
        gain = float(np.median(ratio[ok])) if ok.sum() > cfg.gain_min_pixels else 1.0
        gain = gain if np.isfinite(gain) and gain > eps else 1.0
        image = np.where(invalid, 1.0, ratio / gain).astype(np.float32)
        residual = norm_mod.estimate_background(image * scale, sat_excl | invalid, cfg) / scale
        image = np.where(invalid, 1.0, image / np.maximum(residual, eps)).astype(np.float32)
        method = f"reference(gain={gain:.3f})+surface" + norm_mod._no_background_note(sat_excl | invalid)
        norm = Normalized(image, sat_excl, invalid, method, gain)
    image = norm.image if dark else (2.0 - norm.image).astype(np.float32)
    return norm, sat, image, sat_excl | norm.invalid


def _assert_identical(a: EdgeResult, b: EdgeResult) -> None:
    """Every field equal to the bit (provenance compared without its wall times)."""
    for f in dataclasses.fields(EdgeResult):
        x, y = getattr(a, f.name), getattr(b, f.name)
        if f.name == "provenance":
            x = {k: v for k, v in x.items() if k != "timings_ms"}
            y = {k: v for k, v in y.items() if k != "timings_ms"}
        if isinstance(x, np.ndarray):
            assert x.dtype == y.dtype and x.shape == y.shape, f.name
            np.testing.assert_array_equal(x, y, err_msg=f.name)
        elif isinstance(x, frozenset):
            assert x == y, f.name
        else:
            assert repr(x) == repr(y), f.name


def _now_and_before(monkeypatch, raw, roi, ref, params=EdgeParams()) -> tuple[EdgeResult, EdgeResult]:
    now = detect_edge(raw, *roi, reference=ref, params=params)
    monkeypatch.setattr(engine, "normalize_frame_roi", _stage_a_1_3)
    before = detect_edge(raw, *roi, reference=ref, params=params)
    monkeypatch.undo()
    return now, before


@pytest.mark.parametrize("roi", [((1, 250), (1, 400)), ((31, 230), (61, 380))])
def test_clean_references_give_bit_identical_results(monkeypatch, roi):
    ref = _scene(0.0, 1)
    assert not _old_rule(ref).any()
    now, before = _now_and_before(monkeypatch, _scene(45.0, 2), roi, ref)
    assert np.isfinite(now.radius)
    _assert_identical(now, before)


def test_bright_bubbles_and_no_reference_are_bit_identical(monkeypatch):
    bright = (65535 - _scene(40.0, 2)).astype(np.uint16)
    ref = (65535 - _scene(0.0, 1)).astype(np.uint16)
    assert not _old_rule(ref).any()
    _assert_identical(*_now_and_before(monkeypatch, bright, ((1, 250), (1, 400)), ref,
                                       EdgeParams(polarity="bright")))
    _assert_identical(*_now_and_before(monkeypatch, _scene(40.0, 2, wall=True), ((1, 250), (1, 400)), None))


def test_a_dark_reference_band_changes_the_result_as_intended(monkeypatch):
    # the band case is where 1.4.0 and 1.3.0 differ: the old rule hid the band
    now, before = _now_and_before(monkeypatch, _scene(45.0, 2, band=True), ((1, 250), (1, 400)),
                                  _scene(0.0, 1, band=True))
    assert before.provenance["region"]["hidden_fraction"] > 0.05
    assert now.provenance["region"]["hidden_fraction"] == 0.0
