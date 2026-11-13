"""Reference pixels without usable data (too dark, saturated, non-finite): one definition,
used by the normalisation and by the Auto view's "not usable" fill, and the refactor that
shared it leaves every measurement bit-identical."""

from __future__ import annotations

import dataclasses
from dataclasses import replace

import cv2
import numpy as np
import pytest

from _golden_cases import SceneSpec, render_scene
from bubbletrack.model.conventions import roi_to_slice
from bubbletrack.model.edge import engine
from bubbletrack.model.edge import normalize as norm_mod
from bubbletrack.model.edge.config import NormalizeConfig
from bubbletrack.model.edge.engine import EdgeParams, EdgeResult, detect_edge, unusable_mask
from bubbletrack.model.edge.normalize import (
    Normalized, normalize_frame_roi, normalize_roi, reference_unusable, saturation_mask, to_gray_frame,
)

CFG = NormalizeConfig()
SHAPE = (250, 400)
CENTRE = (125.0, 200.0)
WALL_COLS = 40  # a black wall at the left: no light in the reference
GLARE = (slice(20, 50), slice(300, 340))  # a saturated block
BAND_ROWS = slice(118, 128)  # a dark band through the bubble


def _scene(radius: float, seed: int, *, wall=False, glare=False, band=False) -> np.ndarray:
    """A uint16 frame; the static features are part of the setup (reference and frames)."""
    img = render_scene(SceneSpec(radius=radius, center=CENTRE, shape=SHAPE, seed=seed))
    if wall:
        img[:, :WALL_COLS] = 300
    if glare:
        img[GLARE] = 65535
    if band:
        img[BAND_ROWS, :] = 200
    return img


def _with_nan(img: np.ndarray, rows: slice, cols: slice) -> np.ndarray:
    out = img.astype(np.float64)
    out[rows, cols] = np.nan
    return out


# ------------------------------------------------------------------ one definition

@pytest.mark.parametrize("kind", ["wall", "glare", "clean"])
def test_reference_unusable_is_the_reference_part_of_invalid(kind):
    ref = to_gray_frame(_scene(0.0, 1, wall=kind == "wall", glare=kind == "glare")).values
    frame = to_gray_frame(_scene(40.0, 2, wall=kind == "wall", glare=kind == "glare")).values
    frame_bad = np.zeros(SHAPE, bool)
    frame_bad[200:204, 100:110] = True                    # the frame's own non-finite pixels
    ref_bad = saturation_mask(ref, 65535.0, CFG)
    out = normalize_roi(frame, np.zeros(SHAPE, bool), CFG, reference=ref, ref_invalid=ref_bad,
                        invalid=frame_bad)
    unusable = reference_unusable(ref, ref_bad, CFG)
    assert unusable.dtype == bool and unusable.shape == SHAPE
    np.testing.assert_array_equal(out.invalid & ~frame_bad, unusable & ~frame_bad)
    np.testing.assert_array_equal(out.invalid, unusable | frame_bad)
    if kind == "clean":
        assert not unusable.any()
    elif kind == "wall":
        assert unusable[:, :WALL_COLS - 2].all() and not unusable[:, WALL_COLS + 4:].any()
    else:
        assert unusable[GLARE].all() and ref_bad[GLARE].all()
        assert unusable.sum() < 2 * ref_bad.sum()        # the block, slightly widened


def test_reference_unusable_includes_the_dilation():
    ref = to_gray_frame(_scene(0.0, 1, wall=True)).values
    wide = reference_unusable(ref, None, CFG)
    narrow = reference_unusable(ref, None, replace(CFG, ref_invalid_dilate_px=1))
    assert (wide >= narrow).all() and wide.sum() > narrow.sum()


@pytest.mark.parametrize("kind", ["wall", "glare", "clean"])
def test_engine_unusable_mask_matches_what_the_engine_ignores(kind):
    ref_raw = _with_nan(_scene(0.0, 1, wall=kind == "wall", glare=kind == "glare"),
                        slice(60, 64), slice(250, 256))   # non-finite reference pixels
    raw = _with_nan(_scene(40.0, 2, wall=kind == "wall", glare=kind == "glare"),
                    slice(200, 204), slice(100, 110))     # the frame's own non-finite pixels
    gridx, gridy = (11, 240), (21, 390)
    roi = roi_to_slice(gridx, gridy)
    frame, ref = to_gray_frame(raw), to_gray_frame(ref_raw)
    norm, _, _, _ = normalize_frame_roi(frame, ref, roi, dark=True, cfg=CFG)  # detect_edge's stage A
    mask = unusable_mask(ref_raw, gridx, gridy)
    own = frame.non_finite[roi]
    assert mask.shape == norm.invalid.shape == (230, 370)
    np.testing.assert_array_equal(mask & ~own, norm.invalid & ~own)
    np.testing.assert_array_equal(norm.invalid, mask | own)
    assert mask[60 - 10:64 - 10, 250 - 20:256 - 20].all()  # the reference's NaNs (ROI coords)
    if kind == "clean":
        assert mask.sum() < 100                             # only the widened NaN patch
    else:
        assert mask.sum() > 1000


def test_engine_unusable_mask_validates_its_roi():
    with pytest.raises(ValueError):
        unusable_mask(_scene(0.0, 1), ("a", 3), (1, 10))


# ------------------------------------------------------------------ measurements unchanged

def _stage_a_before(frame, ref, roi, *, dark, cfg):
    """The engine's ``_normalize`` as it was before stage A moved to ``normalize_frame_roi``
    and the reference mask was factored out (frozen copy: what the refactor is compared with)."""
    values = frame.values[roi]
    sat = saturation_mask(values, frame.full_scale, cfg)
    sat_excl = sat if dark else np.zeros_like(sat)
    ref_values, ref_bad = None, None
    if ref is not None:
        ref_values = ref.values[roi]
        ref_bad = saturation_mask(ref_values, ref.full_scale, cfg) | ref.non_finite[roi]
    norm = _normalize_roi_before(values, sat_excl, cfg, reference=ref_values, ref_invalid=ref_bad,
                                 invalid=frame.non_finite[roi])
    image = norm.image if dark else (2.0 - norm.image).astype(np.float32)
    return norm, sat, image, sat_excl | norm.invalid


def _normalize_roi_before(values, saturated, cfg, *, reference=None, ref_invalid=None, invalid=None):
    """``normalize_roi`` as it was before the reference mask was factored out (frozen copy)."""
    eps, scale = norm_mod._EPS, norm_mod._SURFACE_SCALE
    note = norm_mod._no_background_note
    invalid = np.zeros(values.shape, bool) if invalid is None else invalid.copy()
    if reference is None:
        surface = norm_mod.estimate_background(values, saturated | invalid, cfg)
        image = values / np.maximum(surface, eps)
        image = np.where(invalid, 1.0, image).astype(np.float32)
        method = f"surface(cubic+normconv {cfg.bg_sigma:g})" + note(saturated | invalid)
        return Normalized(image, saturated, invalid, method, 1.0)
    blurred = cv2.GaussianBlur(reference, (0, 0), cfg.ref_blur_sigma)
    med = float(np.median(blurred))
    dead = blurred < cfg.ref_invalid_level * med
    if ref_invalid is not None:
        dead |= ref_invalid
    k = cfg.ref_invalid_dilate_px
    invalid |= cv2.dilate(dead.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
    ratio = values / np.maximum(blurred, eps)
    ok = ~invalid & ~saturated
    gain = float(np.median(ratio[ok])) if ok.sum() > cfg.gain_min_pixels else 1.0
    gain = gain if np.isfinite(gain) and gain > eps else 1.0
    image = np.where(invalid, 1.0, ratio / gain).astype(np.float32)
    residual = norm_mod.estimate_background(image * scale, saturated | invalid, cfg) / scale
    image = np.where(invalid, 1.0, image / np.maximum(residual, eps)).astype(np.float32)
    method = f"reference(gain={gain:.3f})+surface" + note(saturated | invalid)
    return Normalized(image, saturated, invalid, method, gain)


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


SCENES = {
    "clean": dict(),
    "wall": dict(wall=True),
    "glare": dict(glare=True),
    "band_through_bubble": dict(band=True),
}


def _now_and_before(monkeypatch, raw, roi, ref, params=EdgeParams()) -> tuple[EdgeResult, EdgeResult]:
    now = detect_edge(raw, *roi, reference=ref, params=params)
    monkeypatch.setattr(engine, "normalize_frame_roi", _stage_a_before)
    before = detect_edge(raw, *roi, reference=ref, params=params)
    monkeypatch.undo()
    return now, before


@pytest.mark.parametrize("name", list(SCENES))
@pytest.mark.parametrize("roi", [((1, 250), (1, 400)), ((31, 230), (61, 380))])
def test_detect_edge_is_bit_identical_to_before_the_refactor(monkeypatch, name, roi):
    now, before = _now_and_before(monkeypatch, _scene(45.0, 2, **SCENES[name]), roi,
                                  _scene(0.0, 1, **SCENES[name]))
    assert np.isfinite(before.radius) or name == "band_through_bubble"
    _assert_identical(now, before)


def test_detect_edge_with_non_finite_pixels_is_bit_identical(monkeypatch):
    ref = _with_nan(_scene(0.0, 1, wall=True), slice(60, 64), slice(250, 256))
    raw = _with_nan(_scene(45.0, 2, wall=True), slice(200, 204), slice(100, 110))
    _assert_identical(*_now_and_before(monkeypatch, raw, ((1, 250), (1, 400)), ref,
                                       EdgeParams(non_round=True)))


def test_bright_bubbles_and_no_reference_are_bit_identical(monkeypatch):
    bright = (65535 - _scene(40.0, 2, glare=True)).astype(np.uint16)  # a bright bubble, dark glare
    ref = (65535 - _scene(0.0, 1, glare=True)).astype(np.uint16)
    _assert_identical(*_now_and_before(monkeypatch, bright, ((1, 250), (1, 400)), ref,
                                       EdgeParams(polarity="bright")))
    _assert_identical(*_now_and_before(monkeypatch, _scene(40.0, 2, wall=True), ((1, 250), (1, 400)), None))
