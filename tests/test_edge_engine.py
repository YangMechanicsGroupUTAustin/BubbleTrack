"""End-to-end behaviour of ``detect_edge`` on synthetic shadowgraphs with exact ground truth."""

from __future__ import annotations

import json
import time
from functools import lru_cache

import numpy as np
import pytest
from scipy import ndimage

from _edge_synth import Bubble, Scene, render
from bubbletrack.model.circle_fit import circle_fit_taubin
from bubbletrack.model.edge import ENGINE_VERSION, EdgeParams, detect_edge

FULL = ((1, 250), (1, 400))
BASE = Scene(bubbles=(Bubble(rc=118.4, cc=207.7, r=41.3),))
HALO6 = Scene(bubbles=(Bubble(rc=125.0, cc=200.0, r=40.0, halo_w=6, halo_trans=0.6),))
HALO12 = Scene(bubbles=(Bubble(rc=125.0, cc=200.0, r=40.0, rim=False, halo_w=12, halo_trans=0.45),))
TWO = Scene(bubbles=(Bubble(rc=125, cc=110, r=30), Bubble(rc=125, cc=280, r=50, spot=False)))
CLIPPED = Scene(bubbles=(Bubble(rc=24.0, cc=205.0, r=60.0),))


@lru_cache(maxsize=None)
def frame(scene: Scene):
    img, meta = render(scene)
    img.flags.writeable = False
    meta["ref"].flags.writeable = False
    return img, meta


def measure(scene: Scene, *, ref: bool = True, **params):
    img, meta = frame(scene)
    return detect_edge(img, *FULL, reference=meta["ref"] if ref else None, params=EdgeParams(**params))


def rel_err(res, scene: Scene, target: int = 0) -> float:
    b = scene.bubbles[target]
    return abs(res.radius - b.r) / b.r


# ------------------------------------------------------------------ coordinates

def test_centre_follows_the_app_convention_like_taubin():
    """Pixel (i, j) has coordinate (i+1, j+1): same as the existing detectors + Taubin."""
    scene = Scene(bubbles=(Bubble(rc=100.3, cc=170.7, r=35.0, rim=False, ring=False, spot=False),), noise=0.3)
    img, meta = frame(scene)
    disk = meta["clean"] < 0.5 * scene.bg
    rows, cols = np.nonzero(disk & ~ndimage.binary_erosion(disk))
    rc_t, cc_t, _ = circle_fit_taubin(np.column_stack([rows + 1, cols + 1]).astype(float))
    res = detect_edge(img, *FULL)
    assert (res.rc, res.cc) == pytest.approx((rc_t, cc_t), abs=0.15)
    assert (res.rc, res.cc) == pytest.approx((101.3, 171.7), abs=0.05)
    assert res.radius == pytest.approx(35.0, rel=0.005)
    # Taubin on the engine's own edge points reproduces its circle (same coordinate frame)
    assert circle_fit_taubin(res.edge_xy) == pytest.approx((res.rc, res.cc, res.radius), abs=0.05)


def test_roi_offset_keeps_full_image_coordinates():
    full = measure(BASE)
    img, meta = frame(BASE)
    sub = detect_edge(img, (41, 210), (121, 330), reference=meta["ref"])
    assert (sub.rc, sub.cc, sub.radius) == pytest.approx((full.rc, full.cc, full.radius), abs=0.05)
    assert sub.mask.shape == (170, 210)
    assert sub.edge_xy[:, 0].min() >= 41 and sub.edge_xy[:, 1].min() >= 121


def test_edge_points_lie_on_the_reported_circle():
    res = measure(BASE)
    d = np.hypot(res.edge_xy[:, 0] - res.rc, res.edge_xy[:, 1] - res.cc)
    assert res.edge_xy.shape[1] == 2 and len(res.edge_xy) > 200
    assert np.median(np.abs(d - res.radius)) < 0.2


# ------------------------------------------------------------------ inputs

def _as(img: np.ndarray, kind: str) -> np.ndarray:
    if kind == "float32_counts":
        return img.astype(np.float32)
    if kind == "float64_unit":
        return img / 65535.0
    if kind == "uint8":
        return (img // 257).astype(np.uint8)
    if kind == "bgr16":
        return np.dstack([img] * 3)
    if kind == "bgra16":
        return np.dstack([img] * 3 + [np.full_like(img, 65535)])
    raise ValueError(kind)


@pytest.mark.parametrize("kind, tol", [("float32_counts", 1e-9), ("float64_unit", 1e-4), ("bgr16", 2e-3),
                                       ("bgra16", 2e-3), ("uint8", 5e-3)])
def test_any_dtype_and_layout_gives_the_same_radius(kind, tol):
    img, meta = frame(BASE)
    want = detect_edge(img, *FULL, reference=meta["ref"])
    got = detect_edge(_as(img, kind), *FULL, reference=_as(meta["ref"], kind))
    assert got.radius == pytest.approx(want.radius, rel=tol)


def test_inputs_are_never_modified():
    img, meta = frame(BASE)
    raw, ref = img.copy(), meta["ref"].copy()
    detect_edge(raw, *FULL, reference=ref, params=EdgeParams(edge_definition="outer"))
    np.testing.assert_array_equal(raw, img)
    np.testing.assert_array_equal(ref, meta["ref"])


def test_non_finite_pixels_do_not_break_the_measurement():
    img, _ = frame(BASE)
    raw = img.astype(np.float64)
    raw[5:9, 5:300] = np.nan
    raw[240, 390] = np.inf
    res = detect_edge(raw, *FULL)
    assert rel_err(res, BASE) < 0.01


def test_reference_of_another_size_is_rejected():
    img, meta = frame(BASE)
    with pytest.raises(ValueError, match="reference"):
        detect_edge(img, *FULL, reference=meta["ref"][:-1])


@pytest.mark.parametrize("gx, gy", [((1,), (1, 400)), (None, (1, 400)), (("a", 3), (1, 400))])
def test_malformed_roi_is_rejected(gx, gy):
    img, _ = frame(BASE)
    with pytest.raises(ValueError):
        detect_edge(img, gx, gy)


def test_tiny_roi_returns_a_flagged_empty_result():
    img, _ = frame(BASE)
    res = detect_edge(img, (10, 13), (10, 200))
    assert np.isnan(res.radius) and not res.accepted and "roi_too_small" in res.flags


def test_roi_beyond_the_image_is_clamped():
    img, meta = frame(BASE)
    res = detect_edge(img, (-20, 400), (0, 999), reference=meta["ref"])
    assert rel_err(res, BASE) < 0.01
    assert any("clamped" in f for f in res.provenance["fallbacks"])


# ------------------------------------------------------------------ behaviour

@pytest.mark.parametrize("scene", [
    Scene(), Scene(static=1), Scene(static=2, vignette=0.3, gradient=0.2), Scene(speckle=0.45),
    Scene(noise=4.0), Scene(static=1, speckle=0.3, noise=2.0),
], ids=["clean", "static", "static_strong_illum", "speckle", "noisy", "speckle_static_noisy"])
@pytest.mark.parametrize("use_ref", [True, False], ids=["ref", "noref"])
def test_no_bubble_frame_has_low_confidence(scene, use_ref):
    res = measure(scene, ref=use_ref)
    assert res.confidence < 0.3 and not res.accepted


def test_a_frame_below_the_gate_reports_the_circle():
    """An untrusted region (no-bubble texture here) falls back to the circle fit, still rejected."""
    res = measure(Scene(static=1), ref=False)
    assert np.isfinite(res.radius) and not res.accepted
    assert res.radius_kind == "circle" and "circle_fallback" in res.flags
    assert res.radius == res.r_circle and np.isfinite(res.r_area)


def test_reference_free_mode_is_flagged():
    with_ref, without = measure(BASE), measure(BASE, ref=False)
    assert "no_reference" in without.flags and "no_reference" not in with_ref.flags
    assert without.provenance["normalize"].startswith("surface")
    assert rel_err(without, BASE) < 0.005 and rel_err(with_ref, BASE) < 0.005


def test_clean_bubble_has_no_shoulder():
    res = measure(BASE)
    assert res.accepted and "shoulder" not in res.flags
    assert 1.0 < res.shoulder_width < 3.0
    assert res.outer_radius == pytest.approx(res.r_circle + res.shoulder_width)   # steep mode: circle fits
    assert res.r_area == res.radius == pytest.approx(res.r_circle, abs=0.1)


def test_two_step_edge_raises_the_shoulder_flag():
    res = measure(HALO6)
    assert rel_err(res, HALO6) < 0.02                     # steep edge = the bubble wall
    assert "shoulder" in res.flags and res.shoulder_width > 3.5
    assert res.outer_radius == pytest.approx(47.0, abs=1.0)


def test_outer_definition_reports_the_outer_extent():
    """Outer mode measures the region at the outer-extent level (its area radius); steep mode
    reports B from the circle fit of the outer-extent points.  On a round bubble they agree."""
    steep, outer = measure(HALO6), measure(HALO6, edge_definition="outer")
    assert outer.radius == outer.outer_radius == pytest.approx(steep.outer_radius, abs=0.1)
    assert outer.shoulder_width == pytest.approx(steep.shoulder_width, abs=0.1)
    assert outer.provenance["steep_radius"] == pytest.approx(steep.radius, abs=1e-9)
    assert outer.r_circle == pytest.approx(steep.outer_radius, abs=1e-9)
    assert "shoulder" in outer.flags and outer.accepted


def test_edge_choice_inner_takes_the_ring_nearest_the_core():
    outer, inner = measure(HALO12), measure(HALO12, edge_choice="inner")
    assert outer.radius == pytest.approx(52.0, rel=0.02)   # halo's outer edge straddles mid level
    assert rel_err(inner, HALO12) < 0.02


def test_clipped_bubble_is_completed_with_its_circle():
    res = measure(CLIPPED)
    assert {"clipped", "clipped_completed"} <= res.flags and res.radius_kind == "area+circle_completion"
    assert res.r_area == res.radius == pytest.approx(res.r_circle, rel=0.005)
    assert rel_err(res, CLIPPED) < 0.01


def test_seed_and_prior_choose_the_target():
    auto = measure(TWO)
    assert rel_err(auto, TWO, target=1) < 0.02 and "ambiguous_selection" in auto.flags
    seeded = measure(TWO, seed=(130.0, 115.0))
    assert rel_err(seeded, TWO) < 0.02 and "ambiguous_selection" not in seeded.flags
    primed = measure(TWO, prior=(126.0, 111.0, 30.0))
    assert rel_err(primed, TWO) < 0.02


def test_bright_polarity_measures_a_bright_bubble():
    h, w = 250, 400
    yy, xx = np.mgrid[0:h, 0:w]
    cover = np.clip(37.5 - np.hypot(yy - 120.2, xx - 190.6) + 0.5, 0, 1)
    clean = ndimage.gaussian_filter(8000.0 * (1 + 2.0 * cover), 1.0)
    rng = np.random.default_rng(4)
    img = np.clip(clean + 150 * rng.standard_normal((h, w)), 0, 65535).astype(np.uint16)
    res = detect_edge(img, *FULL, params=EdgeParams(polarity="bright"))
    assert res.accepted and res.radius == pytest.approx(37.5, rel=0.01)
    assert (res.rc, res.cc) == pytest.approx((121.2, 191.6), abs=0.1)
    dark = detect_edge(img, *FULL)
    assert not (dark.accepted and abs(dark.radius - 37.5) < 0.75)


def test_result_carries_mask_and_provenance():
    res = measure(BASE)
    assert res.mask.dtype == bool and res.mask.shape == (250, 400)
    assert res.mask[118, 227] and not res.mask[10, 10]       # bubble body (beside the bright spot) vs background
    prov = res.provenance
    assert prov["engine"] == ENGINE_VERSION and prov["failure"] is None
    for key in ("normalize", "segment", "select", "edges", "fit", "outer", "params", "fallbacks"):
        assert key in prov
    assert set(prov["confidence_terms"]) == {"support", "precision", "completion", "snr", "selection",
                                             "edge_unique"}
    assert {"normalize", "segment_select", "edges_fit", "outer", "region", "total"} <= set(prov["timings_ms"])
    json.dumps(prov)                                         # storable in sessions / exports as is
    assert 0 < res.radius_sigma < 0.1 and 0.9 < res.arc_coverage <= 1 and res.inlier_fraction > 0.9


def test_failure_names_its_reason():
    img = np.full((250, 400), 30000, np.uint16)
    res = detect_edge(img, *FULL)
    assert np.isnan(res.radius) and np.isnan(res.rc) and res.confidence == 0.0
    assert res.edge_xy.shape == (0, 2) and res.provenance["failure"] in res.flags


def test_speed_on_a_250x400_roi():
    """Median of 20 frames.  Target 40 ms; the absolute bound is generous so CI does not flake,
    the region stage's share is checked instead (it scales with machine load like the rest).

    Measured at the port (edge-1.0.0): median 22.5 ms with a reference, 20.1 ms reference-free.
    Region layer (edge-1.1.0): the region stage adds 14 % (median) to the other stages, i.e. about
    26-29 ms on that machine; on a loaded machine both versions ran at 47-56 vs 52-68 ms.
    """
    scenes = [Scene(bubbles=(Bubble(rc=125.0, cc=200.0, r=10.0 + 4.0 * k),), seed=k) for k in range(20)]
    frames = [frame(s) for s in scenes]
    detect_edge(frames[0][0], *FULL, reference=frames[0][1]["ref"])       # warm-up
    times, share = [], []
    for img, meta in frames:
        t0 = time.perf_counter()
        res = detect_edge(img, *FULL, reference=meta["ref"])
        times.append(time.perf_counter() - t0)
        laps = res.provenance["timings_ms"]
        share.append(laps["region"] / (laps["total"] - laps["region"]))
    assert np.median(times) <= 0.200
    assert np.median(share) <= 0.35
