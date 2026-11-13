"""The default radius: the least-squares circle through the bubble outline (owner, 2026-11-13, "A").

With "Bubble may be non-round" off, R and the centre are Taubin's circle through the region's
genuine outline points (``Region.edge``: not on the ROI border, not next to excluded pixels),
refitted once without the points whose radial residual is more than 3 robust sigmas
(1.4826 MAD) off; on every frame.  The robust RANSAC + Tukey fit of the ray edges still finds
the bubble inside the engine; it is reported only when fewer than ~20 outline points remain or
the outline fit fails (``provenance["circle"]["fallback"]``).

Why: the robust fit latches onto one flank of an elongated bubble (1.7:1, area radius 60:
R 98.9, centre 51 px off); the outline circle gives R 63.0 centred within 0.5 px.  The
expected radii are the evaluation's (``circle_estimator/synth_results.jsonl``, estimator
"out_trim"), on the same synthetic frames.

Checks that the reported circle cannot fool: inside the ROI, ``non_circular`` when it is more
than 5 % off the area-equivalent radius (the radius the other definition reports); cut by the
ROI border, ``clipped_not_round`` when the ellipse-completed area OR the robust ray circle
disagrees with it beyond the clipped tolerance.
"""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache

import numpy as np
import pytest

import _edge_shapes as ES
from _edge_synth import Bubble, Scene, boundary_radius, render, truth
from bubbletrack.model.edge import EdgeParams, detect_edge
from bubbletrack.model.edge.config import DEFAULT_CONFIG, FitConfig, QualityConfig
from bubbletrack.model.edge.engine import detect_edge_with_config
from bubbletrack.model.edge.fit import outline_circle, trimmed_taubin
from bubbletrack.model.edge_pipeline import EdgeJob, fit_from_result
from bubbletrack.model.guided_fit import drawn_circle, fit_guided

H, W = 250, 400
FULL = ((1, H), (1, W))
OFF, ON = EdgeParams(), EdgeParams(non_round=True)
REVIEW_FLAGS = {"non_circular", "clipped_not_round", "clipped_unmeasurable"}


def _ring(n: int, r: float = 50.0, c=(100.0, 120.0), noise: float = 0.0, seed: int = 0) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    rng = np.random.default_rng(seed)
    rr = r + noise * rng.standard_normal(n)
    return np.column_stack([c[0] + rr * np.sin(t), c[1] + rr * np.cos(t)])


# ------------------------------------------------------------------ the estimator

def test_trimmed_taubin_is_taubin_on_a_clean_circle():
    fit = trimmed_taubin(_ring(400, noise=0.2), 3.0)
    assert (fit.rc, fit.cc, fit.radius) == pytest.approx((100.0, 120.0, 50.0), abs=0.05)
    assert fit.weights.mean() > 0.95 and fit.method == "taubin+trim"
    assert 0 < fit.radius_sigma < 0.05


def test_one_trimming_pass_drops_a_bump_and_refits():
    """A 20-point bump 8 px outside the wall pulls plain Taubin off; one pass removes it."""
    p = _ring(400, noise=0.2)
    p[:20] += 8.0 * (p[:20] - [100.0, 120.0]) / 50.0
    fit = trimmed_taubin(p, 3.0)
    assert not fit.weights[:20].any() and fit.weights[20:].mean() > 0.95
    assert (fit.rc, fit.cc, fit.radius) == pytest.approx((100.0, 120.0, 50.0), abs=0.1)


def test_the_outline_circle_needs_20_points():
    assert FitConfig().outline_min_points == 20 and FitConfig().outline_trim_k == 3.0
    assert outline_circle(_ring(19), FitConfig()) is None
    assert outline_circle(_ring(20), FitConfig()).radius == pytest.approx(50.0)
    straight = np.column_stack([np.zeros(50), np.arange(50.0)])
    assert outline_circle(straight, FitConfig()) is None          # no circle: the fit fails


# ------------------------------------------------------------------ scenes of the evaluation

@lru_cache(maxsize=None)
def _bubble(aspect: float = 1.0, angle: float = 0.0, r: float = 60.0, rc: float = 125.0, cc: float = 200.0,
            glare: bool = False, debris: tuple = ()):
    b = Bubble(rc=rc, cc=cc, r=r, aspect=aspect, angle=angle)
    img, meta = render(Scene(bubbles=(b,), debris=debris))
    if glare:  # the owner's repro: a 5x5 saturated blob just outside the right wall
        c0 = int(round(cc + float(boundary_radius(b, np.array([0.0]))[0]))) + 1
        img = img.copy()
        img[int(rc) - 2:int(rc) + 3, c0:c0 + 5] = img.max()
    return img, meta["ref"], truth(b)["r_eq"]


def _shape(name: str):
    return next(s for s in ES.SHAPES if s.name == name)


@lru_cache(maxsize=None)
def _shape_frame(name: str, cut: float = 0.0):
    """A shape of ``_edge_shapes`` (its centroid, area radius), cut by the ROI top as in the evaluation."""
    mask = _shape(name).mask
    img, radii = ES.render_masks([(mask, ES.DARK)], seed=0)
    yy, xx = ES._grid()
    body = mask(yy, xx)
    centre = (float(yy[body].mean()) + 1, float(xx[body].mean()) + 1)          # 1-based
    gridx = (1, H)
    if cut > 0:
        rows = np.nonzero(body.any(1))[0]
        top, bot = rows[0] / ES.SS - 0.5, (rows[-1] + 1) / ES.SS - 0.5
        gridx = (int(np.ceil(top + cut * (bot - top) + 0.5)) + 1, H)
    return img, gridx, radii[0], centre


def _run_bubble(params=OFF, **kw):
    img, ref, r_eq = _bubble(**kw)
    return detect_edge(img, *FULL, reference=ref, params=params), r_eq


def _run_shape(name: str, cut: float = 0.0, params=OFF):
    img, gridx, r_eq, centre = _shape_frame(name, cut)
    return detect_edge(img, gridx, (1, W), reference=ES.reference(), params=params), r_eq, centre


# ------------------------------------------------------------------ non-round bubbles: a centred circle

ELONGATED = {  # (aspect, angle, R): evaluation's outline circle radius
    "1.5:1 along": ((1.5, 0.0, 60.0), 61.53),
    "1.5:1 across": ((1.5, np.pi / 2, 60.0), 61.52),
    "1.7:1 along": ((1.7, 0.0, 60.0), 62.91),
    "1.7:1 across": ((1.7, np.pi / 2, 60.0), 62.90),
    "1.7:1 along R30": ((1.7, 0.0, 30.0), 31.43),
}


@pytest.mark.parametrize("name", sorted(ELONGATED))
def test_an_elongated_bubble_gets_the_centred_least_squares_circle(name):
    (aspect, angle, r), expected = ELONGATED[name]
    res, r_eq = _run_bubble(aspect=aspect, angle=angle, r=r)
    assert (res.rc, res.cc) == pytest.approx((126.0, 201.0), abs=1.0)           # 1-based truth
    assert res.radius == pytest.approx(expected, abs=0.15)
    assert 0 < res.radius - r_eq <= 0.05 * r_eq
    assert res.accepted and res.radius_kind == "circle" and res.radius == res.r_circle
    assert res.provenance["circle"]["fallback"] is None


@pytest.mark.parametrize("name, expected", [("peanut", 60.03), ("five_lobe", 56.73)])
def test_lobed_bubbles_get_the_centred_least_squares_circle(name, expected):
    res, r_eq, centre = _run_shape(name)
    assert (res.rc, res.cc) == pytest.approx(centre, abs=1.0)
    assert res.radius == pytest.approx(expected, abs=0.15)
    assert ("non_circular" in res.flags) == (abs(res.radius - res.r_area) / res.r_area > 0.05)


def test_the_owners_repro_is_centred():
    """``centre_offset_demo.py``: 1.7:1, area radius 60, glare blob at the right wall; the robust
    circle was 51.5 px off (R 98.9).  The outline circle: centred within 1 px, R 63.0."""
    res, r_eq = _run_bubble(aspect=1.7, angle=np.pi / 2, glare=True)
    assert {"clipped", "saturation"} <= res.flags
    assert (res.rc, res.cc) == pytest.approx((126.0, 201.0), abs=1.0)
    assert res.radius == pytest.approx(63.0, abs=0.15)
    assert abs(res.provenance["steep_r_circle"] - r_eq) > 0.3 * r_eq             # the ray fit hugs a flank


# ------------------------------------------------------------------ round bubbles stay exact

@pytest.mark.parametrize("r", (20.0, 40.0, 60.0, 100.0))
def test_a_round_bubble_within_half_a_percent(r):
    res, r_eq = _run_bubble(r=r)
    assert res.accepted and res.radius == pytest.approx(r_eq, rel=0.005)
    assert (res.rc, res.cc) == pytest.approx((126.0, 201.0), abs=0.2)
    assert not REVIEW_FLAGS & res.flags


@pytest.mark.parametrize("side", ("top", "left"))
@pytest.mark.parametrize("cut", (0.1, 0.3, 0.5))
@pytest.mark.parametrize("r", (40.0, 60.0))
def test_a_round_bubble_cut_by_the_border_within_half_a_percent(r, cut, side):
    pos = r * (1 - 2 * cut) - 0.5
    rc, cc = (pos, 200.0) if side == "top" else (125.0, pos)
    res, r_eq = _run_bubble(r=r, rc=rc, cc=cc)
    assert res.accepted and res.radius == pytest.approx(r_eq, rel=0.005)
    assert (res.rc, res.cc) == pytest.approx((rc + 1, cc + 1), abs=0.3)
    assert "clipped" in res.flags and not REVIEW_FLAGS & res.flags


DEBRIS = {  # the evaluation's "I robustness" debris scenes: (R, debris), outline circle R
    "debris_attached": (40.0, ((125, 244, 7, 0.08), (88, 185, 6, 0.1)), 40.22),
    "debris_attached_big": (40.0, ((125, 246, 10, 0.08),), 40.98),
    "debris_3_attached": (50.0, ((125, 254, 6, 0.08), (80, 170, 5, 0.1), (170, 180, 5, 0.1)), 50.01),
}


@pytest.mark.parametrize("name", sorted(DEBRIS))
def test_debris_touching_the_wall_costs_at_most_2_5_percent(name):
    r, debris, expected = DEBRIS[name]
    res, _ = _run_bubble(r=r, debris=debris)
    assert res.accepted and abs(res.radius / r - 1) <= 0.025
    assert res.radius == pytest.approx(expected, abs=0.15)


# ------------------------------------------------------------------ the checks

FIVE_PERCENT_CASES = [dict(aspect=a, angle=g) for a in (1.3, 1.5, 1.7) for g in (0.0, np.pi / 4, np.pi / 2)]


@pytest.mark.parametrize("kw", FIVE_PERCENT_CASES, ids=lambda kw: f"{kw['aspect']}:1@{kw['angle']:.2f}")
def test_every_frame_the_definition_changes_by_more_than_5_percent_is_flagged(kw):
    """Inside the ROI the check IS the other definition: the area radius ON reports."""
    off, _ = _run_bubble(OFF, **kw)
    on, _ = _run_bubble(ON, **kw)
    assert QualityConfig().circle_check_rel == 0.05
    assert off.provenance["circle_check"]["method"] == "area"
    assert on.radius == pytest.approx(off.r_area, abs=1e-9)
    changes = abs(off.radius - on.radius) / on.radius
    assert ("non_circular" in off.flags) == (changes > 0.05)
    assert on.r_circle == pytest.approx(off.radius, abs=1e-9)                  # the circle, either mode


def test_a_cut_dented_bubble_is_caught_by_the_robust_probe():
    """Kidney, 30 % cut: the outline circle and the ellipse completion are wrong together
    (-11 % vs the area truth) and agree with each other; the robust ray circle does not."""
    res, r_eq, _ = _run_shape("kidney", 0.3)
    q = QualityConfig()
    est = res.provenance["circle_check"]["r_estimate"]
    assert res.provenance["circle_check"]["method"] == "ellipse"
    assert res.radius < 0.95 * r_eq                                            # really wrong
    assert abs(res.radius - est) <= max(q.clipped_check_rel * est, q.clipped_check_px)   # ellipse misses it
    robust = res.provenance["steep_r_circle"]
    assert abs(res.radius - robust) > max(q.clipped_check_rel * robust, q.clipped_check_px)
    assert "clipped_not_round" in res.flags


def test_a_round_bubble_cut_by_the_border_passes_both_probes():
    res, _ = _run_bubble(rc=60 * 0.4 - 0.5)
    robust = res.provenance["steep_r_circle"]
    assert abs(res.radius - robust) < 0.01 * robust
    assert "clipped_not_round" not in res.flags


def test_the_circle_confidence_judges_the_reported_circle():
    res, _ = _run_bubble(rc=60 * 0.0 - 0.5)                                    # half the bubble cut
    circle = res.provenance["circle"]
    cfg = QualityConfig()
    assert 0.45 < circle["hidden_arc"] < 0.55
    assert res.provenance["confidence_terms"]["completion"] == pytest.approx(
        min(1.0, (1 - circle["hidden_arc"]) / (1 - cfg.completion_free)), abs=1e-3)
    assert res.radius_sigma >= circle["sigma_fit"] - 1e-4                      # max(fit, outline noise)


# ------------------------------------------------------------------ fallback and provenance

def test_with_too_few_outline_points_the_robust_ray_circle_stands_in():
    img, ref, _ = _bubble()
    cfg = replace(DEFAULT_CONFIG, fit=replace(DEFAULT_CONFIG.fit, outline_min_points=10 ** 6))
    res = detect_edge_with_config(img, *FULL, reference=ref, params=OFF, config=cfg)
    circle = res.provenance["circle"]
    assert circle["fallback"] == "too few outline points" and circle["method"] == res.provenance["fit"]
    assert res.radius == res.provenance["steep_r_circle"]
    assert (res.rc, res.cc) == pytest.approx(res.provenance["steep_centre"], abs=1e-9)


def test_the_provenance_names_the_circle():
    res, _ = _run_bubble(aspect=1.5, angle=np.pi / 2)
    circle = res.provenance["circle"]
    assert circle["method"] == "taubin+trim(outline)" and circle["fallback"] is None
    assert circle["points"] == len(res.edge_xy) and 0.9 < circle["kept"] <= 1.0
    assert tuple(circle["centre"]) == pytest.approx((res.rc, res.cc), abs=1e-3)
    assert circle["radius"] == pytest.approx(res.radius, abs=1e-3)
    assert res.provenance["engine"] == "edge-1.3.0"


# ------------------------------------------------------------------ every consumer gets the same circle

def test_a_guess_below_the_gate_is_the_same_circle():
    img, ref, _ = _bubble(aspect=1.7, angle=np.pi / 2)
    shown = detect_edge(img, *FULL, reference=ref)
    strict = EdgeParams(confidence_gate=1.0)
    low = detect_edge(img, *FULL, reference=ref, params=strict)
    assert not low.accepted and (low.radius, low.rc, low.cc) == (shown.radius, shown.rc, shown.cc)
    guess = fit_from_result(low, EdgeJob(strict, ref, 400.0)).suggestion
    assert (guess.radius, guess.rc, guess.cc) == pytest.approx((shown.radius, shown.rc, shown.cc))
    assert (shown.rc, shown.cc) == pytest.approx((126.0, 201.0), abs=1.0)


def test_a_dragged_circle_measures_the_same_circle():
    img, ref, _ = _bubble(aspect=1.7, angle=np.pi / 2)
    fit, verdict = fit_guided(img, *FULL, EdgeJob(OFF, ref, 400.0), drawn_circle(126.0, 201.0, 60.0))
    assert verdict.accepted
    assert (fit.rc, fit.cc) == pytest.approx((126.0, 201.0), abs=1.0)
    assert fit.radius == pytest.approx(62.9, abs=0.3)


def _soft_ellipse(rc: float, cc: float, a: float, b: float, shape=(220, 260)) -> np.ndarray:
    """P(bubble) of an ellipse with half-axes *a* (rows) and *b* (columns)."""
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    d = np.hypot((yy - rc) / a, (xx - cc) / b)
    return (0.5 * (1.0 + np.tanh((1.0 - d) * min(a, b)))).astype(np.float32)


def test_ml_assists_own_outline_gets_the_same_circle():
    """ML's outline of a 1.7:1 region: the same centred least-squares circle as Auto's."""
    from bubbletrack.model.teach import MeasureConfig, ml_region

    a, b = 60.0 * np.sqrt(1.7), 60.0 / np.sqrt(1.7)
    reg = ml_region(_soft_ellipse(110.0, 130.0, a, b), config=MeasureConfig(refine=False))
    assert (reg.rc, reg.cc) == pytest.approx((111.0, 131.0), abs=1.0)          # origin (1, 1)
    assert 0 < reg.radius - reg.r_area <= 0.06 * reg.r_area
