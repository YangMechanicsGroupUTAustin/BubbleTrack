"""The two radius definitions (owner decisions 2026-11-13, "Bubble may be non-round").

Off (default, ``EdgeParams(non_round=False)``): R is the robust circle through the bubble
wall, the centre its centre.  The area-equivalent radius is kept as an independent check:
a bubble inside the ROI whose circle and area radii differ by more than 5 % is flagged
``non_circular`` (Review: worth a check); a bubble cut by the ROI border is compared with its
visible area completed by an ellipse fitted to its wall (flag ``clipped_not_round``).

On (``non_round=True``): R is the area-equivalent radius of the outline, the centre its
centroid; a bubble cut by the border is completed with its circle only when the visible wall
agrees with it, else it gets no result (flag ``clipped_unmeasurable``, Auto's guess kept).

Clipped shapes are the demo's (``clip_shapes_demo.py``): R_eq = 60, cut by the top edge
so that 10 / 30 / 50 % of the diameter is hidden.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pytest

from _edge_shapes import H, SHAPES, W, reference, render_shape
from _edge_synth import Bubble, Scene, render, truth
from bubbletrack.model.edge import EdgeParams, detect_edge
from bubbletrack.model.edge.config import QualityConfig

FULL = ((1, H), (1, W))
R = 60.0
OFF, ON = EdgeParams(), EdgeParams(non_round=True)
ELLIPSES = {"along": dict(aspect=1.5, angle=0.0), "across": dict(aspect=1.5, angle=np.pi / 2)}
LOBES = dict(wobble=((3, 0.08, 0.0),))
CUTS = (0.1, 0.3, 0.5)
REVIEW_FLAGS = {"clipped_not_round", "clipped_unmeasurable", "non_circular"}


@lru_cache(maxsize=None)
def _scene(cut: float, **shape):
    b = Bubble(rc=R * (1 - 2 * cut) if cut else 125.0, cc=200.0, r=R, **shape)
    img, meta = render(Scene(bubbles=(b,)))
    return img, meta["ref"], truth(b)["r_eq"]


def _measure(cut: float, params: EdgeParams, **shape):
    img, ref, r_eq = _scene(cut, **shape)
    return detect_edge(img, *FULL, reference=ref, params=params), r_eq


def _shape_key(kw: dict) -> dict:
    return {k: (tuple(v) if isinstance(v, list) else v) for k, v in kw.items()}


# ------------------------------------------------------------------ the default: the circle

def test_default_is_the_circle_definition():
    assert EdgeParams().non_round is False
    res, _ = _measure(0.0, OFF, aspect=1.2)
    assert res.radius == res.r_circle and res.radius_kind == "circle"
    centre = res.provenance["steep_centre"]
    assert (res.rc, res.cc) == pytest.approx(centre, abs=1e-9)          # the circle's centre
    assert res.r_area == pytest.approx(np.sqrt(res.area_px / np.pi))    # the area is still measured
    assert "area_radius" not in res.flags


def test_non_round_mode_reports_the_area_and_marks_it():
    res, r_eq = _measure(0.0, ON, aspect=1.5)
    assert res.radius == res.r_area and res.radius_kind == "area"
    assert res.radius == pytest.approx(r_eq, rel=0.01)
    assert (res.rc, res.cc) == pytest.approx(res.centroid, abs=1e-9)
    assert "area_radius" in res.flags


def test_non_round_must_be_a_bool():
    with pytest.raises(ValueError):
        EdgeParams(non_round=1)


@pytest.mark.parametrize("shape", SHAPES, ids=[s.name for s in SHAPES])
def test_non_circular_flags_a_circle_more_than_5_percent_off_the_area(shape):
    img, _ = render_shape(shape)
    res = detect_edge(img, *FULL, reference=reference(), params=EdgeParams(seed=(shape.seed[0] + 1,
                                                                                  shape.seed[1] + 1)))
    rel = abs(res.r_circle - res.r_area) / res.r_area
    assert QualityConfig().circle_check_rel == 0.05
    assert ("non_circular" in res.flags) == (rel > 0.05), (shape.name, rel)
    assert res.radius == res.r_circle and res.accepted


def test_a_misfit_lowers_the_confidence_but_keeps_the_result():
    """The robust circle of this 1.5:1 ellipse runs along one flat side (+57 %); rotated by 90
    degrees it runs through the middle (-3 %, within 5 %: no flag)."""
    round_, _ = _measure(0.0, OFF)
    oval, r_eq = _measure(0.0, OFF, **ELLIPSES["across"])
    assert oval.radius > 1.3 * r_eq
    assert "non_circular" in oval.flags and oval.accepted
    assert oval.confidence < round_.confidence - 0.2
    assert oval.provenance["confidence_terms"]["roundness"] < 1.0
    assert round_.provenance["confidence_terms"]["roundness"] == 1.0


def test_the_circle_confidence_terms():
    res, _ = _measure(0.5, OFF)
    terms = res.provenance["confidence_terms"]
    assert set(terms) == {"support", "precision", "completion", "snr", "selection", "edge_unique",
                          "roundness"}
    on, _ = _measure(0.5, ON)
    assert set(on.provenance["confidence_terms"]) == {"support", "precision", "completion", "snr",
                                                      "selection", "edge_unique"}


def test_the_circle_extrapolation_comes_from_the_hidden_arc():
    """75 % of the diameter cut: ~67 % of the circumference but ~80 % of the area is hidden."""
    img, ref, _ = _clipped(0.75)
    off = detect_edge(img, *FULL, reference=ref)
    on = detect_edge(img, *FULL, reference=ref, params=ON)
    arc = off.provenance["region"]["hidden_arc"]
    assert 0.6 < arc < 0.72
    cfg = QualityConfig()
    assert off.provenance["confidence_terms"]["completion"] == pytest.approx(
        (1 - arc) / (1 - cfg.completion_free), abs=1e-3)
    assert off.provenance["confidence_terms"]["completion"] > on.provenance["confidence_terms"]["completion"]


@lru_cache(maxsize=None)
def _clipped(cut: float):
    return _scene(cut)


# ------------------------------------------------------------------ round and lobed bubbles cut by the border

@pytest.mark.parametrize("params", [OFF, ON], ids=["circle", "area"])
@pytest.mark.parametrize("cut", CUTS)
def test_round_clipped_bubbles_measure_within_half_a_percent(cut, params):
    res, r_eq = _measure(cut, params)
    assert res.accepted and res.radius == pytest.approx(r_eq, rel=0.005)
    assert "clipped" in res.flags and not REVIEW_FLAGS & res.flags
    assert res.provenance["circle_check"]["method"] == "ellipse"


@pytest.mark.parametrize("params", [OFF, ON], ids=["circle", "area"])
@pytest.mark.parametrize("cut", (0.0, *CUTS))
def test_three_lobes_of_8_percent_are_still_fine(cut, params):
    res, r_eq = _measure(cut, params, **_shape_key(LOBES))
    assert res.accepted and res.radius == pytest.approx(r_eq, rel=0.01)
    assert not REVIEW_FLAGS & res.flags, res.flags


def test_only_the_area_mode_completes_with_the_circle():
    off, _ = _measure(0.3, OFF)
    on, _ = _measure(0.3, ON)
    assert "clipped_completed" not in off.flags and off.radius_kind == "circle"
    assert "clipped_completed" in on.flags and on.radius_kind == "area+circle_completion"


# ------------------------------------------------------------------ non-round bubbles cut by the border

@pytest.mark.parametrize("cut", CUTS)
@pytest.mark.parametrize("orientation", sorted(ELLIPSES))
def test_clipped_ellipse_in_circle_mode_goes_to_review(orientation, cut):
    """The circle through the visible wall is 14-58 % off on these: never silently."""
    res, r_eq = _measure(cut, OFF, **ELLIPSES[orientation])
    error = abs(res.radius / r_eq - 1)
    assert error <= 0.05 or not res.accepted or "clipped_not_round" in res.flags, (error, res.flags)
    assert "clipped_not_round" in res.flags          # here: all of them are off by more than 5 %
    assert res.radius == res.r_circle


@pytest.mark.parametrize("cut", CUTS)
@pytest.mark.parametrize("orientation", sorted(ELLIPSES))
def test_clipped_ellipse_in_area_mode_gets_no_result_but_a_guess(orientation, cut):
    res, r_eq = _measure(cut, ON, **ELLIPSES[orientation])
    error = abs(res.radius / r_eq - 1)
    assert error <= 0.05 or not res.accepted, (error, res.flags)
    assert "clipped_unmeasurable" in res.flags and not res.accepted
    assert np.isfinite(res.radius) and res.radius > 0 and res.confidence > 0   # Auto's guess
    assert res.radius == res.r_area and "area_radius" in res.flags


def test_area_mode_non_circular_is_information_only():
    """The flag still describes the outline; whether it lists a frame is Review's business
    (``area_radius`` marks the definition)."""
    res, _ = _measure(0.0, ON, aspect=1.5)
    assert {"non_circular", "area_radius"} <= res.flags and res.accepted


# ------------------------------------------------------------------ below the confidence gate

def test_below_the_gate_each_mode_keeps_its_definition():
    """A no-bubble texture frame: rejected in both modes; its guess is the circle (off) or the
    area of the drawn outline (on), so Accept stores what is shown."""
    img, meta = render(Scene(static=1))
    off = detect_edge(img, *FULL)
    on = detect_edge(img, *FULL, params=ON)
    assert not off.accepted and not on.accepted
    assert off.radius == off.r_circle and off.radius_kind == "circle"
    assert on.radius == on.r_area and on.radius_kind in ("area", "area+circle_completion")
    assert "circle_fallback" not in off.flags | on.flags
