"""Stage F: the radius under the user's definition, its flags and its confidence terms.

Circle (default, ``EdgeParams.non_round`` False; owner decision 2026-11-13)
    R and the centre are the conventional least-squares circle through the
    bubble's measured outline: Taubin's circle through the genuine outline points
    (not on the ROI border, not next to excluded pixels), refitted once without
    the points more than 3 robust sigmas off (``Outline.circle``, ``r_circle``),
    on every frame.  The robust ray fit stands in only when fewer than
    ``FitConfig.outline_min_points`` outline points remain or that fit fails
    (``provenance["circle"]["fallback"]``).  A bubble cut by the ROI border is
    fitted on its visible outline.  Independent checks that the reported circle
    cannot fool (``shape_check``): inside the ROI, ``non_circular`` when it differs
    from the outline's area-equivalent radius (the other definition's R) by more
    than ``QualityConfig.circle_check_rel``; cut by the border,
    ``clipped_not_round`` when the visible wall completed with an ellipse OR the
    robust ray circle disagrees with it.  The confidence judges the circle: its
    fit precision, the hidden share of its circumference, and how far the check is
    off (``roundness``).

Area ("Bubble may be non-round")
    R is the area-equivalent radius of the outline, the centre the centroid of the same
    (completed) region.  A bubble cut by the border is completed with its robust ray
    circle only when the visible wall agrees (the ellipse check); otherwise it is
    ``clipped_unmeasurable``: no result, its candidate kept as Auto's guess.
    ``non_circular`` describes the outline only.

Every result of this mode carries ``area_radius`` (the definition, frame by frame).
Coordinates are 0-based ROI pixels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bubbletrack.model.edge.config import QualityConfig
from bubbletrack.model.edge.outline import Outline
from bubbletrack.model.edge.quality import area_terms, circle_terms, is_non_circular
from bubbletrack.model.edge.result import FLAG_UNMEASURABLE
from bubbletrack.model.edge.shape_check import CircleCheck
from bubbletrack.model.provenance import AREA_RADIUS_FLAG as FLAG_AREA
FLAG_NON_CIRCULAR = "non_circular"
FLAG_CLIPPED_NOT_ROUND = "clipped_not_round"
FLAG_COMPLETED = "clipped_completed"


@dataclass(frozen=True)
class FrameStats:
    """Frame-level inputs of the confidence (shared by both definitions)."""

    contrast: float
    noise: float
    runner_up: float
    multi_fraction: float


@dataclass(frozen=True)
class Measured:
    """The reported radius of one frame under one definition."""

    radius: float
    centre: tuple[float, float]
    sigma: float
    kind: str
    flags: frozenset[str]
    terms: dict[str, float]
    check: float
    """|R_circle - R_estimate| / R_estimate of the circle check (NaN when not made)."""


def check_off(r_circle: float, check: CircleCheck) -> float:
    """Relative difference of the circle from the area estimate (NaN when unknown)."""
    est = check.r_estimate
    if not (np.isfinite(est) and est > 0 and np.isfinite(r_circle)):
        return float("nan")
    return float(abs(r_circle - est) / est)


def _disagrees(radius: float, probe: float, cfg: QualityConfig) -> bool:
    """*radius* is off the clipped-bubble *probe* beyond ``max(clipped_check_rel * probe,
    clipped_check_px)`` (an unknown probe: no)."""
    if not (np.isfinite(probe) and probe > 0):
        return False
    return abs(radius - probe) > max(cfg.clipped_check_rel * probe, cfg.clipped_check_px)


def clipped_disagrees(radius: float, check: CircleCheck, cfg: QualityConfig) -> bool:
    """A clipped region whose ellipse-completed area is off *radius* beyond the clipped
    tolerance (unknown: no)."""
    return check.method == "ellipse" and _disagrees(radius, check.r_estimate, cfg)


def circle_sigma(line: Outline) -> float:
    """Standard error of the reported circle's radius: its fit covariance (a short visible arc
    makes it large), but never below the noise-limited position of the outline it runs through.

    Both describe the same wall noise, so the larger one counts.  A covariance alone (of the
    robust fit's inliers, edge-1.2.0) passed far more wrong circles on textured frames
    (labelled lic_hard 01: 18 wrong accepted instead of 0).
    """
    fit_sigma = line.circle.fit.radius_sigma
    return float(max(fit_sigma if np.isfinite(fit_sigma) else 0.0, line.evidence.sigma_radius))


def circle_flags(radius: float, off: float, check: CircleCheck, robust_radius: float,
                 cfg: QualityConfig) -> set[str]:
    """The default definition's review flags for a reported circle of *radius*.

    Inside the ROI (``check.method == "area"``): ``non_circular`` when it is more than
    ``circle_check_rel`` off the area-equivalent radius (*off*).  Cut by the border: two
    probes, ``clipped_not_round`` when the ellipse-completed area or the robust ray circle
    (*robust_radius*) disagrees with it.  A dented bubble can fool the ellipse together with
    the outline circle (a kidney cut by 30 %: both -11 %), not the robust circle.
    """
    if check.method == "area":
        return {FLAG_NON_CIRCULAR} if np.isfinite(off) and off > cfg.circle_check_rel else set()
    if clipped_disagrees(radius, check, cfg) or _disagrees(radius, robust_radius, cfg):
        return {FLAG_CLIPPED_NOT_ROUND}
    return set()


def measure_circle(line: Outline, check: CircleCheck, frame: FrameStats, cfg: QualityConfig) -> Measured:
    """Default definition: the least-squares circle through the outline (``line.circle``)."""
    circle = line.circle
    fit = circle.fit
    radius = float(fit.radius)
    sigma = circle_sigma(line)
    off = check_off(radius, check)
    flags = circle_flags(radius, off, check, float(line.stats.fit.radius), cfg)
    terms = circle_terms(line.evidence, radius, sigma, circle.hidden_arc, off, frame.contrast,
                         frame.noise, frame.runner_up, frame.multi_fraction, cfg)
    return Measured(radius, (fit.rc, fit.cc), float(sigma), "circle", frozenset(flags), terms, off)


def measure_area(line: Outline, check: CircleCheck, frame: FrameStats, cfg: QualityConfig) -> Measured:
    """"Bubble may be non-round": the area-equivalent radius of the outline."""
    region, fit = line.region, line.stats.fit
    flags = {FLAG_AREA}
    if is_non_circular(fit.radius, region.radius, region.solidity, cfg):
        flags.add(FLAG_NON_CIRCULAR)
    if region.completed:
        flags.add(FLAG_COMPLETED)
    if clipped_disagrees(region.radius, check, cfg):
        flags.add(FLAG_UNMEASURABLE)
    terms = area_terms(region, line.evidence, line.sigma_r, frame.contrast, frame.noise,
                       frame.runner_up, frame.multi_fraction, cfg)
    kind = "area+circle_completion" if region.completed else "area"
    return Measured(region.radius, region.centroid, line.sigma_r, kind, frozenset(flags), terms,
                    check_off(fit.radius, check))


def measure(line: Outline, check: CircleCheck, frame: FrameStats, cfg: QualityConfig, *,
            non_round: bool) -> Measured:
    return (measure_area if non_round else measure_circle)(line, check, frame, cfg)
