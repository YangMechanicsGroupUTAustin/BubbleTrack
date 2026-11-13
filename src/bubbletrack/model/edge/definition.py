"""Stage F: the radius under the user's definition, its flags and its confidence terms.

Circle (default, ``EdgeParams.non_round`` False)
    R is the robust circle fit through the genuine wall points (``r_circle``), the
    centre its centre.  A bubble cut by the ROI border is simply fitted on its visible
    arcs.  The area-equivalent radius is the independent check (``shape_check``):
    inside the ROI, ``non_circular`` when the two differ by more than
    ``QualityConfig.circle_check_rel``; cut by the border, ``clipped_not_round`` when
    the visible wall completed with an ellipse disagrees with the circle.  The
    confidence judges the circle: its fit precision, the hidden share of its
    circumference, and how far the check is off (``roundness``).

Area ("Bubble may be non-round")
    R is the area-equivalent radius of the outline, the centre its centroid.  A bubble
    cut by the border is completed with its circle only when the visible wall agrees
    (the same check); otherwise it is ``clipped_unmeasurable``: no result, its
    candidate kept as Auto's guess.  ``non_circular`` describes the outline only.

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


def clipped_disagrees(radius: float, check: CircleCheck, cfg: QualityConfig) -> bool:
    """A clipped region whose ellipse-completed area is off the circle's beyond
    ``max(clipped_check_rel * R, clipped_check_px)`` (unknown: no)."""
    if check.method != "ellipse" or not np.isfinite(check.r_estimate):
        return False
    return abs(radius - check.r_estimate) > max(cfg.clipped_check_rel * check.r_estimate,
                                                cfg.clipped_check_px)


def circle_sigma(line: Outline) -> float:
    """Standard error of the circle radius: its fit covariance (a short visible arc makes it
    large), but never below the noise-limited position of the outline it runs through.

    Both describe the same wall noise, so the larger one counts.  The covariance alone shrinks
    to the inliers the robust fit kept: on textured frames it passed far more wrong circles
    (labelled lic_hard 01: 18 wrong accepted instead of 0).
    """
    fit_sigma = line.stats.fit.radius_sigma
    return float(max(fit_sigma if np.isfinite(fit_sigma) else 0.0, line.evidence.sigma_radius))


def measure_circle(line: Outline, check: CircleCheck, frame: FrameStats, cfg: QualityConfig) -> Measured:
    """Default definition: the robust circle through the wall."""
    region, fit = line.region, line.stats.fit
    radius = float(fit.radius)
    sigma = circle_sigma(line)
    off = check_off(radius, check)
    flags = set()
    if region.clipped:
        if clipped_disagrees(radius, check, cfg):
            flags.add(FLAG_CLIPPED_NOT_ROUND)
    elif np.isfinite(off) and off > cfg.circle_check_rel:
        flags.add(FLAG_NON_CIRCULAR)
    terms = circle_terms(line.evidence, radius, sigma, region.hidden_arc, off, frame.contrast,
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
    if region.clipped and clipped_disagrees(region.radius, check, cfg):
        flags.add(FLAG_UNMEASURABLE)
    centre = (fit.rc, fit.cc) if region.clipped else region.visible.centroid
    terms = area_terms(region, line.evidence, line.sigma_r, frame.contrast, frame.noise,
                       frame.runner_up, frame.multi_fraction, cfg)
    kind = "area+circle_completion" if region.completed else "area"
    return Measured(region.radius, centre, line.sigma_r, kind, frozenset(flags), terms,
                    check_off(fit.radius, check))


def measure(line: Outline, check: CircleCheck, frame: FrameStats, cfg: QualityConfig, *,
            non_round: bool) -> Measured:
    return (measure_area if non_round else measure_circle)(line, check, frame, cfg)
