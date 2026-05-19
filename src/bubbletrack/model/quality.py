"""Fit quality scoring for auto-tune and real-time feedback."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bubbletrack.model.constants import (
    FIT_ACCEPT_MIN_EDGE_DENSITY,
    FIT_ACCEPT_MIN_SECTOR_COVERAGE,
    FIT_ACCEPT_PARTIAL_MIN_SECTOR_COVERAGE,
    QUALITY_MAX_RADIUS_FRACTION,
    QUALITY_MAX_RMS_RATIO,
    QUALITY_MIN_RADIUS_FRACTION,
    QUALITY_N_SECTORS,
    QUALITY_TARGET_EDGE_DENSITY,
)


@dataclass(frozen=True)
class FitQuality:
    """Quality metrics for a single circle fit."""

    score: float           # Composite [0, 1], higher = better
    n_edge_points: int     # Number of detected edge pixels
    edge_density: float    # Points per px of circumference
    rms_residual: float    # RMS distance from fitted circle (px)
    sector_coverage: float # Fraction of angular sectors with points [0, 1]
    radius: float          # Fitted radius (px)


def compute_fit_quality(
    edge_xy: np.ndarray,
    rc: float,
    cc: float,
    radius: float,
    roi_short_side: int,
) -> FitQuality:
    """Score a circle fit result.

    Parameters
    ----------
    edge_xy : (N, 2) array of [row, col] edge points (full-image coords).
    rc, cc  : Circle centre (row, col).
    radius  : Fitted radius in pixels.
    roi_short_side : min(ROI height, ROI width) for plausibility check.

    Returns
    -------
    FitQuality with composite score in [0, 1].
    """
    n = edge_xy.shape[0]
    circumference = 2.0 * np.pi * radius

    # 1. Edge density
    edge_density = n / circumference if circumference > 0 else 0.0
    density_score = min(edge_density / QUALITY_TARGET_EDGE_DENSITY, 1.0)

    # 2. RMS residual
    dists = np.linalg.norm(edge_xy - np.array([rc, cc]), axis=1)
    residuals = dists - radius
    rms = float(np.sqrt(np.mean(residuals ** 2)))
    rms_ratio = rms / radius if radius > 0 else 1.0
    rms_score = max(0.0, 1.0 - rms_ratio / QUALITY_MAX_RMS_RATIO)

    # 3. Angular sector coverage
    dy = edge_xy[:, 0] - rc
    dx = edge_xy[:, 1] - cc
    angles = np.arctan2(dy, dx)  # [-pi, pi]
    sector_edges = np.linspace(-np.pi, np.pi, QUALITY_N_SECTORS + 1)
    sectors_hit: set[int] = set()
    for a in angles:
        idx = int(np.searchsorted(sector_edges, a, side="right")) - 1
        idx = max(0, min(idx, QUALITY_N_SECTORS - 1))
        sectors_hit.add(idx)
    sector_coverage = len(sectors_hit) / QUALITY_N_SECTORS

    # 4. Radius plausibility
    if roi_short_side > 0:
        r_frac = radius / roi_short_side
        if QUALITY_MIN_RADIUS_FRACTION <= r_frac <= QUALITY_MAX_RADIUS_FRACTION:
            plausibility = 1.0
        else:
            plausibility = 0.0
    else:
        plausibility = 0.5

    # Composite score — weighted geometric mean.
    # Unlike an additive average, a geometric mean ensures that a single
    # terrible metric (e.g. RMS ≈ 0) pulls the whole score down sharply,
    # producing a much more discriminating quality distribution.
    _floor = 0.01  # avoid log(0); maps to score ≈ 0
    log_score = (
        0.25 * np.log(max(density_score, _floor))
        + 0.30 * np.log(max(rms_score, _floor))
        + 0.30 * np.log(max(sector_coverage, _floor))
        + 0.15 * np.log(max(plausibility, _floor))
    )
    score = float(np.exp(log_score))

    # A partial arc can fit a mathematically plausible circle with excellent
    # RMS, but its radius is mostly extrapolated.  Keep the result, but cap
    # confidence so the chart marks it as marginal/unreliable instead of green.
    if sector_coverage < 0.375:
        score = min(score, 0.30)
    elif sector_coverage < 0.625:
        score = min(score, 0.60)

    return FitQuality(
        score=round(score, 3),
        n_edge_points=n,
        edge_density=round(edge_density, 3),
        rms_residual=round(rms, 2),
        sector_coverage=round(sector_coverage, 3),
        radius=round(radius, 2),
    )


def compute_all_quality_scores(
    radius: np.ndarray,
    circle_fit_par: np.ndarray | None,
    circle_xy: list | None,
    gridx: tuple[int, int],
    gridy: tuple[int, int],
) -> np.ndarray:
    """Compute per-frame quality scores for all frames.

    Parameters
    ----------
    radius : 1-D array of fitted radii (-1 or NaN = no fit).
    circle_fit_par : (N, 2) array of [row_c, col_c] or None.
    circle_xy : list of (M, 2) edge-point arrays (or None entries).
    gridx, gridy : 1-indexed ROI bounds for plausibility check.

    Returns
    -------
    np.ndarray
        1-D array of scores in [0, 1].  Frames without a valid fit get 0.
    """
    n = len(radius)
    scores = np.zeros(n)
    roi_w = gridx[1] - gridx[0] + 1
    roi_h = gridy[1] - gridy[0] + 1
    # If ROI is trivially small (unconfigured default), pass 0 so
    # compute_fit_quality falls back to plausibility=0.5 (neutral).
    roi_short = min(roi_w, roi_h) if roi_w > 1 and roi_h > 1 else 0

    for i in range(n):
        if radius[i] <= 0 or not np.isfinite(radius[i]):
            continue
        if circle_xy is None or circle_xy[i] is None:
            continue
        edge_xy = circle_xy[i]
        if edge_xy.shape[0] < 3:
            continue
        if circle_fit_par is None or not np.isfinite(circle_fit_par[i]).all():
            continue
        q = compute_fit_quality(
            edge_xy,
            float(circle_fit_par[i, 0]),
            float(circle_fit_par[i, 1]),
            float(radius[i]),
            roi_short,
        )
        scores[i] = q.score

    return scores


def is_fit_acceptable(
    edge_xy: np.ndarray,
    rc: float,
    cc: float,
    radius: float,
    roi_short_side: int,
    *,
    allow_partial: bool = False,
) -> tuple[bool, FitQuality]:
    """Return whether a fitted circle is reliable enough to accept.

    This is intentionally stricter than display-time quality coloring.  A
    partial arc can have low RMS and therefore a plausible-looking circle, but
    its radius is extrapolated.  For automatic R(t) extraction we reject those
    incomplete fits so they do not create spurious spikes.
    """
    q = compute_fit_quality(edge_xy, rc, cc, radius, roi_short_side)
    min_coverage = (
        FIT_ACCEPT_PARTIAL_MIN_SECTOR_COVERAGE
        if allow_partial else FIT_ACCEPT_MIN_SECTOR_COVERAGE
    )
    ok = (
        q.sector_coverage >= min_coverage
        and q.edge_density >= FIT_ACCEPT_MIN_EDGE_DENSITY
    )
    return ok, q


def score_failed_fit() -> FitQuality:
    """Return a zero-score FitQuality for failed detections."""
    return FitQuality(
        score=0.0,
        n_edge_points=0,
        edge_density=0.0,
        rms_residual=float("inf"),
        sector_coverage=0.0,
        radius=0.0,
    )
