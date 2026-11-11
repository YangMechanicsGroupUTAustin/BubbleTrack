"""Stage F: per-measurement statistics, confidence terms and flags.

Confidence is a product of six terms in [0, 1]; any single weak term (short
arc, many outliers, imprecise fit, low SNR, rival candidate, double edge)
pulls it down.  Flags name the conditions a user should know about.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bubbletrack.model.edge.config import FitConfig, QualityConfig
from bubbletrack.model.edge.fit import CircleFit, arc_stats, polygon_area


@dataclass(frozen=True)
class FitStats:
    """Geometry statistics of one fitted edge (steep or outer definition)."""

    fit: CircleFit
    inliers: np.ndarray
    """(K, 2) inlier points [row, col] in the fitting frame."""
    arc_coverage: float
    max_gap_deg: float
    inlier_fraction: float
    rms_px: float
    r_area: float
    """Area-equivalent radius; NaN unless the edge is complete and unclipped."""


def fit_stats(points: np.ndarray, fit: CircleFit, cfg: QualityConfig, fit_cfg: FitConfig, *,
              clipped: bool) -> FitStats:
    """Inliers, arc coverage, residual RMS and (for complete arcs) the area-equivalent radius."""
    inl = fit.weights > fit_cfg.inlier_weight
    pts_in = points[inl]
    coverage, gap = arc_stats(pts_in, fit.rc, fit.cc, fit_cfg.arc_bins)
    res = np.hypot(pts_in[:, 0] - fit.rc, pts_in[:, 1] - fit.cc) - fit.radius
    rms = float(np.sqrt(np.mean(res ** 2))) if len(res) else float("nan")
    r_area = float("nan")
    if not clipped and gap < cfg.complete_arc_gap_deg:
        r_area = float(np.sqrt(polygon_area(points, fit.rc, fit.cc) / np.pi))
    return FitStats(fit=fit, inliers=pts_in, arc_coverage=coverage, max_gap_deg=gap,
                    inlier_fraction=float(inl.sum() / max(len(points), 1)), rms_px=rms, r_area=r_area)


def confidence_terms(stats: FitStats, contrast: float, noise: float, runner_up: float,
                     multi_fraction: float, cfg: QualityConfig) -> dict[str, float]:
    """The six confidence terms, each in [0, 1]."""
    radius = stats.fit.radius
    sig_r = stats.fit.radius_sigma
    rel = sig_r / radius if np.isfinite(sig_r) and radius > 0 else 1.0
    return {
        "arc": min(1.0, stats.arc_coverage / cfg.arc_full),                    # extrapolation risk
        "inliers": stats.inlier_fraction ** 2,                                  # contamination
        "precision": float(np.exp(-rel / cfg.precision_scale)),                 # fit standard error
        "snr": float(np.clip((contrast / max(noise, 1e-4) - cfg.snr_floor) / cfg.snr_span, 0, 1)),
        "selection": float(np.sqrt(max(0.0, 1.0 - min(runner_up, 1.0)))),       # rival candidate
        "edge_unique": 1.0 - 0.5 * min(1.0, multi_fraction / cfg.multi_scale),  # halo / double ring
    }


def confidence(terms: dict[str, float]) -> float:
    return float(np.clip(np.prod(list(terms.values())), 0.0, 1.0))


def measurement_flags(*, clipped: bool, multi_fraction: float, runner_up: float, contrast: float,
                      noise: float, saturated: bool, stats: FitStats, shoulder_width: float,
                      cfg: QualityConfig) -> set[str]:
    """Flags describing one successful measurement."""
    flags: set[str] = set()
    if clipped:
        flags.add("clipped")
    if multi_fraction > cfg.multi_edge_flag:
        flags.add("multiple_edges")
    if runner_up > cfg.ambiguous_flag:
        flags.add("ambiguous_selection")
    if contrast < cfg.low_snr_ratio * noise:
        flags.add("low_snr")
    if saturated:
        flags.add("saturation")
    radius = stats.fit.radius
    if np.isfinite(stats.r_area) and radius > 0 and abs(stats.r_area - radius) / radius > cfg.non_circular_rel:
        flags.add("non_circular")
    if np.isfinite(shoulder_width) and shoulder_width > cfg.shoulder_flag_px:
        flags.add("shoulder")
    return flags
