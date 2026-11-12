"""Stage F: circle-fit statistics, outline evidence, confidence terms and flags.

Confidence is a product of six terms in [0, 1] that judge the OUTLINE, not the
shape: how much of it is a real step (support), how precisely noise lets it be
located (precision), how much of the region had to be extrapolated past the
ROI border (completion), the frame SNR, a rival candidate (selection) and a
double edge (edge_unique).  A clean kidney-shaped bubble is as confident as a
round one.  Flags name the conditions a user should know about.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from bubbletrack.model.edge.config import FitConfig, QualityConfig, RegionConfig
from bubbletrack.model.edge.fit import CircleFit, arc_stats
from bubbletrack.model.edge.geometry import outward_normals
from bubbletrack.model.edge.region import Region

_MAX_POSITION_SIGMA_PX = 2.0


@dataclass(frozen=True)
class FitStats:
    """Geometry statistics of one fitted circle (steep or outer definition)."""

    fit: CircleFit
    inliers: np.ndarray
    """(K, 2) inlier points [row, col] in the fitting frame."""
    arc_coverage: float
    max_gap_deg: float
    inlier_fraction: float
    rms_px: float


def fit_stats(points: np.ndarray, fit: CircleFit, fit_cfg: FitConfig) -> FitStats:
    """Inliers, arc coverage and residual RMS of a robust circle fit."""
    inl = fit.weights > fit_cfg.inlier_weight
    pts_in = points[inl]
    coverage, gap = arc_stats(pts_in, fit.rc, fit.cc, fit_cfg.arc_bins)
    res = np.hypot(pts_in[:, 0] - fit.rc, pts_in[:, 1] - fit.cc) - fit.radius
    rms = float(np.sqrt(np.mean(res ** 2))) if len(res) else float("nan")
    return FitStats(fit=fit, inliers=pts_in, arc_coverage=coverage, max_gap_deg=gap,
                    inlier_fraction=float(inl.sum() / max(len(points), 1)), rms_px=rms)


@dataclass(frozen=True)
class OutlineEvidence:
    """How well the genuine part of a region's outline is supported by the image."""

    support: float
    """Length fraction of the outline across which the image steps by at least the minimum step."""
    sigma_radius: float
    """Standard error (px) of the area radius due to noise along the outline."""


def outline_evidence(img: np.ndarray, region: Region, min_step: float, noise: float, blur: float,
                     cfg: RegionConfig) -> OutlineEvidence:
    """Step amplitude across, and noise-limited position of, every genuine outline vertex.

    The step is ``img(v + a n) - img(v - a n)`` with ``a = cfg.step_px`` along the
    outward normal ``n``.  A vertex's position error is ``noise / gradient``
    (capped at 2 px); errors are correlated over the blur scale, which sets
    the effective number of independent vertices.
    """
    keep = region.edge_mask
    pts, lengths = region.contour[keep], region.edge_length
    total = float(lengths.sum())
    if len(pts) == 0 or total <= 0:
        return OutlineEvidence(0.0, float("inf"))
    n = outward_normals(region.contour)[keep]
    a = cfg.step_px
    out = ndimage.map_coordinates(img, (pts + a * n).T, order=1, mode="nearest")
    inn = ndimage.map_coordinates(img, (pts - a * n).T, order=1, mode="nearest")
    amp = out - inn
    support = float(lengths[amp >= min_step].sum() / total)
    grad = np.maximum(amp / (2 * a), noise / _MAX_POSITION_SIGMA_PX)
    corr = max(1.0, 2.0 * np.sqrt(np.pi) * blur)
    sigma_area = float(np.sqrt(np.sum(lengths * (noise / grad) ** 2) * corr))
    return OutlineEvidence(support, sigma_area / (2 * np.pi * max(region.radius, 1e-9)))


def radius_sigma(region: Region, evidence: OutlineEvidence, circle_sigma: float) -> float:
    """Standard error of the area radius: outline noise plus the extrapolated (hidden) part."""
    hidden = region.hidden_fraction if region.completed else 0.0
    extra = hidden * circle_sigma if hidden > 0 else 0.0
    if not np.isfinite(extra):
        extra = hidden * region.radius
    return float(np.hypot(evidence.sigma_radius, extra))


def confidence_terms(region: Region, evidence: OutlineEvidence, sigma_r: float, contrast: float,
                     noise: float, runner_up: float, multi_fraction: float,
                     cfg: QualityConfig) -> dict[str, float]:
    """The six confidence terms, each in [0, 1]."""
    rel = sigma_r / region.radius if np.isfinite(sigma_r) and region.radius > 0 else 1.0
    hidden = region.hidden_fraction if region.completed else 0.0
    return {
        "support": evidence.support ** 2,                                        # outline is a real edge
        "precision": float(np.exp(-rel / cfg.precision_scale)),                 # noise along the outline
        "completion": float(np.clip((1.0 - hidden) / (1.0 - cfg.completion_free), 0, 1)),  # extrapolation
        "snr": float(np.clip((contrast / max(noise, 1e-4) - cfg.snr_floor) / cfg.snr_span, 0, 1)),
        "selection": float(np.sqrt(max(0.0, 1.0 - min(runner_up, 1.0)))),       # rival candidate
        "edge_unique": 1.0 - 0.5 * min(1.0, multi_fraction / cfg.multi_scale),  # halo / double ring
    }


def confidence(terms: dict[str, float]) -> float:
    return float(np.clip(np.prod(list(terms.values())), 0.0, 1.0))


def is_non_circular(r_circle: float, r_area: float, solidity: float, cfg: QualityConfig) -> bool:
    off_circle = (np.isfinite(r_circle) and r_area > 0
                  and abs(r_circle - r_area) / r_area > cfg.non_circular_rel)
    return bool(off_circle or (np.isfinite(solidity) and solidity < cfg.non_circular_solidity))


def measurement_flags(*, region: Region, non_circular: bool, multi_fraction: float, runner_up: float,
                      contrast: float, noise: float, saturated: bool, shoulder_width: float,
                      cfg: QualityConfig) -> set[str]:
    """Flags describing one successful measurement."""
    flags: set[str] = set()
    if region.clipped:
        flags.add("clipped")
    if region.completed:
        flags.add("clipped_completed")
    if region.fragment_count > 0:
        flags.add("fragmented")
    if non_circular:
        flags.add("non_circular")
    if multi_fraction > cfg.multi_edge_flag:
        flags.add("multiple_edges")
    if runner_up > cfg.ambiguous_flag:
        flags.add("ambiguous_selection")
    if contrast < cfg.low_snr_ratio * noise:
        flags.add("low_snr")
    if saturated:
        flags.add("saturation")
    if np.isfinite(shoulder_width) and shoulder_width > cfg.shoulder_flag_px:
        flags.add("shoulder")
    return flags
