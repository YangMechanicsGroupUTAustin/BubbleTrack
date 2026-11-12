"""Stage G orchestration: the bubble region at its edge level, with its outline evidence.

Edge definition A is the 50 % crossing between each ray's local dark and
bright plateaus.  The region is traced on the same image at the same levels:
the per-ray crossing levels, smoothed by a running median over
``RegionConfig.level_window_deg`` of polar angle and interpolated around the
circle centre, form a level map; the outline is the zero crossing of
``image - level map``.  On a round bubble it runs through the ray edge points;
where the wall is locally fainter it still follows the local 50 % point.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from bubbletrack.model.edge.body import edge_rim
from bubbletrack.model.edge.config import EdgeConfig
from bubbletrack.model.edge.quality import FitStats, OutlineEvidence, outline_evidence, radius_sigma
from bubbletrack.model.edge.region import Region, find_region

_NOISE_SUBSAMPLE = 3
_MIN_NOISE_PIXELS = 20
_ANGLE_BINS = 720
_MAP_STEP = 5
"""The level map is computed every _MAP_STEP pixels (odd: nodes fall on pixel centres) and interpolated."""


@dataclass(frozen=True)
class RayPoints:
    """Edge points of one definition: positions, crossing levels and ray indices."""

    points: np.ndarray
    levels: np.ndarray
    index: np.ndarray
    n_rays: int


@dataclass(frozen=True)
class Outline:
    """The bubble at one edge definition: its region, outline evidence and circle fit."""

    region: Region
    evidence: OutlineEvidence
    sigma_r: float
    stats: FitStats
    """The robust circle fit of the same definition (``r_circle``; completes clipped regions)."""
    level: float
    """Median of the per-ray crossing levels."""


def background_noise(img: np.ndarray, background: np.ndarray) -> float:
    """Robust spread of *img* over background pixels (its noise at its own smoothing scale)."""
    sample = img[::_NOISE_SUBSAMPLE, ::_NOISE_SUBSAMPLE][background[::_NOISE_SUBSAMPLE, ::_NOISE_SUBSAMPLE]]
    if sample.size < _MIN_NOISE_PIXELS:
        sample = img.ravel()
    return max(float(1.4826 * np.median(np.abs(sample - np.median(sample)))), 1e-4)


def level_map(shape: tuple[int, int], points: np.ndarray, levels: np.ndarray, centre: tuple[float, float],
              window_deg: float) -> np.ndarray | float:
    """Edge level at every pixel: per-ray levels smoothed over angle about *centre*.

    A window of 0 (or too few rays) gives the global median.
    """
    n = len(levels)
    k = int(round(n * window_deg / 360.0)) | 1
    if window_deg <= 0 or k < 3 or n < 2 * k:
        return float(np.median(levels))
    th = np.mod(np.arctan2(points[:, 0] - centre[0], points[:, 1] - centre[1]), 2 * np.pi)
    order = np.argsort(th)
    th, lv = th[order], levels[order]
    ext = np.concatenate([lv[-(k // 2):], lv, lv[:k // 2]])
    smooth = np.median(sliding_window_view(ext, k), axis=1)
    grid = (np.arange(_ANGLE_BINS) + 0.5) * (2 * np.pi / _ANGLE_BINS)
    table = np.interp(grid, th, smooth, period=2 * np.pi).astype(np.float32)
    h, w = shape
    f = _MAP_STEP
    yy, xx = np.mgrid[0:h + f:f, 0:w + f:f].astype(np.float32)
    phase = cv2.phase(xx - np.float32(centre[1]), yy - np.float32(centre[0]))
    coarse = table[(phase * (_ANGLE_BINS / (2 * np.pi))).astype(np.int32) % _ANGLE_BINS]
    # the level varies slowly with angle: interpolate a coarse grid (exact at its nodes)
    return cv2.resize(coarse, (coarse.shape[1] * f, coarse.shape[0] * f), interpolation=cv2.INTER_LINEAR)[
        (f - 1) // 2:(f - 1) // 2 + h, (f - 1) // 2:(f - 1) // 2 + w]


def measure_outline(img: np.ndarray, rays: RayPoints, exclude: np.ndarray,
                    candidate: tuple[float, float], preferred: list[tuple[str, tuple[float, float]]],
                    stats: FitStats, *, min_step: float, noise: float, blur: float, config: EdgeConfig,
                    allowed: np.ndarray | None = None) -> Outline | None:
    """The main bubble region of *img* at its edge levels, or None if there is none.

    Parameters
    ----------
    img : ndarray
        The image the levels were measured on (background ~1, bubble dark).
    rays : RayPoints
        The ray edge points of this definition, with their levels and ray indices.
    exclude : ndarray of bool
        Pixels without usable data.
    candidate : (row, col)
        A dark point of the selected candidate.
    preferred : list of (name, (row, col))
        Points that choose the main body before the candidate (prior centre, seed).
    stats : FitStats
        Circle fit of the same edge definition.
    min_step : float
        Smallest step across the outline that counts as an edge.
    noise, blur : float
        Noise of *img* and its smoothing scale (px), for the outline precision.
    config : EdgeConfig
    allowed : ndarray of bool, optional
        Pixels the region may occupy.
    """
    if len(rays.levels) == 0:
        return None
    fit = stats.fit
    centre = (fit.rc, fit.cc)
    lmap = level_map(img.shape, rays.points, rays.levels, centre, config.region.level_window_deg)
    inl = fit.weights > config.fit.inlier_weight
    rim = edge_rim(img.shape, rays.points[inl], rays.index[inl], rays.n_rays, centre, config.region.rim_inset_px)
    region = find_region(img - lmap, 0.0, exclude, candidate, fit.circle, config.region,
                         search_px=0.5 * fit.radius + config.region.crop_margin_px + 1, preferred=preferred,
                         allowed=allowed, rim=rim)
    if region is None:
        return None
    evidence = outline_evidence(img, region, min_step, noise, blur, config.region)
    return Outline(region, evidence, radius_sigma(region, evidence, fit.radius_sigma), stats,
                   float(np.median(rays.levels)))
