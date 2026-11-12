"""ML's own bubble region: the P >= 0.5 blob of the probability map.

Steps: smooth P lightly, threshold at ``level``, fill holes, choose the blob
(the one holding the prior / seed, else the nearest, else the biggest
confident one), optionally move its outline to the half-contrast wall of the
engine view nearby (the Edge definition of the wall), complete a blob cut by
the ROI border with its circle fit, and score it (:mod:`.confidence`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage
from skimage.measure import find_contours

from bubbletrack.model.circle_fit import circle_fit_taubin
from bubbletrack.model.teach.config import MeasureConfig
from bubbletrack.model.teach.confidence import region_confidence, region_terms

_MIN_RING_PX = 20
_MAX_REFINE_CHANGE = 0.3


def _empty_points() -> np.ndarray:
    return np.empty((0, 2))


@dataclass(frozen=True, eq=False)
class MLRegion:
    """ML's bubble region.  Coordinates are 1-based full-image ``(row, col)``."""

    radius: float
    """Area-equivalent radius (completed with the circle fit where the ROI border cuts it)."""
    rc: float
    cc: float
    confidence: float
    mask: np.ndarray
    """The region on the ROI grid (bool)."""
    clipped: bool
    refined: bool
    """True when the outline was moved to the half-contrast wall of the engine view."""
    terms: dict = field(default_factory=dict)
    contour: np.ndarray = field(default_factory=_empty_points)
    """(N, 2) closed outline of the visible region."""


def _local(point, origin: tuple[int, int]):
    if point is None:
        return None
    return (point[0] - origin[0], point[1] - origin[1]) + tuple(point[2:])


def _holds(labels: np.ndarray, point) -> int:
    r, c = int(round(point[0])), int(round(point[1]))
    if 0 <= r < labels.shape[0] and 0 <= c < labels.shape[1]:
        return int(labels[r, c])
    return 0


def _choose(labels: np.ndarray, valid: np.ndarray, areas: np.ndarray, centroids: np.ndarray,
            mean_p: np.ndarray, seed, prior) -> int:
    """Label of the chosen blob (1-based label index)."""
    ids = np.flatnonzero(valid) + 1
    radii = np.sqrt(areas / np.pi)
    for point in (prior, seed):
        if point is None:
            continue
        hit = _holds(labels, point)
        if hit and valid[hit - 1]:
            return hit
        d = np.hypot(centroids[ids - 1, 0] - point[0], centroids[ids - 1, 1] - point[1])
        if point is prior:
            pr = max(float(prior[2]), 1.0)
            score = d / pr + np.abs(radii[ids - 1] - pr) / pr
        else:
            score = np.maximum(d - radii[ids - 1], 0.0) - 1e-3 * radii[ids - 1]
        return int(ids[np.argmin(score)])
    return int(ids[np.argmax(areas[ids - 1] * mean_p[ids - 1])])


def _refine(mask: np.ndarray, view: np.ndarray, cfg: MeasureConfig) -> np.ndarray | None:
    """Move the outline to where the smoothed view crosses half-way between the
    plateaus just inside and just outside it (None when that is not measurable)."""
    r0 = math.sqrt(mask.sum() / math.pi)
    b = max(3.0, cfg.refine_band * r0)
    vs = ndimage.gaussian_filter(np.asarray(view, np.float32), 1.0)
    d_in = ndimage.distance_transform_edt(mask)
    d_out = ndimage.distance_transform_edt(~mask)
    core = d_in > b
    inner = core & (d_in <= 2 * b)
    inner = inner if inner.sum() >= _MIN_RING_PX else core
    outer = (d_out > b) & (d_out <= 2 * b)
    if inner.sum() < _MIN_RING_PX or outer.sum() < _MIN_RING_PX:
        return None
    lo, hi = float(np.median(vs[inner])), float(np.median(vs[outer]))
    if hi - lo < cfg.refine_min_contrast:
        return None
    band = (d_in <= b) & mask | (d_out <= b) & ~mask
    new = ndimage.binary_fill_holes(core | (band & (vs < 0.5 * (lo + hi))))
    labels, n = ndimage.label(new)
    if n > 1:
        overlap = np.bincount(labels[core], minlength=n + 1)
        overlap[0] = 0
        new = labels == int(np.argmax(overlap))
    r1 = math.sqrt(new.sum() / math.pi)
    return new if abs(r1 - r0) <= _MAX_REFINE_CHANGE * r0 else None


def _border_count(mask: np.ndarray) -> int:
    return int(mask[0].sum() + mask[-1].sum() + mask[1:-1, 0].sum() + mask[1:-1, -1].sum())


def _hidden_area(rc: float, cc: float, r: float, shape: tuple[int, int]) -> float:
    """Pixel area of the disk that lies outside the ROI grid."""
    r0, r1 = int(math.floor(rc - r)), int(math.ceil(rc + r)) + 1
    c0, c1 = int(math.floor(cc - r)), int(math.ceil(cc + r)) + 1
    yy, xx = np.ogrid[r0:r1, c0:c1]
    disk = np.hypot(yy - rc, xx - cc) < r
    outside = (yy < 0) | (yy >= shape[0]) | (xx < 0) | (xx >= shape[1])
    return float((disk & outside).sum())


def _measure(mask: np.ndarray) -> tuple[float, float, float, bool]:
    """(radius, rc, cc, clipped), 0-based ROI coordinates."""
    area = float(mask.sum())
    rc, cc = ndimage.center_of_mass(mask)
    radius = math.sqrt(area / math.pi)
    touching = _border_count(mask)
    if touching < max(3, 0.05 * 2 * math.pi * radius):
        return radius, float(rc), float(cc), False
    edge = [c for c in find_contours(mask.astype(np.float32), 0.5) if len(c) >= 3]
    pts = np.concatenate(edge) if edge else np.empty((0, 2))
    if len(pts) < 10:
        return radius, float(rc), float(cc), False
    fit_rc, fit_cc, fit_r = circle_fit_taubin(pts)
    if not (np.isfinite(fit_r) and 0 < fit_r < 3 * radius + 10):
        return radius, float(rc), float(cc), False
    total = area + _hidden_area(fit_rc, fit_cc, fit_r, mask.shape)
    return math.sqrt(total / math.pi), float(fit_rc), float(fit_cc), True


def _outline(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask.astype(np.float32), 1)
    contours = find_contours(padded, 0.5)
    return max(contours, key=len) - 1.0 if contours else _empty_points()


def ml_region(prob: np.ndarray, view: np.ndarray | None = None, *, origin: tuple[int, int] = (1, 1),
              seed=None, prior=None, config: MeasureConfig = MeasureConfig()) -> MLRegion | None:
    """ML's bubble region on the ROI probability map *prob* (None when there is none).

    Parameters
    ----------
    prob : ndarray
        P(bubble) of the ROI pixels.
    view : ndarray, optional
        The engine view of the same ROI (frame / reference); used to move the
        outline to the half-contrast wall when ``config.refine``.
    origin : (row, col)
        1-based full-image coordinate of ROI pixel (0, 0).
    seed : (row, col), optional
        The user's click (1-based full image): the blob holding it, else the nearest.
    prior : (rc, cc, R), optional
        Expected circle (1-based): the blob holding its centre, else the most similar.
    """
    p = np.asarray(prob, np.float32)
    ps = ndimage.gaussian_filter(p, config.smooth_sigma) if config.smooth_sigma > 0 else p
    labels, n = ndimage.label(ndimage.binary_fill_holes(ps >= config.level))
    if n == 0:
        return None
    idx = np.arange(1, n + 1)
    areas = np.bincount(labels.ravel(), minlength=n + 1)[1:].astype(float)
    valid = areas >= math.pi * config.min_radius_px ** 2
    if not valid.any():
        return None
    centroids = np.array(ndimage.center_of_mass(labels > 0, labels, idx)).reshape(-1, 2)
    mean_p = np.asarray(ndimage.mean(ps, labels, idx), float)
    k = _choose(labels, valid, areas, centroids, mean_p, _local(seed, origin), _local(prior, origin))
    mask = labels == k
    refined = False
    if view is not None and config.refine:
        new = _refine(mask, view, config)
        if new is not None:
            mask, refined = new, True
    radius, rc, cc, clipped = _measure(mask)
    terms = region_terms(p, mask, radius, config.level)
    return MLRegion(radius=radius, rc=rc + origin[0], cc=cc + origin[1], confidence=region_confidence(terms),
                    mask=mask, clipped=clipped, refined=refined, terms=terms,
                    contour=_outline(mask) + np.array(origin, float))
