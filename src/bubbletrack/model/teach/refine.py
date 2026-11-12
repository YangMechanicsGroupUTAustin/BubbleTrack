"""Move ML's outline to the half-contrast wall of the engine view (the Edge definition).

The classifier's P = 0.5 outline is a good *shape* but its *position* is only
known to within the wall zone that the examples leave unlabelled.  The wall is
where the view (frame / reference, background ~1, bubble ~0) crosses half-way
between its plateaus just inside and just outside the outline.

Per-pixel thresholding is biased by the skewed ratio noise of moving speckle, so
the view is averaged along the outline instead: in each angular sector, the mean
view as a function of the signed distance ``s`` to ML's outline (< 0 inside), and
the crossing taken with the integral estimator ``s* = -b + sum_s f(s)``, with
``f = (hi - profile) / (hi - lo)`` clipped to [0, 1] over the window
``|s| < b`` (exact for a step, robust to noise).  Pixels with ``s < s*`` of their
sector form the refined region.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from bubbletrack.model.teach.config import MeasureConfig

_MIN_PLATEAU_PX = 20
_MIN_SECTOR_PX = 30
_MAX_CHANGE = 0.3


@dataclass(frozen=True, eq=False)
class Refinement:
    mask: np.ndarray
    """The refined region (ROI bool)."""
    offsets: np.ndarray
    """Outline shift of every sector (px, positive = outward)."""
    diagnostics: dict


def _crossing(sums: np.ndarray, counts: np.ndarray, fallback: np.ndarray, lo: float, hi: float,
              b: float) -> float:
    """Integral estimate of where a binned profile steps from *lo* to *hi*."""
    prof = np.where(counts > 0, sums / np.maximum(counts, 1), fallback)
    frac = np.clip((hi - prof) / (hi - lo), 0.0, 1.0)
    return float(-b + frac.sum())


def refine_outline(mask: np.ndarray, view: np.ndarray, config: MeasureConfig = MeasureConfig()) -> Refinement | None:
    """Refined region of *mask* on *view* (both ROI-sized), or None when the wall
    contrast is below ``config.refine_min_contrast`` or cannot be measured."""
    r0 = math.sqrt(mask.sum() / math.pi)
    out = _pass(mask, view, config, config.refine_band)
    for k in range(1, max(int(config.refine_passes), 1)):
        if out is None:
            break
        nxt = _pass(out.mask, view, config, config.refine_band / 2 ** k)
        out = out if nxt is None else nxt
    if out is None:
        return None
    r1 = math.sqrt(out.mask.sum() / math.pi)
    if abs(r1 - r0) > _MAX_CHANGE * r0:
        return None
    return Refinement(out.mask, out.offsets, {**out.diagnostics, "shift": r1 / r0 - 1.0})


def _pass(mask: np.ndarray, view: np.ndarray, config: MeasureConfig, band: float) -> Refinement | None:
    r0 = math.sqrt(mask.sum() / math.pi)
    b = max(3.0, band * r0)
    w = max(3.0, 0.15 * r0)
    d_in = ndimage.distance_transform_edt(mask)
    d_out = ndimage.distance_transform_edt(~mask)
    s = np.where(mask, 0.5 - d_in, d_out - 0.5)
    zone = np.abs(s) <= b + w
    sz, v = s[zone], np.asarray(view, np.float64)[zone]
    inner, outer = sz <= -b, sz >= b
    if inner.sum() < _MIN_PLATEAU_PX or outer.sum() < _MIN_PLATEAU_PX:
        return None
    lo, hi = float(v[inner].mean()), float(v[outer].mean())
    noise = float(v[outer].std())
    if hi - lo < config.refine_min_contrast:
        return None
    k_sectors = max(int(config.refine_sectors), 1)
    rc, cc = ndimage.center_of_mass(mask)
    yy, xx = np.nonzero(zone)
    sector = ((np.arctan2(yy - rc, xx - cc) + np.pi) / (2 * np.pi) * k_sectors).astype(int) % k_sectors
    n_bins = int(math.ceil(2 * b))
    win = np.abs(sz) < b
    bins = np.clip(((sz[win] + b) // 1.0).astype(int), 0, n_bins - 1)
    key = sector[win] * n_bins + bins
    sums = np.bincount(key, weights=v[win], minlength=k_sectors * n_bins).reshape(k_sectors, n_bins)
    counts = np.bincount(key, minlength=k_sectors * n_bins).reshape(k_sectors, n_bins)
    g_sums, g_counts = sums.sum(axis=0), counts.sum(axis=0)
    g_prof = np.where(g_counts > 0, g_sums / np.maximum(g_counts, 1), 0.5 * (lo + hi))
    s_global = _crossing(g_sums, g_counts, g_prof, lo, hi, b)
    offsets = np.array([_crossing(sums[k], counts[k], g_prof, lo, hi, b) if counts[k].sum() >= _MIN_SECTOR_PX
                        else s_global for k in range(k_sectors)])
    new = mask.copy()
    new[zone] = sz < offsets[sector]
    diag = {"view_lo": lo, "view_hi": hi, "view_cnr": (hi - lo) / max(noise, 1e-6)}
    return Refinement(mask=new, offsets=offsets, diagnostics=diag)
