"""Split a dark body into lobes joined by thin necks (touching bubbles), keep the main one.

A lobe is a basin of the distance map (inscribed radius) around one of its
local maxima.  Two basins whose pass (saddle) is at least ``neck_ratio`` of
the smaller one's peak belong to the same body (a peanut, a lobed or wobbly
bubble); a lower pass is a neck between separate bodies.  Basins are merged
from the highest pass down, as in a merge tree, so the result does not depend
on the order of the peaks.  Coordinates are 0-based ROI pixels.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage
from skimage import segmentation

from bubbletrack.model.edge.config import RegionConfig


def bbox(mask: np.ndarray, margin: int) -> tuple[int, int, int, int]:
    rows, cols = np.any(mask, axis=1), np.any(mask, axis=0)
    r0, r1 = int(np.argmax(rows)), len(rows) - int(np.argmax(rows[::-1]))
    c0, c1 = int(np.argmax(cols)), len(cols) - int(np.argmax(cols[::-1]))
    h, w = mask.shape
    return max(0, r0 - margin), min(h, r1 + margin), max(0, c0 - margin), min(w, c1 + margin)


def _peaks(dt: np.ndarray, cfg: RegionConfig) -> tuple[int, np.ndarray]:
    """Clusters of local maxima of the distance map that are at least a lobe deep."""
    top = (dt >= cv2.dilate(dt, np.ones((3, 3), np.uint8))) & (dt >= cfg.min_lobe_px)
    return cv2.connectedComponents(top.astype(np.uint8), connectivity=8)


def _passes(basins: np.ndarray, dt: np.ndarray, n: int) -> dict[tuple[int, int], float]:
    """Highest crossing between each pair of touching basins: max of min(dt) over neighbour pairs."""
    table = np.full(n * n, -1.0)
    for a, b, va, vb in ((basins[:, :-1], basins[:, 1:], dt[:, :-1], dt[:, 1:]),
                         (basins[:-1, :], basins[1:, :], dt[:-1, :], dt[1:, :])):
        cross = (a != b) & (a > 0) & (b > 0)
        lo, hi = np.minimum(a[cross], b[cross]), np.maximum(a[cross], b[cross])
        np.maximum.at(table, lo * n + hi, np.minimum(va[cross], vb[cross]))
    keys = np.nonzero(table >= 0)[0]
    return {(int(k // n), int(k % n)): float(table[k]) for k in keys}


def _merge(n: int, peaks: np.ndarray, passes: dict[tuple[int, int], float], ratio: float) -> np.ndarray:
    """Group basins 1..n-1 whose pass is wide relative to the smaller group's peak."""
    parent = np.arange(n)
    best = peaks.astype(float).copy()

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for (a, b), s in sorted(passes.items(), key=lambda kv: -kv[1]):
        ra, rb = root(a), root(b)
        if ra != rb and s >= ratio * min(best[ra], best[rb]):
            parent[rb] = ra
            best[ra] = max(best[ra], best[rb])
    return np.array([root(i) for i in range(n)])


def split_necks(body: np.ndarray, anchor: tuple[float, float], cfg: RegionConfig) -> tuple[np.ndarray, int]:
    """The lobe of *body* holding *anchor*, and the number of separate lobes.

    Parameters
    ----------
    body : ndarray of bool
        One filled, 8-connected dark region (ROI shape).
    anchor : (row, col)
        A point of the main body; when it lies outside every lobe, the deepest lobe is kept.
    cfg : RegionConfig
    """
    r0, r1, c0, c1 = bbox(body, 1)
    crop = np.pad(body[r0:r1, c0:c1], 1)
    dt = cv2.distanceTransform(crop.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    n, markers = _peaks(dt, cfg)
    if n <= 2:
        return body, 1
    basins = segmentation.watershed(-dt, markers, mask=crop)
    peaks = np.zeros(n)
    peaks[1:] = ndimage.maximum(dt, basins, index=np.arange(1, n))
    group = _merge(n, peaks, _passes(basins, dt, n), cfg.neck_ratio)
    groups = group[basins]
    n_lobes = len(np.unique(group[1:]))
    if n_lobes == 1:
        return body, 1
    ar = int(np.clip(round(anchor[0]) - r0 + 1, 0, crop.shape[0] - 1))
    ac = int(np.clip(round(anchor[1]) - c0 + 1, 0, crop.shape[1] - 1))
    keep = groups[ar, ac] if crop[ar, ac] else groups.flat[int(np.argmax(dt))]
    main = np.zeros_like(body)
    main[r0:r1, c0:c1] = ((groups == keep) & crop)[1:-1, 1:-1]
    return main, n_lobes
