"""How much to trust an ML outline (its own region, not Auto's wall).

Terms, each in [0, 1]:

* ``separation``: mean P inside the region minus mean P in a ring 0.3-0.6 R
  outside it (beyond the wall zone that verified frames leave unlabelled, where
  P is undecided by construction);
* ``shape``: solidity (area / convex hull) of the region; bubbles are convex blobs;
* ``clutter``: 1 minus the other bubble-like area (P >= 0.5) within R of the
  region, relative to its own area.

The confidence is their product.  The other entries (``undecided``,
``circularity``, ...) are diagnostics for the study.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import ndimage
from skimage.measure import regionprops

SOLIDITY_LOW, SOLIDITY_HIGH = 0.75, 0.92
CLUTTER_SPAN = 0.6
FAR_RING = (0.3, 0.6)


def _clip01(x: float) -> float:
    return float(min(max(x, 0.0), 1.0))


def _mean(values: np.ndarray, empty: float) -> float:
    return float(values.mean()) if values.size else empty


def region_terms(prob: np.ndarray, mask: np.ndarray, radius: float, level: float = 0.5) -> dict[str, float]:
    """Confidence terms and diagnostics of the region *mask* (ROI bool) on the map *prob*."""
    band_w = max(2.0, 0.1 * radius)
    d_in = ndimage.distance_transform_edt(mask)
    d_out = ndimage.distance_transform_edt(~mask)
    inner = d_in > band_w
    inner = inner if inner.any() else mask
    far = (d_out > max(FAR_RING[0] * radius, band_w)) & (d_out <= max(FAR_RING[1] * radius, 2 * band_w))
    band = (mask & (d_in <= band_w)) | (~mask & (d_out <= band_w))
    p_in, p_far = _mean(prob[inner], 0.0), _mean(prob[far], 0.0)
    props = regionprops(mask.astype(np.uint8))
    solidity = float(props[0].solidity) if props else 0.0
    perimeter = float(props[0].perimeter) if props else 0.0
    area = float(mask.sum())
    near = (d_out <= max(radius, 3.0)) & ~mask
    other = float((prob[near] >= level).sum()) / max(area, 1.0)
    return {
        "separation": _clip01(p_in - p_far),
        "shape": _clip01((solidity - SOLIDITY_LOW) / (SOLIDITY_HIGH - SOLIDITY_LOW)),
        "clutter": _clip01(1.0 - other / CLUTTER_SPAN),
        "p_in": p_in,
        "p_far": p_far,
        "undecided": _mean(((prob[band] > 0.2) & (prob[band] < 0.8)).astype(float), 1.0),
        "solidity": solidity,
        "circularity": 4 * math.pi * area / perimeter ** 2 if perimeter > 0 else 0.0,
        "clutter_ratio": other,
        "area_frac": area / mask.size,
    }


def region_confidence(terms: dict[str, float]) -> float:
    return float(terms["separation"] * terms["shape"] * terms["clutter"])
