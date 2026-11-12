"""How much to trust an ML outline (its own region, not Auto's wall).

Terms, each in [0, 1]:

* ``separation``: mean P inside the region minus mean P in a ring just outside;
* ``sharpness``: 1 minus the share of undecided pixels (0.2 < P < 0.8) in a
  thin band around the outline, beyond what a sharp step gives;
* ``shape``: solidity (area / convex hull) of the region; bubbles are convex blobs;
* ``clutter``: 1 minus the other bubble-like area (P >= 0.5) near the region,
  relative to its own area.

The confidence is their product: every term must be good.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.measure import regionprops

SHARP_BASE = 0.3     # undecided share of the band that a sharp, smoothed step already gives
SHARP_SPAN = 0.6
SOLIDITY_LOW, SOLIDITY_HIGH = 0.75, 0.92
CLUTTER_SPAN = 0.6


def _clip01(x: float) -> float:
    return float(min(max(x, 0.0), 1.0))


def region_terms(prob: np.ndarray, mask: np.ndarray, radius: float, level: float = 0.5) -> dict[str, float]:
    """Confidence terms of the region *mask* (ROI bool) on the probability map *prob*."""
    band_w = max(2.0, 0.1 * radius)
    ring_w = max(3.0, 0.3 * radius)
    d_in = ndimage.distance_transform_edt(mask)
    d_out = ndimage.distance_transform_edt(~mask)
    inner = d_in > band_w
    inner = inner if inner.any() else mask
    ring = (d_out > band_w) & (d_out <= band_w + ring_w)
    band = (mask & (d_in <= band_w)) | (~mask & (d_out <= band_w))
    p_in = float(prob[inner].mean())
    p_out = float(prob[ring].mean()) if ring.any() else 0.0
    undecided = float(((prob[band] > 0.2) & (prob[band] < 0.8)).mean()) if band.any() else 1.0
    props = regionprops(mask.astype(np.uint8))
    solidity = float(props[0].solidity) if props else 0.0
    near = (d_out <= max(radius, 3.0)) & ~mask
    other = float((prob[near] >= level).sum()) / max(float(mask.sum()), 1.0)
    return {
        "separation": _clip01(p_in - p_out),
        "sharpness": _clip01(1.0 - (undecided - SHARP_BASE) / SHARP_SPAN),
        "shape": _clip01((solidity - SOLIDITY_LOW) / (SOLIDITY_HIGH - SOLIDITY_LOW)),
        "clutter": _clip01(1.0 - other / CLUTTER_SPAN),
        "solidity": solidity,
        "undecided": undecided,
    }


def region_confidence(terms: dict[str, float]) -> float:
    return float(terms["separation"] * terms["sharpness"] * terms["shape"] * terms["clutter"])
