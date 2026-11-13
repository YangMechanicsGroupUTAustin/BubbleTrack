"""Stage F: is the bubble round enough for its circle?  (the circle check)

The default radius is the robust circle through the bubble wall; it is the
bubble's size only when the bubble is round.  This module estimates the
area-equivalent radius WITHOUT the circle, so the two can be compared:

* a bubble inside the ROI: the area of its outline; also one whose outline only
  touches the ROI border or excluded pixels (a glare blob next to the wall), with
  a hidden part too small to matter (``QualityConfig.clipped_check_hidden``);
* a bubble cut by the ROI border or by excluded pixels: its visible area plus
  the hidden part of an ELLIPSE fitted to its genuine wall.  The circle cannot
  be its own check there: the area completed with the circle agrees with the
  circle by construction, while a non-round bubble's visible wall is often
  close to some circular arc (half of a 1.5:1 ellipse is within 2 % of one).
  A round bubble's ellipse is its circle; for a non-round one the two
  completions differ, and by about the error of the circle's radius.

Coordinates are 0-based ROI pixels ``[row, col]``; the ROI covers
``[-0.5, h - 0.5] x [-0.5, w - 0.5]``.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from bubbletrack.model.edge.geometry import polygon_moments
from bubbletrack.model.edge.region import Region

_MIN_POINTS = 6
_POLY_VERTICES = 720
_MAX_ASPECT = 4.0
"""An ellipse longer than this many times its width is a fit to a nearly straight wall: no check."""
_MAX_AXIS_FACTOR = 4.0
"""Nor one whose long half-axis exceeds this many circle radii."""


@dataclass(frozen=True)
class CircleCheck:
    """The circle's radius against an area-equivalent radius measured without it."""

    r_estimate: float
    """Area-equivalent radius from the outline (NaN when it could not be estimated)."""
    method: str
    """"area" (inside the ROI, or hardly hidden), "ellipse" (clipped: completed with an ellipse)
    or "none"."""

    def rel_diff(self, radius: float) -> float:
        """|r_estimate - radius| / radius (NaN when unknown)."""
        if not (np.isfinite(self.r_estimate) and np.isfinite(radius) and radius > 0):
            return float("nan")
        return float(abs(self.r_estimate - radius) / radius)


NO_CHECK = CircleCheck(float("nan"), "none")


def circle_check(region: Region, circle_radius: float, exclude: np.ndarray,
                 max_hidden: float = 0.0) -> CircleCheck:
    """Area-equivalent radius of *region* estimated independently of its circle fit.

    Parameters
    ----------
    region : Region
        The measured bubble region (``completed`` regions carry the circle's hidden part,
        which is only used when it is at most *max_hidden* of the area).
    circle_radius : float
        Radius of the circle fit (bounds the ellipse that is accepted as a model).
    exclude : ndarray of bool
        Pixels without usable data (the ROI shape).
    max_hidden : float
        A clipped region with at most this share of its area hidden is measured by its
        (completed) area: the circle's guess for so small a part cannot move it.
    """
    if not region.clipped or region.hidden_fraction <= max_hidden:
        return CircleCheck(float(np.sqrt(region.area / np.pi)), "area")
    ellipse = fit_ellipse(region.edge, circle_radius)
    if ellipse is None:
        return NO_CHECK
    hidden = ellipse_hidden_area(ellipse, exclude.shape, exclude & ~region.mask)
    return CircleCheck(float(np.sqrt((region.visible.area + hidden) / np.pi)), "ellipse")


def fit_ellipse(points: np.ndarray, circle_radius: float):
    """Direct least-squares ellipse (OpenCV) through *points* ``[row, col]``, or None
    when there are too few points or the ellipse is degenerate."""
    if len(points) < _MIN_POINTS or not np.isfinite(circle_radius) or circle_radius <= 0:
        return None
    try:
        (cx, cy), (w, h), angle = cv2.fitEllipseDirect(points[:, ::-1].astype(np.float32))
    except cv2.error:
        return None
    a, b = 0.5 * max(w, h), 0.5 * min(w, h)
    if not (np.isfinite([cx, cy, a, b, angle]).all() and b > 0):
        return None
    if a / b > _MAX_ASPECT or a > _MAX_AXIS_FACTOR * circle_radius:
        return None
    return (cx, cy), (w, h), angle


def ellipse_polygon(ellipse, n: int = _POLY_VERTICES) -> np.ndarray:
    """(n, 2) vertices ``[row, col]`` of an OpenCV ellipse ``((cx, cy), (w, h), angle_deg)``."""
    (cx, cy), (w, h), angle = ellipse
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ca, sa = np.cos(np.radians(angle)), np.sin(np.radians(angle))
    u, v = 0.5 * w * np.cos(t), 0.5 * h * np.sin(t)
    return np.column_stack([cy + u * sa + v * ca, cx + u * ca - v * sa])


def _clip_half_plane(p: np.ndarray, axis: int, bound: float, keep_below: bool) -> np.ndarray:
    """Sutherland-Hodgman step: the part of polygon *p* on one side of ``x[axis] = bound``."""
    if len(p) == 0:
        return p
    inside = p[:, axis] <= bound if keep_below else p[:, axis] >= bound
    prev, prev_in = np.roll(p, 1, axis=0), np.roll(inside, 1)
    out = []
    for a, b, a_in, b_in in zip(prev, p, prev_in, inside):
        if a_in != b_in:
            t = (bound - a[axis]) / (b[axis] - a[axis])
            out.append(a + t * (b - a))
        if b_in:
            out.append(b)
    return np.array(out) if out else np.empty((0, 2))


def clip_to_box(p: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
    """Convex polygon *p* clipped to the box ``(r0, r1, c0, c1)``."""
    r0, r1, c0, c1 = box
    for axis, bound, below in ((0, r0, False), (0, r1, True), (1, c0, False), (1, c1, True)):
        p = _clip_half_plane(p, axis, bound, below)
    return p


def ellipse_hidden_area(ellipse, shape: tuple[int, int], closure: np.ndarray) -> float:
    """Area of the ellipse outside the ROI box plus its coverage of *closure* pixels
    (excluded pixels next to the region), the latter sampled at pixel centres."""
    h, w = shape
    poly = ellipse_polygon(ellipse)
    total = polygon_moments(poly).area
    inside = clip_to_box(poly, (-0.5, h - 0.5, -0.5, w - 0.5))
    hidden = total - (polygon_moments(inside).area if len(inside) >= 3 else 0.0)
    if closure.any():
        rr, cc = np.nonzero(closure)
        (cx, cy), (ew, eh), angle = ellipse
        ca, sa = np.cos(np.radians(angle)), np.sin(np.radians(angle))
        dx, dy = cc - cx, rr - cy
        u, v = dx * ca + dy * sa, -dx * sa + dy * ca
        hidden += float(np.count_nonzero((u / (0.5 * ew)) ** 2 + (v / (0.5 * eh)) ** 2 <= 1.0))
    return float(max(hidden, 0.0))
