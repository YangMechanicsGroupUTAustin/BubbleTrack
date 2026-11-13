"""Is a fit's boundary an ordered outline (draw a polyline) or a point cloud?"""

from __future__ import annotations

import numpy as np

# A polyline whose length exceeds this multiple of its convex-hull perimeter
# is not a traced contour (raster-ordered edge pixels zig-zag across the shape).
_CONTOUR_LENGTH_RATIO = 2.0
_MIN_CONTOUR_POINTS = 8


def _hull_perimeter(points: np.ndarray) -> float:
    """Perimeter of the convex hull of (N, 2) points (monotone chain)."""
    pts = np.unique(np.asarray(points, dtype=float), axis=0)
    if len(pts) < 3:
        return 0.0
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def half(seq):
        out: list = []
        for p in seq:
            while len(out) >= 2:
                (ax, ay), (bx, by) = out[-2], out[-1]
                if (bx - ax) * (p[1] - ay) - (by - ay) * (p[0] - ax) > 0:
                    break
                out.pop()
            out.append(p)
        return out

    hull = np.array(half(pts)[:-1] + half(pts[::-1])[:-1])
    closed = np.vstack([hull, hull[:1]])
    return float(np.sum(np.hypot(*np.diff(closed, axis=0).T)))


def is_ordered_contour(points: np.ndarray | None) -> bool:
    """True when *points* trace a closed outline in order (draw as a polyline).

    Edge pixels collected in raster order jump back and forth across the
    shape, so their path is many times longer than the outline itself.
    """
    if points is None or len(points) < _MIN_CONTOUR_POINTS:
        return False
    pts = np.asarray(points, dtype=float)
    closed = np.vstack([pts, pts[:1]])
    length = float(np.sum(np.hypot(*np.diff(closed, axis=0).T)))
    hull = _hull_perimeter(pts)
    return hull > 0 and length <= _CONTOUR_LENGTH_RATIO * hull
