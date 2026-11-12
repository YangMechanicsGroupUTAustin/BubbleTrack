"""Polygon and disk geometry of the region layer.

Points are ``(N, 2)`` arrays of ``[row, col]`` (0-based ROI pixel centres).  A
pixel ``(i, j)`` covers the square ``[i - 0.5, i + 0.5] x [j - 0.5, j + 0.5]``,
so an ROI of shape ``(h, w)`` covers the box ``[-0.5, h - 0.5] x [-0.5, w - 0.5]``.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Moments:
    """Area and first moments (sum of row, sum of col) of a planar region."""

    area: float = 0.0
    m_row: float = 0.0
    m_col: float = 0.0

    def __add__(self, other: Moments) -> Moments:
        return Moments(self.area + other.area, self.m_row + other.m_row, self.m_col + other.m_col)

    def __sub__(self, other: Moments) -> Moments:
        return Moments(self.area - other.area, self.m_row - other.m_row, self.m_col - other.m_col)

    @property
    def centroid(self) -> tuple[float, float]:
        if self.area <= 0:
            return float("nan"), float("nan")
        return self.m_row / self.area, self.m_col / self.area


def polygon_moments(p: np.ndarray) -> Moments:
    """Area and first moments of a closed polygon (vertices in order, either orientation)."""
    if len(p) < 3:
        return Moments()
    y, x = p[:, 0], p[:, 1]
    y1, x1 = np.roll(y, -1), np.roll(x, -1)
    cross = x * y1 - x1 * y
    a = 0.5 * float(cross.sum())
    if abs(a) < 1e-12:
        return Moments()
    m_col = float(((x + x1) * cross).sum()) / 6.0
    m_row = float(((y + y1) * cross).sum()) / 6.0
    s = 1.0 if a > 0 else -1.0
    return Moments(abs(a), s * m_row, s * m_col)


def signed_area(p: np.ndarray) -> float:
    """Shoelace area in the (col, row) frame: positive for counter-clockwise on screen axes."""
    y, x = p[:, 0], p[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def vertex_lengths(p: np.ndarray) -> np.ndarray:
    """Contour length represented by each vertex of a closed polygon (half of each adjacent side)."""
    side = np.hypot(*(np.roll(p, -1, axis=0) - p).T)
    return 0.5 * (side + np.roll(side, 1))


def outward_normals(p: np.ndarray) -> np.ndarray:
    """Unit normals of a closed polygon pointing away from its interior."""
    t = np.roll(p, -1, axis=0) - np.roll(p, 1, axis=0)
    n = np.column_stack([t[:, 1], -t[:, 0]])
    n /= np.maximum(np.hypot(n[:, 0], n[:, 1]), 1e-12)[:, None]
    return n if signed_area(p) < 0 else -n


def hull_area(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    hull = cv2.convexHull(points[:, ::-1].astype(np.float32))
    return float(cv2.contourArea(hull))


def disk_box_moments(circle: tuple[float, float, float], box: tuple[float, float, float, float],
                     nodes: int) -> Moments:
    """Moments of the part of a disk inside an axis-aligned box ``(r0, r1, c0, c1)``.

    Midpoint quadrature over ``col = cc + R sin(phi)`` restricted to the box's
    column range, so the integrand has no jumps (relative error ~1e-6).
    """
    rc, cc, radius = circle
    r0, r1, c0, c1 = box
    if radius <= 0:
        return Moments()
    lo_phi = np.arcsin(np.clip((c0 - cc) / radius, -1.0, 1.0))
    hi_phi = np.arcsin(np.clip((c1 - cc) / radius, -1.0, 1.0))
    if hi_phi <= lo_phi:
        return Moments()
    dphi = (hi_phi - lo_phi) / nodes
    phi = lo_phi + (np.arange(nodes) + 0.5) * dphi
    half = radius * np.cos(phi)
    col = cc + radius * np.sin(phi)
    dcol = half * dphi
    lo = np.maximum(r0, rc - half)
    hi = np.minimum(r1, rc + half)
    length = np.clip(hi - lo, 0.0, None)
    used = length > 0
    return Moments(float(np.sum(length * dcol)), float(np.sum(np.where(used, 0.5 * (hi ** 2 - lo ** 2), 0.0) * dcol)),
                   float(np.sum(length * col * dcol)))


def disk_moments(circle: tuple[float, float, float]) -> Moments:
    rc, cc, radius = circle
    a = np.pi * radius ** 2
    return Moments(a, a * rc, a * cc)


def disk_pixel_moments(circle: tuple[float, float, float], pixels: np.ndarray) -> Moments:
    """Moments of the disk's coverage of the given pixel squares (``(K, 2)`` integer [row, col])."""
    if len(pixels) == 0:
        return Moments()
    rc, cc, radius = circle
    d = np.hypot(pixels[:, 0] - rc, pixels[:, 1] - cc)
    cover = np.clip(radius - d + 0.5, 0.0, 1.0)
    return Moments(float(cover.sum()), float(np.sum(cover * pixels[:, 0])), float(np.sum(cover * pixels[:, 1])))
