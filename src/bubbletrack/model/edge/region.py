"""Stage G: the bubble as a region of the edge image, whatever its shape.

The bubble is the dark region enclosed by its OUTER iso-contour at the edge
level (holes such as the bright centre spot or an inner ring are filled).
Bodies joined only by a thin neck are split and the main one is kept; the
part of the main body hidden by the ROI border or by excluded pixels is
completed with the robust circle fit.  Coordinates are 0-based ROI pixels
``[row, col]``; the ROI covers ``[-0.5, h - 0.5] x [-0.5, w - 0.5]``.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage
from skimage import measure

from bubbletrack.model.edge.body import close_body, filled_components, label_at, nearest_label
from bubbletrack.model.edge.config import RegionConfig
from bubbletrack.model.edge.geometry import (
    Moments,
    disk_box_moments,
    disk_moments,
    disk_pixel_moments,
    hull_area,
    polygon_moments,
    vertex_lengths,
)
from bubbletrack.model.edge.lobes import bbox, split_necks

_EPS = 1e-6
_BORDER_TOL = 1e-6


@dataclass(frozen=True)
class Region:
    """The main bubble body at one iso level."""

    mask: np.ndarray
    """Filled main body (bool, ROI shape)."""
    contour: np.ndarray
    """(N, 2) closed outline of the visible part, ROI-border closure included."""
    edge_mask: np.ndarray
    """(N,) True for contour vertices on a genuine edge (not on the ROI border, not at excluded pixels)."""
    lengths: np.ndarray
    """(N,) contour length each vertex stands for."""
    visible: Moments
    hidden: Moments
    """Part of the fitted disk outside the valid region (zero unless ``completed``)."""
    perimeter: float
    hull_area: float
    clipped: bool
    """The outline runs along the ROI border or excluded pixels."""
    completed: bool
    """A hidden part was added from the circle fit."""
    others: np.ndarray
    """Dark regions at this level other than the main body (other components, split-off lobes)."""
    fragment_count: int
    fragment_area: float
    n_lobes: int
    anchor: str
    hidden_arc: float = 0.0
    """Fraction of the fitted circle's circumference outside the valid region (0 unless ``clipped``)."""

    @property
    def edge(self) -> np.ndarray:
        return self.contour[self.edge_mask]

    @property
    def edge_length(self) -> np.ndarray:
        return self.lengths[self.edge_mask]

    @property
    def area(self) -> float:
        return self.visible.area + self.hidden.area

    @property
    def radius(self) -> float:
        return float(np.sqrt(self.area / np.pi))

    @property
    def centroid(self) -> tuple[float, float]:
        return (self.visible + self.hidden).centroid

    @property
    def hidden_fraction(self) -> float:
        return self.hidden.area / self.area if self.area > 0 else 1.0

    @property
    def solidity(self) -> float:
        return float(min(1.0, self.area / self.hull_area)) if self.hull_area > 0 else float("nan")

    @property
    def circularity(self) -> float:
        return float(4 * np.pi * self.area / self.perimeter ** 2) if self.perimeter > 0 else float("nan")


def trace(img: np.ndarray, level: float, body: np.ndarray, margin: int) -> np.ndarray:
    """Sub-pixel outline of *body* on the iso-level *level* of *img* (closed, without repeat).

    Pixels are forced to the side of the level that *body* assigns them (mirrored
    about the level, so genuine crossings stay exactly where they were).  Along
    the ROI border the outline runs at the border itself (-0.5 / size - 0.5).
    """
    h, w = img.shape
    r0, r1, c0, c1 = bbox(body, margin)
    dev = np.maximum(np.abs(img[r0:r1, c0:c1].astype(np.float64) - level), _EPS)
    forced = np.where(body[r0:r1, c0:c1], level - dev, level + dev)
    padded = level + np.abs(np.pad(forced, 1, mode="edge") - level)
    padded[1:-1, 1:-1] = forced
    contours = measure.find_contours(padded, level, fully_connected="low")
    if not contours:
        return np.empty((0, 2))
    c = max(contours, key=len)[:-1] + np.array([r0 - 1, c0 - 1], float)
    c[:, 0] = np.clip(c[:, 0], -0.5, h - 0.5)
    c[:, 1] = np.clip(c[:, 1], -0.5, w - 0.5)
    return c


def _on_border(c: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    return ((np.abs(c[:, 0] + 0.5) < _BORDER_TOL) | (np.abs(c[:, 0] - (h - 0.5)) < _BORDER_TOL)
            | (np.abs(c[:, 1] + 0.5) < _BORDER_TOL) | (np.abs(c[:, 1] - (w - 0.5)) < _BORDER_TOL))


def _near(mask: np.ndarray, c: np.ndarray, k: int) -> np.ndarray:
    if not mask.any():
        return np.zeros(len(c), bool)
    grown = cv2.dilate(mask.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
    h, w = mask.shape
    ri = np.clip(np.rint(c[:, 0]), 0, h - 1).astype(int)
    ci = np.clip(np.rint(c[:, 1]), 0, w - 1).astype(int)
    return grown[ri, ci]


def _hidden(circle, body: np.ndarray, closure: np.ndarray, cfg: RegionConfig) -> tuple[Moments, np.ndarray]:
    """Moments of the fitted disk outside the valid region, and its hidden arc points."""
    h, w = body.shape
    hidden = disk_moments(circle) - disk_box_moments(circle, (-0.5, h - 0.5, -0.5, w - 0.5), cfg.disk_nodes)
    rc, cc, radius = circle
    r0, r1 = max(0, int(np.floor(rc - radius - 1))), min(h, int(np.ceil(rc + radius + 2)))
    c0, c1 = max(0, int(np.floor(cc - radius - 1))), min(w, int(np.ceil(cc + radius + 2)))
    if r1 > r0 and c1 > c0:
        rr, cl = np.nonzero(closure[r0:r1, c0:c1])
        hidden = hidden + disk_pixel_moments(circle, np.column_stack([rr + r0, cl + c0]))
    th = np.linspace(0, 2 * np.pi, cfg.arc_samples, endpoint=False)
    arc = np.column_stack([rc + radius * np.sin(th), cc + radius * np.cos(th)])
    out = (arc[:, 0] < -0.5) | (arc[:, 0] > h - 0.5) | (arc[:, 1] < -0.5) | (arc[:, 1] > w - 0.5)
    ri = np.clip(np.rint(arc[:, 0]), 0, h - 1).astype(int)
    ci = np.clip(np.rint(arc[:, 1]), 0, w - 1).astype(int)
    out |= closure[ri, ci]
    if hidden.area < 0:
        hidden = Moments()
    return hidden, arc[out]


def _fragments(others: np.ndarray, main: np.ndarray, radius: float, cfg: RegionConfig) -> tuple[int, float]:
    """Other dark regions (and split-off lobes) within ``fragment_reach`` radii of the main body."""
    if not others.any():
        return 0, 0.0
    n, lab = cv2.connectedComponents(others.astype(np.uint8), connectivity=8)
    dist = cv2.distanceTransform((~main).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    idx = np.arange(1, n)
    near = np.asarray(ndimage.minimum(dist, lab, idx)) <= cfg.fragment_reach * radius
    area = np.asarray(ndimage.sum_labels(np.ones_like(dist), lab, idx))
    big = area >= max(cfg.fragment_min_px, cfg.fragment_min_frac * np.pi * radius ** 2)
    keep = near & big
    return int(keep.sum()), float(area[keep].sum())


def own_cell(region: Region) -> np.ndarray | None:
    """Pixels nearer to the main body than to any other dark region (None when there is none).

    Confines the outer-extent region to the steep main body's neighbourhood, so a
    satellite that merges with it at the higher level is not swallowed.
    """
    if not region.others.any():
        return None
    d_main = cv2.distanceTransform((~region.mask).astype(np.uint8), cv2.DIST_L2, 3)
    d_other = cv2.distanceTransform((~region.others).astype(np.uint8), cv2.DIST_L2, 3)
    return d_main <= d_other


def _main_body(labels: np.ndarray, exclude: np.ndarray, candidate: tuple[float, float],
               preferred: list[tuple[str, tuple[float, float]]], circle, cfg: RegionConfig,
               search_px: float, rim: np.ndarray | None) -> tuple[np.ndarray | None, str, tuple[float, float]]:
    """The body holding the first preferred point (prior centre, seed), else the candidate's.

    The candidate's body is closed with the circle first, so a click on its bright
    core or on an island inside it still means that body.
    """
    cand = nearest_label(labels, candidate, search_px)
    body_c = (close_body(labels == cand, exclude, circle, cfg.crop_margin_px, rim, gap_frac=cfg.gap_frac,
                         gap_max_px=cfg.gap_max_px) if cand else None)
    for name, point in preferred:
        if body_c is not None and label_at(body_c, point):
            return body_c, name, point
        other = label_at(labels, point)
        if other:
            return labels == other, name, point
    if body_c is None:
        return None, "none", candidate
    return body_c, "candidate" if label_at(labels, candidate) else "nearest", candidate


def find_region(img: np.ndarray, level: float, exclude: np.ndarray, candidate: tuple[float, float],
                circle: tuple[float, float, float] | None, cfg: RegionConfig, *, search_px: float,
                preferred: list[tuple[str, tuple[float, float]]] = (),
                allowed: np.ndarray | None = None, rim: np.ndarray | None = None) -> Region | None:
    """The main body at *level* of *img* (dark below the level), or None when there is none.

    Parameters
    ----------
    img : ndarray
        Edge image, background ~1, bubble dark; excluded pixels already bright.
    level : float
        Iso level of the edge in *img* (the engine passes ``image - level map`` and 0).
    exclude : ndarray of bool
        Pixels without usable data (saturated glare, dead reference pixels).
    candidate : (row, col)
        A dark point of the selected candidate (its deepest pixel).
    circle : (rc, cc, R) or None
        Robust circle fit: confirms interiors closed by excluded pixels or the ROI
        border, and completes a clipped body.
    cfg : RegionConfig
    search_px : float
        When *candidate* is not dark at this level, the nearest region within this distance.
    preferred : list of (name, (row, col))
        Points that choose the body before the candidate (prior centre, seed click).
    allowed : ndarray of bool, optional
        Pixels the region may occupy (see :func:`own_cell`).
    rim : ndarray of bool, optional
        The steep edge found by the rays (``body.edge_rim``): closes the body where its wall is broken.
    """
    dark = img < level
    _, labels = filled_components(dark if allowed is None else dark & allowed)
    body, how, anchor = _main_body(labels, exclude, candidate, list(preferred), circle, cfg, search_px, rim)
    if body is None:
        return None
    main, n_lobes = split_necks(body, anchor, cfg)
    contour = trace(img, level, main, cfg.crop_margin_px)
    if len(contour) < 3:
        return None
    closure_px = exclude & ~main
    closing = _on_border(contour, img.shape) | _near(closure_px, contour, cfg.invalid_touch_px)
    lengths = vertex_lengths(contour)
    visible = polygon_moments(contour)
    clipped = bool(closing.any())
    hidden, arc = Moments(), np.empty((0, 2))
    if clipped and circle is not None and np.isfinite(circle[2]) and circle[2] > 0:
        hidden, arc = _hidden(circle, main, closure_px, cfg)
    completed = hidden.area > _EPS
    edge = contour[~closing]
    perimeter = float(lengths[~closing].sum()) + (2 * np.pi * circle[2] * len(arc) / cfg.arc_samples
                                                  if completed else 0.0)
    hull = hull_area(np.vstack([edge, arc]) if completed else contour)
    radius = float(np.sqrt((visible.area + hidden.area) / np.pi))
    others = ((labels > 0) & ~body) | (body & ~main)
    n_frag, frag_area = _fragments(others, main, radius, cfg)
    return Region(mask=main, contour=contour, edge_mask=~closing, lengths=lengths,
                  visible=visible, hidden=hidden, perimeter=perimeter, hull_area=hull, clipped=clipped,
                  completed=completed, others=others, fragment_count=n_frag, fragment_area=frag_area,
                  n_lobes=n_lobes, anchor=how, hidden_arc=len(arc) / cfg.arc_samples)
