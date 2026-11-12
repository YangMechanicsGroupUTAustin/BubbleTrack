"""Which dark pixels form the bubble body: components, filled interiors, anchors.

The bubble's interior belongs to it whatever it shows (a bright centre spot,
rings, a transparent core): the body is the region inside its OUTER boundary.
Interiors enclosed by dark pixels alone are always filled.  Interiors closed
only with the help of excluded pixels (saturated glare, dead reference pixels)
or of the ROI border (a clipped bubble's bright core), or by the steep edge the
rays found where the dark wall itself is broken, or across a gap in the wall a
few pixels wide (a transparent bubble, a faint wall), are filled when the
robust circle fit confirms they lie inside the bubble.  Coordinates are
0-based ROI pixels.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage

from bubbletrack.model.edge.lobes import bbox

_INSIDE_FRACTION = 0.9
"""An interior closed by excluded pixels or the ROI border must lie this much inside the circle."""


def _outside_lut(labels: np.ndarray, n: int) -> np.ndarray:
    """True for label 0 and for every label that touches the array border."""
    out = np.zeros(n, bool)
    out[0] = True
    out[np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])] = True
    return out


def filled_components(dark: np.ndarray) -> tuple[int, np.ndarray]:
    """8-connected dark components with their enclosed holes filled: ``(n_labels, labels)``."""
    nb, bl = cv2.connectedComponents((~dark).astype(np.uint8), connectivity=4)
    filled = dark | ~_outside_lut(bl, nb)[bl]
    return cv2.connectedComponents(filled.astype(np.uint8), connectivity=8)


def label_at(labels: np.ndarray, point: tuple[float, float]) -> int:
    h, w = labels.shape
    r, c = int(round(point[0])), int(round(point[1]))
    return int(labels[r, c]) if 0 <= r < h and 0 <= c < w else 0


def nearest_label(labels: np.ndarray, point: tuple[float, float], search_px: float) -> int:
    """Label at *point*, else of the nearest labelled pixel within *search_px* (0 if none)."""
    found = label_at(labels, point)
    if found:
        return found
    h, w = labels.shape
    r, c = point
    k = int(np.ceil(search_px))
    r0, r1 = max(0, int(r) - k), min(h, int(r) + k + 1)
    c0, c1 = max(0, int(c) - k), min(w, int(c) + k + 1)
    rr, cc = np.nonzero(labels[r0:r1, c0:c1])
    if len(rr) == 0:
        return 0
    d = np.hypot(rr + r0 - r, cc + c0 - c)
    i = int(np.argmin(d))
    return int(labels[rr[i] + r0, cc[i] + c0]) if d[i] <= search_px else 0


def edge_rim(shape: tuple[int, int], points: np.ndarray, rays: np.ndarray, n_rays: int,
             centre: tuple[float, float], inset_px: float) -> np.ndarray | None:
    """The steep edge as a closed-off line: segments joining edge points of neighbouring rays.

    Points are moved *inset_px* towards *centre* so the line runs just inside the
    edge; only rays that are direct neighbours are joined, so a stretch where the
    rays found no edge (a dent, a bite) stays open.
    """
    if len(points) < 2:
        return None
    order = np.argsort(rays)
    p, r = points[order], rays[order]
    v = p - np.asarray(centre)
    p = p - inset_px * v / np.maximum(np.hypot(v[:, 0], v[:, 1]), 1e-9)[:, None]
    nxt = np.roll(np.arange(len(r)), -1)
    joined = (r[nxt] - r) % n_rays == 1
    if not joined.any():
        return None
    seg = np.stack([p[joined][:, ::-1], p[nxt[joined]][:, ::-1]], axis=1)
    img = np.zeros(shape, np.uint8)
    cv2.polylines(img, list(np.round(seg * 16).astype(np.int32)), False, 1, lineType=cv2.LINE_8, shift=4)
    return img > 0


def _enclosed(barrier: np.ndarray, walls: tuple[bool, bool, bool, bool]) -> np.ndarray:
    """Pixels outside *barrier* that it encloses, ROI-border sides in *walls* closed."""
    pad = np.pad(barrier, 1)
    top, bottom, left, right = walls
    pad[0, :] |= top
    pad[-1, :] |= bottom
    pad[:, 0] |= left
    pad[:, -1] |= right
    n, lab = cv2.connectedComponents((~pad).astype(np.uint8), connectivity=4)
    return (~_outside_lut(lab, n)[lab])[1:-1, 1:-1]


def close_body(body: np.ndarray, exclude: np.ndarray, circle: tuple[float, float, float] | None,
               margin_px: float, rim: np.ndarray | None = None, *, gap_frac: float = 0.0,
               gap_max_px: float = 0.0) -> np.ndarray:
    """*body* with the interiors that excluded pixels, the ROI border, the ray edge or a narrow
    gap close, when the circle confirms them (see the module doc).

    Parameters
    ----------
    body : ndarray of bool
        One dark component, its plain holes already filled.
    exclude : ndarray of bool
        Pixels without usable data.
    circle : (rc, cc, R) or None
        The robust circle fit; without it nothing beyond plain holes is filled.
    margin_px : float
        Tolerance outside the circle.
    rim : ndarray of bool, optional
        The steep edge found by the rays (see :func:`edge_rim`), a further barrier.
    gap_frac, gap_max_px : float
        Gaps in the wall narrower than about twice ``min(gap_frac * R, gap_max_px)`` are
        bridged (morphological closing) before looking for enclosed interiors; 0 disables it.
    """
    h, w = body.shape
    r0, r1, c0, c1 = bbox(body, 0)
    if circle is None or not np.isfinite(circle[2]):
        return body
    if rim is not None:
        rr0, rr1, rc0, rc1 = bbox(rim, 0)
        r0, r1, c0, c1 = min(r0, rr0), max(r1, rr1), min(c0, rc0), max(c1, rc1)
    walls = (r0 == 0, r1 == h, c0 == 0, c1 == w)
    crop, ex = body[r0:r1, c0:c1], exclude[r0:r1, c0:c1]
    line = rim[r0:r1, c0:c1] if rim is not None else np.zeros_like(crop)
    if not (any(walls) or ex.any() or line.any()):
        return body
    rr, cc = np.mgrid[r0:r1, c0:c1]
    inside = np.hypot(rr - circle[0], cc - circle[1]) <= circle[2] + margin_px
    barrier = _bridge(crop | ex | line, int(round(min(max(gap_frac * circle[2], 1.0), gap_max_px))))
    added = _confirmed(_enclosed(barrier, walls), inside)
    near = cv2.dilate(added.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    grown = crop | added | (barrier & near & inside)          # bridges and rim join where they closed it
    grown |= _confirmed(_enclosed(grown, walls), inside)      # excluded pixels now enclosed
    out = body.copy()
    out[r0:r1, c0:c1] = grown
    return out


def _bridge(barrier: np.ndarray, k: int) -> np.ndarray:
    """Morphological closing with a disk of radius *k* (no effect for k < 1)."""
    if k < 1:
        return barrier
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    padded = np.pad(barrier.astype(np.uint8), k + 1)
    return (cv2.morphologyEx(padded, cv2.MORPH_CLOSE, disk) > 0)[k + 1:-k - 1, k + 1:-k - 1]


def _confirmed(extra: np.ndarray, inside: np.ndarray) -> np.ndarray:
    """Connected parts of *extra* lying (almost) entirely inside the circle."""
    n, lab = cv2.connectedComponents(extra.astype(np.uint8), connectivity=4)
    if n <= 1:
        return np.zeros_like(extra)
    frac = np.zeros(n)
    frac[1:] = ndimage.mean(inside, lab, index=np.arange(1, n))
    return (frac >= _INSIDE_FRACTION)[lab] & (lab > 0)
