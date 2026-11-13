"""What the user taught: sparse per-frame pixel labels and verified circles.

Two sources of examples, both stored per frame (0-based frame index):

* **strokes**: pixels the user painted as bubble or background, 1-based
  full-image ``(row, col)`` (the app's convention), e.g. from :func:`stroke_pixels`;
* **verified** frames: a circle ``(rc, cc, R)`` the user confirmed (manual fit,
  accepted / locked frame, a dragged circle).  Their training pixels are derived
  on demand (:func:`bubbletrack.model.teach.sampling.training_pixels`), so the
  stored data stays tiny.

:class:`MLExamples` is immutable and round-trips through plain JSON data
(``to_dict`` / ``from_dict``, no pickle), so a session can store it and the
model can be retrained, bit for bit, on load.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np

BUBBLE = "bubble"
BACKGROUND = "background"
LABELS = (BUBBLE, BACKGROUND)
FORMAT_VERSION = 1

Pixel = tuple[int, int]
Circle = tuple[float, float, float]


def _frame(frame) -> int:
    if isinstance(frame, bool) or not isinstance(frame, numbers.Integral) or int(frame) < 0:
        raise ValueError(f"frame must be a non-negative integer, got {frame!r}")
    return int(frame)


def _label(label: str) -> str:
    if label not in LABELS:
        raise ValueError(f"label must be one of {LABELS}, got {label!r}")
    return label


def _pixels(pixels) -> frozenset[Pixel]:
    """Validated 1-based integer (row, col) pairs."""
    arr = np.asarray(pixels if not isinstance(pixels, (set, frozenset)) else sorted(pixels))
    if arr.size == 0:
        return frozenset()
    if arr.ndim != 2 or arr.shape[1] != 2 or arr.dtype.kind not in "iuf":
        raise ValueError(f"pixels must be an (N, 2) array of (row, col), got shape {arr.shape}")
    if not np.isfinite(arr).all() or np.any(arr != np.round(arr)):
        raise ValueError("pixel coordinates must be integers")
    arr = arr.astype(np.int64)
    if (arr < 1).any():
        raise ValueError("pixel coordinates are 1-based: rows and columns start at 1")
    return frozenset(map(tuple, arr.tolist()))


def _circle(circle) -> Circle:
    try:
        out = tuple(float(v) for v in circle)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"circle must be (rc, cc, R), got {circle!r}") from exc
    if len(out) != 3 or not all(math.isfinite(v) for v in out) or out[2] <= 0:
        raise ValueError(f"circle must be three finite numbers (rc, cc, R > 0), got {circle!r}")
    return out


@dataclass(frozen=True)
class FrameStrokes:
    """Painted pixels of one frame (1-based (row, col))."""

    bubble: frozenset[Pixel] = frozenset()
    background: frozenset[Pixel] = frozenset()

    def of(self, label: str) -> frozenset[Pixel]:
        return self.bubble if _label(label) == BUBBLE else self.background

    @property
    def is_empty(self) -> bool:
        return not (self.bubble or self.background)


@dataclass(frozen=True)
class MLExamples:
    """All examples of one recording.  Every method returns a new object."""

    strokes: Mapping[int, FrameStrokes] = field(default_factory=dict)
    verified: Mapping[int, Circle] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "strokes", MappingProxyType(dict(self.strokes)))
        object.__setattr__(self, "verified", MappingProxyType(dict(self.verified)))

    __hash__ = None  # mappings inside: equality only

    # ------------------------------------------------------------------ queries
    @property
    def frames(self) -> tuple[int, ...]:
        """Frames with any example, ascending."""
        return tuple(sorted(set(self.strokes) | set(self.verified)))

    @property
    def is_empty(self) -> bool:
        return not self.frames

    # ------------------------------------------------------------------ edits
    def paint(self, frame: int, label: str, pixels) -> MLExamples:
        """Label *pixels* of *frame*; a pixel painted before with the other label moves."""
        frame, label, px = _frame(frame), _label(label), _pixels(pixels)
        old = self.strokes.get(frame, FrameStrokes())
        if label == BUBBLE:
            new = FrameStrokes(bubble=old.bubble | px, background=old.background - px)
        else:
            new = FrameStrokes(bubble=old.bubble - px, background=old.background | px)
        return self._with_strokes(frame, new)

    def erase(self, frame: int, pixels) -> MLExamples:
        """Remove *pixels* from both labels of *frame*."""
        frame, px = _frame(frame), _pixels(pixels)
        old = self.strokes.get(frame)
        if old is None:
            return self
        return self._with_strokes(frame, FrameStrokes(old.bubble - px, old.background - px))

    def verify(self, frame: int, circle) -> MLExamples:
        """Record a verified circle ``(rc, cc, R)`` (1-based full-image) for *frame*."""
        verified = dict(self.verified)
        verified[_frame(frame)] = _circle(circle)
        return MLExamples(self.strokes, verified)

    def with_verified(self, circles: Mapping[int, Circle]) -> MLExamples:
        """Replace every verified circle by *circles* ({frame: (rc, cc, R)}); strokes are kept.

        For syncing with the app's manual / locked frames in one call.
        """
        return MLExamples(self.strokes, {_frame(f): _circle(c) for f, c in circles.items()})

    def unverify(self, frame: int) -> MLExamples:
        if _frame(frame) not in self.verified:
            return self
        return MLExamples(self.strokes, {k: v for k, v in self.verified.items() if k != frame})

    def clear(self, frame: int) -> MLExamples:
        """Drop every example of *frame* (strokes and verified circle)."""
        frame = _frame(frame)
        return MLExamples({k: v for k, v in self.strokes.items() if k != frame},
                          {k: v for k, v in self.verified.items() if k != frame})

    def _with_strokes(self, frame: int, new: FrameStrokes) -> MLExamples:
        strokes = dict(self.strokes)
        if new.is_empty:
            strokes.pop(frame, None)
        else:
            strokes[frame] = new
        return MLExamples(strokes, self.verified)

    # ------------------------------------------------------------------ JSON
    def to_dict(self) -> dict:
        """Plain JSON-able data (sorted, so equal examples give equal JSON)."""
        return {
            "version": FORMAT_VERSION,
            "strokes": {str(f): {lab: [list(p) for p in sorted(s.of(lab))] for lab in LABELS}
                        for f, s in sorted(self.strokes.items())},
            "verified": {str(f): list(c) for f, c in sorted(self.verified.items())},
        }

    @classmethod
    def from_dict(cls, data) -> MLExamples:
        """Inverse of :meth:`to_dict`.  Raises ValueError for malformed data."""
        if not isinstance(data, Mapping) or data.get("version") != FORMAT_VERSION:
            raise ValueError(f"not ML assist examples (version {FORMAT_VERSION})")
        strokes_in, verified_in = data.get("strokes", {}), data.get("verified", {})
        if not isinstance(strokes_in, Mapping) or not isinstance(verified_in, Mapping):
            raise ValueError("strokes and verified must be mappings")
        out = cls()
        for key, labels in strokes_in.items():
            frame = _key(key)
            if not isinstance(labels, Mapping) or set(labels) - set(LABELS):
                raise ValueError(f"frame {key}: labels must be among {LABELS}")
            for lab, px in labels.items():
                out = out.paint(frame, lab, np.asarray(px, dtype=float).reshape(-1, 2) if px else [])
        for key, circle in verified_in.items():
            out = out.verify(_key(key), circle)
        return out


def _key(key) -> int:
    try:
        return _frame(int(key))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"frame key must be a non-negative integer, got {key!r}") from exc


def stroke_pixels(points: Iterable[tuple[float, float]], radius: float, shape: tuple[int, int]) -> np.ndarray:
    """Pixels covered by a round brush of *radius* px dragged along *points*.

    Parameters
    ----------
    points : sequence of (row, col)
        1-based full-image coordinates of the stroke's vertices (one point: a dot).
    radius : float
        Brush radius in px: pixel centres within ``max(radius, 0.5)`` of the polyline.
    shape : (height, width)
        Frame size; the stroke is clipped to it.

    Returns
    -------
    ndarray (N, 2) int
        Covered pixels, 1-based ``(row, col)``, row-major order.
    """
    pts = np.asarray(list(points), dtype=float).reshape(-1, 2) - 1.0
    h, w = (int(s) for s in shape)
    mask = np.zeros((h, w), bool)
    r = max(float(radius), 0.5)
    segments = list(zip(pts[:-1], pts[1:])) if len(pts) > 1 else [(p, p) for p in pts]
    for a, b in segments:
        lo = np.maximum(np.floor(np.minimum(a, b) - r), 0).astype(int)
        hi = np.minimum(np.ceil(np.maximum(a, b) + r) + 1, (h, w)).astype(int)
        if (hi <= lo).any():
            continue
        yy, xx = np.ogrid[lo[0]:hi[0], lo[1]:hi[1]]
        ab = b - a
        length2 = float(ab @ ab)
        t = 0.0 if length2 == 0 else np.clip(((yy - a[0]) * ab[0] + (xx - a[1]) * ab[1]) / length2, 0.0, 1.0)
        d = np.hypot(yy - (a[0] + t * ab[0]), xx - (a[1] + t * ab[1]))
        mask[lo[0]:hi[0], lo[1]:hi[1]] |= d <= r
    return np.argwhere(mask) + 1
