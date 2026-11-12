"""Non-circular bubble shapes with exact area ground truth, for the region-layer tests.

Ported from the concave-bubble experiment (``concave_test.py``) and extended:
each shape is an exact transmission map rendered at 4x super-sampling (exact
area coverage), blurred by a 1.2-px PSF, scaled to a 32k-DN background, with
1 % Gaussian noise, as uint16.  Coordinates are 0-based pixel centres; the
truth is the area-equivalent radius ``sqrt(A / pi)`` of the dark body,
holes (bright centre spot, inner ring) included.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

H, W = 250, 400
C = (125.0, 200.0)
SS = 4
BG = 32000.0
DARK = 0.06

Mask = Callable[[np.ndarray, np.ndarray], np.ndarray]


def _grid() -> tuple[np.ndarray, np.ndarray]:
    yy = (np.arange(H * SS) + 0.5) / SS - 0.5
    xx = (np.arange(W * SS) + 0.5) / SS - 0.5
    return np.meshgrid(yy, xx, indexing="ij")


def disk(yy, xx, cy, cx, r):
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2


def _circle(yy, xx):
    return disk(yy, xx, *C, 60)


def _shallow_bite(yy, xx):
    return disk(yy, xx, *C, 60) & ~disk(yy, xx, C[0], C[1] + 75, 28)


def _kidney(yy, xx):
    return disk(yy, xx, *C, 60) & ~disk(yy, xx, C[0], C[1] + 62, 40)


def _jet_slot(yy, xx):
    return disk(yy, xx, *C, 60) & ~((np.abs(yy - C[0]) < 8) & (xx > C[1] + 12))


def _peanut(yy, xx):
    return disk(yy, xx, C[0], C[1] - 34, 42) | disk(yy, xx, C[0], C[1] + 34, 42)


def _five_lobe(yy, xx):
    dy, dx = yy - C[0], xx - C[1]
    return np.hypot(dy, dx) <= 55 + 14 * np.cos(5 * np.arctan2(dy, dx))


def _crescent(yy, xx):
    return disk(yy, xx, *C, 60) & ~disk(yy, xx, C[0], C[1] + 28, 50)


def _mushroom(yy, xx):
    """A cap with a concave underside (downward horns) and a stem: rays from the
    cap centre exit the cap, cross the gap and re-enter a horn (3+ crossings)."""
    cap = disk(yy, xx, C[0] - 15, C[1], 50) & ~disk(yy, xx, C[0] + 30, C[1], 38)
    stem = (np.abs(xx - C[1]) < 9) & (yy > C[0] - 15) & (yy < C[0] + 65)
    return cap | stem


@dataclass(frozen=True)
class Shape:
    name: str
    mask: Mask
    seed: tuple[float, float]
    """0-based (row, col) of a click on the main body."""
    spot_ring: bool = False
    """Interior with a saturated centre spot and a bright inner ring (holes of the dark body)."""


SHAPES = (
    Shape("circle", _circle, (C[0], C[1] - 30)),
    Shape("shallow_bite", _shallow_bite, (C[0], C[1] - 30)),
    Shape("kidney", _kidney, (C[0], C[1] - 30)),
    Shape("jet_slot", _jet_slot, (C[0], C[1] - 30)),
    Shape("peanut", _peanut, (C[0], C[1] - 30)),
    Shape("five_lobe", _five_lobe, (C[0], C[1] - 30)),
    Shape("crescent", _crescent, (C[0], C[1] - 45)),
    Shape("mushroom", _mushroom, (C[0] - 40, C[1])),
    Shape("spot_and_ring", _circle, (C[0], C[1] - 30), spot_ring=True),
)


def transmission(shape: Shape) -> tuple[np.ndarray, float]:
    """Pixel transmission map (1 = background) and the exact area-equivalent radius."""
    yy, xx = _grid()
    body = shape.mask(yy, xx)
    t = np.where(body, DARK, 1.0)
    if shape.spot_ring:
        d = np.hypot(yy - C[0], xx - C[1])
        t = np.where(body & (d < 0.22 * 60), 2.2, t)
        t = np.where(body & (np.abs(d - 0.75 * 60) < 2.0), 0.85, t)
    area = body.reshape(H, SS, W, SS).mean(axis=(1, 3)).sum()
    return t.reshape(H, SS, W, SS).mean(axis=(1, 3)), float(np.sqrt(area / np.pi))


def render_transmission(t: np.ndarray, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = cv2.GaussianBlur(BG * t, (0, 0), 1.2)
    img = img + rng.normal(0, 0.01 * BG, img.shape)
    return np.clip(img, 0, 65535).astype(np.uint16)


def reference(seed: int = 99) -> np.ndarray:
    return render_transmission(np.ones((H, W)), seed)


def render_shape(shape: Shape, seed: int = 0) -> tuple[np.ndarray, float]:
    """uint16 frame of *shape* and its true area-equivalent radius."""
    t, r_true = transmission(shape)
    return render_transmission(t, seed), r_true


def render_masks(masks: list[tuple[Mask, float]], seed: int = 0) -> tuple[np.ndarray, list[float]]:
    """Frame with several dark bodies ``(mask, transmission)``; returns each body's exact R_eq."""
    yy, xx = _grid()
    t = np.ones(yy.shape)
    radii = []
    for mask, trans in masks:
        body = mask(yy, xx)
        t = np.where(body, trans, t)
        radii.append(float(np.sqrt(body.reshape(H, SS, W, SS).mean(axis=(1, 3)).sum() / np.pi)))
    return render_transmission(t.reshape(H, SS, W, SS).mean(axis=(1, 3)), seed), radii
