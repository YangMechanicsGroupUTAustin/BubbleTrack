"""Training pixels of one frame: the user's strokes plus automatic examples from a
verified circle (inside ``inner * R`` = bubble; a ring ``ring * R`` and everything
beyond ``far * R`` = background; the wall zone in between is left unlabelled)."""

from __future__ import annotations

import numpy as np

from bubbletrack.model.teach.config import ExampleConfig
from bubbletrack.model.teach.examples import BACKGROUND, BUBBLE, Circle, MLExamples

_EMPTY = np.empty((0, 2), np.int64)


def circle_masks(circle: Circle, shape: tuple[int, int],
                 config: ExampleConfig = ExampleConfig()) -> tuple[np.ndarray, np.ndarray]:
    """(bubble, background) bool masks of a full frame for a verified circle.

    *circle* is ``(rc, cc, R)`` in 1-based full-image coordinates.
    """
    rc, cc, r = circle
    yy, xx = np.ogrid[0:shape[0], 0:shape[1]]
    d = np.hypot(yy - (rc - 1.0), xx - (cc - 1.0))
    bubble = d < config.inner * r
    background = (d > config.ring[0] * r) & (d < config.ring[1] * r)
    if config.far is not None:
        background |= d > config.far * r
    return bubble, background


def _inside(px, shape: tuple[int, int]) -> np.ndarray:
    """1-based pixel set -> 0-based (N, 2) array of those inside the frame."""
    if not px:
        return _EMPTY
    arr = np.array(sorted(px), np.int64) - 1
    keep = (arr[:, 0] < shape[0]) & (arr[:, 1] < shape[1])
    return arr[keep]


def _take(rng: np.random.Generator, pixels: np.ndarray, n: int) -> np.ndarray:
    if n <= 0:
        return _EMPTY
    if len(pixels) <= n:
        return pixels
    return pixels[np.sort(rng.choice(len(pixels), n, replace=False))]


def training_pixels(examples: MLExamples, frame: int, shape: tuple[int, int],
                    config: ExampleConfig = ExampleConfig()) -> tuple[np.ndarray, np.ndarray]:
    """(bubble, background) training pixels of *frame*, 0-based full-image (N, 2).

    Strokes win over automatic examples (a painted pixel is never given the
    other label) and are kept first; automatic pixels fill each class up to
    ``config.max_per_class`` by a deterministic subsample.
    """
    rng = np.random.default_rng([config.seed, int(frame)])
    strokes = examples.strokes.get(frame)
    painted = {lab: _inside(strokes.of(lab), shape) if strokes else _EMPTY for lab in (BUBBLE, BACKGROUND)}
    auto = {BUBBLE: _EMPTY, BACKGROUND: _EMPTY}
    if frame in examples.verified:
        masks = dict(zip((BUBBLE, BACKGROUND), circle_masks(examples.verified[frame], shape, config)))
        for lab, mask in masks.items():
            for other in painted.values():          # painted pixels are never re-labelled
                mask[other[:, 0], other[:, 1]] = False
            auto[lab] = np.argwhere(mask)
    cap = int(config.max_per_class)
    out = []
    for lab in (BUBBLE, BACKGROUND):
        keep = _take(rng, painted[lab], cap)
        fill = _take(rng, auto[lab], cap - len(keep))
        out.append(np.concatenate([keep, fill]) if len(fill) else keep)
    return out[0], out[1]
