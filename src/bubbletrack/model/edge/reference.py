"""Automatic choice of the bubble-free reference frame.

High-speed cavitation recordings start with a few bubble-free frames, then the
plasma flash, then the bubble.  Frame 1 is often abnormal (brighter, with a
glow).  The best reference is the last normal frame before the flash or the
bubble: it has the same illumination as the bubble frames that follow.

Rules, on the first ``search`` frames:

1. The *event* is the first frame (from frame index 2 on) that changed
   structurally from its predecessor (flash glow, bubble), or that is much
   brighter than it (flash).  A change between frames 0 and 1 is blamed on
   the abnormal first frame unless frame 1 is a flash.
2. Walking back from the frame before the event, the first frame that is
   *normal* (brightness consistent with the other pre-event frames; frame 0
   only if it matches frame 1) and has *no bubble* (the reference-free engine
   finds no confident circle) is returned.
"""

from __future__ import annotations

import logging
from typing import Callable

import cv2
import numpy as np

from bubbletrack.model.edge.config import DEFAULT_CONFIG, EdgeConfig, ReferenceConfig
from bubbletrack.model.edge.engine import EdgeParams, detect_edge_with_config
from bubbletrack.model.edge.normalize import to_gray_frame

logger = logging.getLogger(__name__)


def _read(read_raw: Callable[[int], np.ndarray], i: int) -> np.ndarray | None:
    try:
        raw = read_raw(i)
        to_gray_frame(raw)                      # validates the layout and dtype
        return raw
    except Exception as exc:                    # one unreadable frame must not stop the search
        logger.warning("Reference search: frame %d unusable (%s)", i, exc)
        return None


def _thumbnail(raw: np.ndarray, cfg: ReferenceConfig) -> tuple[np.ndarray, float]:
    """Shrunk, smoothed, gain-normalised frame and its median brightness."""
    values = to_gray_frame(raw).values
    h, w = values.shape
    f = cfg.downsample
    small = cv2.resize(values, (max(w // f, 1), max(h // f, 1)), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), cfg.blur_sigma)
    level = float(np.median(values))
    return small / max(float(np.median(small)), 1e-6), level


def _changed_fraction(prev: np.ndarray, cur: np.ndarray, cfg: ReferenceConfig) -> float:
    ratio = cur / np.maximum(prev, 1e-6)
    return float(np.mean((ratio > cfg.change_ratio) | (ratio < 1.0 / cfg.change_ratio)))


def _first_event(thumbs: list, levels: list, cfg: ReferenceConfig) -> int:
    for j in range(1, len(thumbs)):
        if thumbs[j] is None or thumbs[j - 1] is None:
            continue
        if levels[j] > cfg.flash_brightness_ratio * levels[j - 1]:
            return j
        if j >= 2 and _changed_fraction(thumbs[j - 1], thumbs[j], cfg) > cfg.event_change_frac:
            return j
    return len(thumbs)


def _is_normal(c: int, event: int, thumbs: list, levels: list, cfg: ReferenceConfig) -> bool:
    tol = cfg.brightness_tolerance
    if c == 0:
        if len(thumbs) < 2 or thumbs[1] is None:
            return False
        return (abs(levels[1] / max(levels[0], 1e-6) - 1.0) <= tol
                and _changed_fraction(thumbs[0], thumbs[1], cfg) <= cfg.event_change_frac)
    pre = [levels[i] for i in range(1, event) if thumbs[i] is not None]
    typical = float(np.median(pre))
    return abs(levels[c] / max(typical, 1e-6) - 1.0) <= tol


_BUBBLE_TEST_PARAMS = EdgeParams(non_round=True)
"""Whether a frame holds a bubble does not depend on the radius definition: the area
definition's confidence (shape-agnostic) is the one this test was calibrated with."""


def _has_bubble(raw: np.ndarray, config: EdgeConfig) -> bool:
    """A confident bubble of a plausible size (one the area mode could not measure because
    the image edge cuts it is still a bubble)."""
    h, w = raw.shape[:2]
    res = detect_edge_with_config(raw, (1, h), (1, w), reference=None, params=_BUBBLE_TEST_PARAMS,
                                  config=config)
    confident = bool(np.isfinite(res.radius) and res.confidence >= res.confidence_gate)
    return confident and res.radius >= config.reference.min_bubble_radius_px


def choose_reference_frame(read_raw: Callable[[int], np.ndarray], n_frames: int, *,
                           search: int = 10) -> int | None:
    """Pick the bubble-free reference frame among the first *search* frames.

    Parameters
    ----------
    read_raw : callable
        ``read_raw(i)`` returns the full frame with 0-based index *i* (any
        dtype, 2-D or 3/4-channel).  A frame that raises or is malformed is
        skipped.
    n_frames : int
        Number of frames in the sequence.
    search : int
        How many leading frames to consider.

    Returns
    -------
    int or None
        0-based index of the last normal, bubble-free frame before the flash
        or the bubble; None when no such frame exists (e.g. the recording
        starts with a bubble).

    Raises
    ------
    ValueError
        If *n_frames* is negative or *search* is not positive.
    """
    return choose_reference(read_raw, n_frames, search=search, config=DEFAULT_CONFIG)


def choose_reference(read_raw: Callable[[int], np.ndarray], n_frames: int, *, search: int,
                     config: EdgeConfig) -> int | None:
    """:func:`choose_reference_frame` with an explicit :class:`EdgeConfig`."""
    if int(n_frames) < 0:
        raise ValueError(f"n_frames must be >= 0, got {n_frames}")
    if int(search) < 1:
        raise ValueError(f"search must be >= 1, got {search}")
    cfg = config.reference
    k = min(int(search), int(n_frames))
    raws = [_read(read_raw, i) for i in range(k)]
    shape = next((r.shape[:2] for r in raws if r is not None), None)
    raws = [r if r is not None and r.shape[:2] == shape else None for r in raws]
    thumbs, levels = [], []
    for raw in raws:
        thumb, level = _thumbnail(raw, cfg) if raw is not None else (None, float("nan"))
        thumbs.append(thumb)
        levels.append(level)
    event = _first_event(thumbs, levels, cfg)
    for c in range(event - 1, -1, -1):
        if raws[c] is None or not _is_normal(c, event, thumbs, levels, cfg):
            continue
        if _has_bubble(raws[c], config):
            logger.info("Reference search: frame %d holds a bubble", c)
            continue
        return c
    return None
