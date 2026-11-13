"""What the teaching mode draws: the painted strokes over the frame, and the
cards' pictures (the frame around the outline the user judges, in red, or
around the strokes painted on it)."""

from __future__ import annotations

import logging

import numpy as np
from PyQt6.QtGui import QPixmap

from bubbletrack.controller.display_mixin import current_source
from bubbletrack.model import edge_pipeline
from bubbletrack.model.image_io import to_grayscale
from bubbletrack.model.teach import MLExamples
from bubbletrack.ui.ml_images import (
    BACKGROUND_COLOUR, BUBBLE_COLOUR, crop_box, paint_pixels, stretch, stroke_overlay, thumbnail,
)
from bubbletrack.ui.ml_teach_panel import THUMB_SIZE
from bubbletrack.ui.overlay_legend import LegendEntry

logger = logging.getLogger(__name__)

# Colour key lines of the painted strokes.
PAINTED_BUBBLE_KEY = LegendEntry("painted: bubble", BUBBLE_COLOUR, "fill")
PAINTED_BACKGROUND_KEY = LegendEntry("painted: background", BACKGROUND_COLOUR, "fill")
_CIRCLE_POINTS = 72


def draw_strokes(window, state, idx: int) -> None:
    """The painted pixels of frame *idx* over the left image."""
    ex = state.ml_examples
    strokes = ex.strokes.get(idx) if isinstance(ex, MLExamples) else None
    if strokes is None or state.cur_img is None:
        return
    overlay = stroke_overlay(state.cur_img.shape[:2], strokes.bubble, strokes.background)
    if overlay is not None:
        keys = ((PAINTED_BUBBLE_KEY,) if strokes.bubble else ()) + \
            ((PAINTED_BACKGROUND_KEY,) if strokes.background else ())
        window.original_panel.draw_rgba_overlay(overlay, keys)


def _points(outline) -> np.ndarray:
    """The outline's (N, 2) 1-based [row, col] points (its circle when it has none)."""
    if outline.xy is not None and len(outline.xy) >= 3:
        return np.asarray(outline.xy, dtype=float).reshape(-1, 2)
    a = np.linspace(0.0, 2.0 * np.pi, _CIRCLE_POINTS, endpoint=False)
    return np.column_stack([outline.rc + outline.radius * np.sin(a),
                            outline.cc + outline.radius * np.cos(a)])


def _focus(outline, strokes) -> tuple[float, float, float] | None:
    """(rc, cc, R), 1-based: the circle the picture centres on."""
    if outline is not None:
        return outline.rc, outline.cc, outline.radius
    if strokes is None or strokes.is_empty:
        return None
    every = np.array(sorted(strokes.bubble | strokes.background), dtype=float)
    core = np.array(sorted(strokes.bubble), dtype=float) if strokes.bubble else every
    rc, cc = core.mean(axis=0)
    return float(rc), float(cc), float(np.hypot(*(every - (rc, cc)).T).max() / 1.5 + 1.0)


def card_picture(state, frame: int, outline, strokes) -> QPixmap | None:
    """The frame cropped around *outline* (drawn in red) or the painted *strokes*."""
    focus = _focus(outline, strokes)
    if focus is None:
        return None
    try:
        grey = to_grayscale(edge_pipeline.read_raw(current_source(state, frame)))
    except Exception as exc:  # noqa: BLE001 - a card without a picture, never fatal
        logger.warning("No picture for frame %d: %s", frame + 1, exc)
        return None
    (r0, r1), (c0, c1) = crop_box(grey.shape[:2], *focus)
    image = stretch(np.asarray(grey, dtype=np.float32)[r0:r1, c0:c1])
    if strokes is not None and not strokes.is_empty:
        image = paint_pixels(image, strokes.bubble, strokes.background, origin=(r0, c0))
    xy = None if outline is None else _points(outline) - (r0 + 0.5, c0 + 0.5)
    return thumbnail(image, xy, THUMB_SIZE)
