"""What the teaching mode draws: the painted strokes over the frame, and the
cards' pictures (the whole frame with the outline the user judges, in red,
or with the strokes painted on it)."""

from __future__ import annotations

import logging

import numpy as np
from PyQt6.QtGui import QPixmap

from bubbletrack.controller.display_mixin import current_source
from bubbletrack.model import edge_pipeline
from bubbletrack.model.image_io import to_grayscale
from bubbletrack.model.teach import MLExamples
from bubbletrack.ui.ml_images import (
    BACKGROUND_COLOUR, BUBBLE_COLOUR, paint_pixels, stretch, stroke_overlay, thumbnail,
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


def card_picture(state, frame: int, outline, strokes) -> QPixmap | None:
    """The whole frame (the full field of view, never a crop: the owner judges the
    outline in context) with *outline* in red, or the painted *strokes*."""
    painted = strokes is not None and not strokes.is_empty
    if outline is None and not painted:
        return None
    try:
        grey = to_grayscale(edge_pipeline.read_raw(current_source(state, frame)))
    except Exception as exc:  # noqa: BLE001 - a card without a picture, never fatal
        logger.warning("No picture for frame %d: %s", frame + 1, exc)
        return None
    image = stretch(np.asarray(grey, dtype=np.float32))
    if painted:
        image = paint_pixels(image, strokes.bubble, strokes.background, origin=(0, 0))
    # 1-based [row, col] -> pixel coordinates of the picture (pixel centres at +0.5).
    xy = None if outline is None else _points(outline) - 0.5
    return thumbnail(image, xy, THUMB_SIZE)
