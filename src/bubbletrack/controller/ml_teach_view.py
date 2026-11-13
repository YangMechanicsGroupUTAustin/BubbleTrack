"""What the teaching mode draws: strokes on the frame, ML's view on the right
panel, the previews of the suggested frames and the one-line description."""

from __future__ import annotations

import numpy as np
from PyQt6.QtGui import QPixmap

from bubbletrack.controller.ml_preview import FramePreview
from bubbletrack.model.ml_pipeline import FLAG_DISAGREES, FLAG_UNFAMILIAR
from bubbletrack.model.teach import MLExamples
from bubbletrack.ui.image_compare import CompareMode
from bubbletrack.ui.ml_images import OUTLINE_COLOUR, stroke_overlay, thumbnail, tint_image, view_grey
from bubbletrack.ui.ml_teach_panel import THUMB_SIZE

PREVIEW_TITLE = "ML assist: P(bubble)"
SOURCE_TEXT = {"auto_ml": "Auto (ML-guided)", "auto": "Auto", "ml": "ML outline", "none": "no result"}


def roi_origin(state) -> np.ndarray:
    """Subtract from 1-based full-image points to draw them on the ROI panels."""
    return np.array([state.gridx[0] - 1, state.gridy[0] - 1], dtype=float)


def describe(preview: FramePreview, protected: frozenset[int]) -> tuple[str, bool]:
    """(text, warn) for the preview line, e.g. 'Frame 12: ML outline, R = 38.0 px ...'."""
    fit, frame = preview.fit, preview.frame
    if fit.ml_source == "none":
        text = f"Frame {frame + 1}: ML assist finds no bubble"
    else:
        text = (f"Frame {frame + 1}: {SOURCE_TEXT[fit.ml_source]}, R = {fit.fit.radius:.1f} px "
                f"(confidence {fit.fit.confidence:.2f})")
        if fit.ml_source == "ml":
            text += " — goes to Review"
        elif FLAG_DISAGREES in fit.fit.flags:
            text += " — ML sees the bubble elsewhere (goes to Review)"
    if frame in protected:
        text += " — your result here is kept"
    unfamiliar = FLAG_UNFAMILIAR in fit.fit.flags
    if unfamiliar:
        text += " — this frame looks unlike the taught ones"
    return text, unfamiliar


def draw_strokes(window, state, idx: int) -> None:
    """The painted pixels of frame *idx* over the left image."""
    ex = state.ml_examples
    strokes = ex.strokes.get(idx) if isinstance(ex, MLExamples) else None
    if strokes is None or state.cur_img is None:
        return
    overlay = stroke_overlay(state.cur_img.shape[:2], strokes.bubble, strokes.background)
    if overlay is not None:
        window.original_panel.draw_rgba_overlay(overlay)


def show_preview(window, state, preview: FramePreview) -> None:
    """The right panel: the Auto view tinted by P(bubble), with the outline ML assist stores."""
    if window.compare_mode is not CompareMode.SIDE_BY_SIDE:
        return
    grey = view_grey(preview.view)
    rgb = (tint_image(grey, preview.probability) if preview.probability is not None
           else np.repeat((grey[..., None] * 255).astype(np.uint8), 3, axis=2))
    panel = window.binary_panel
    panel.set_title(PREVIEW_TITLE)
    panel.set_image_rgb(rgb)
    fit = preview.fit.fit
    if fit.radius > 0 and len(fit.edge_xy):
        panel.draw_fit_outline(np.asarray(fit.edge_xy, float) - roi_origin(state), OUTLINE_COLOUR)


def card(state, frame: int, preview: FramePreview | None) -> tuple[str, QPixmap | None]:
    """(caption, preview picture) of a suggested frame."""
    caption = f"Frame {frame + 1}"
    if preview is None:
        return caption, None
    fit = preview.fit
    if fit.fit.radius > 0:
        caption += f" · {SOURCE_TEXT[fit.ml_source]} {fit.fit.radius:.0f} px"
        xy = np.asarray(fit.fit.edge_xy, float) - roi_origin(state)
    else:
        caption += " · no result"
        xy = None
    return caption, thumbnail(view_grey(preview.view), xy, THUMB_SIZE)
