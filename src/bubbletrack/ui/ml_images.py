"""Pictures of ML assist's teaching mode (numpy -> Qt), no controller logic.

* the strokes of a frame as a translucent overlay (green bubble, red background);
* the probability tint of the right panel (P(bubble) in green over the Auto view);
* the small previews of the suggested frames, with ML's outline.
"""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import QColor, QImage, QPainter, QPen, QPixmap, QPolygonF

BUBBLE_COLOUR = "#22c55e"
BACKGROUND_COLOUR = "#ef4444"
OUTLINE_COLOUR = "#ef4444"  # the measured outline, as for every fit
STROKE_OVERLAY_ALPHA = 120
TINT_STRENGTH = 0.65  # tint opacity at P = 1
VIEW_WHITE = 1.25  # Auto view value shown as white (as in the Auto view panel)


def _rgb(colour: str) -> np.ndarray:
    return np.array([int(colour[i:i + 2], 16) for i in (1, 3, 5)], dtype=float)


def stroke_overlay(shape: tuple[int, int], bubble, background) -> np.ndarray | None:
    """(H, W, 4) uint8 overlay of painted pixels (1-based (row, col) sets); None if none."""
    if not bubble and not background:
        return None
    h, w = int(shape[0]), int(shape[1])
    out = np.zeros((h, w, 4), np.uint8)
    for pixels, colour in ((bubble, BUBBLE_COLOUR), (background, BACKGROUND_COLOUR)):
        if not pixels:
            continue
        px = np.array(sorted(pixels), dtype=np.int64) - 1
        px = px[(px[:, 0] < h) & (px[:, 1] < w)]
        out[px[:, 0], px[:, 1], :3] = _rgb(colour).astype(np.uint8)
        out[px[:, 0], px[:, 1], 3] = STROKE_OVERLAY_ALPHA
    return out


def view_grey(view: np.ndarray) -> np.ndarray:
    """The Auto view (background ~1, bubble ~0) as a [0, 1] display image."""
    return np.clip(np.nan_to_num(np.asarray(view, np.float32), nan=1.0) / VIEW_WHITE, 0.0, 1.0)


def tint_image(grey: np.ndarray, probability: np.ndarray) -> np.ndarray:
    """(H, W, 3) uint8: *grey* [0, 1] with P(bubble) shown as a green tint."""
    g = np.clip(np.asarray(grey, np.float32), 0.0, 1.0)[..., None] * 255.0
    a = TINT_STRENGTH * np.clip(np.asarray(probability, np.float32), 0.0, 1.0)[..., None]
    out = (1.0 - a) * g + a * _rgb(BUBBLE_COLOUR)[None, None, :]
    return np.clip(out, 0, 255).astype(np.uint8)


def _qimage_grey(grey: np.ndarray) -> QImage:
    img8 = np.ascontiguousarray((np.clip(grey, 0.0, 1.0) * 255).astype(np.uint8))
    h, w = img8.shape
    return QImage(img8.tobytes(), w, h, w, QImage.Format.Format_Grayscale8).copy()


def thumbnail(grey: np.ndarray, outline: np.ndarray | None, size: tuple[int, int]) -> QPixmap:
    """A small preview: *grey* [0, 1] scaled into *size* (w, h), *outline* (N, 2)
    [row, col] in the image's coordinates as the image panels draw them
    (ROI-local, ``xy - (gridx[0] - 1, gridy[0] - 1)``) on top."""
    image = _qimage_grey(grey)
    pix = QPixmap.fromImage(image).scaled(size[0], size[1], Qt.AspectRatioMode.KeepAspectRatio,
                                          Qt.TransformationMode.SmoothTransformation)
    if outline is None or len(outline) < 3 or image.width() == 0:
        return pix
    scale = pix.width() / image.width()
    poly = QPolygonF([QPointF(c * scale, r * scale)
                      for r, c in np.asarray(outline, float).reshape(-1, 2)])
    painter = QPainter(pix)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(QPen(QColor(OUTLINE_COLOUR), 1.5))
    painter.drawPolygon(poly)
    painter.end()
    return pix
