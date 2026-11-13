"""Pictures of ML assist's teaching mode (numpy -> Qt), no controller logic.

* the strokes of a frame as a translucent overlay (green bubble, red background);
* the cards' small pictures: the frame cropped around the outline the user
  judges (the outline in red), or around the strokes painted on it.
"""

from __future__ import annotations

import math

import numpy as np
from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import QColor, QImage, QPainter, QPen, QPixmap, QPolygonF

BUBBLE_COLOUR = "#22c55e"
BACKGROUND_COLOUR = "#ef4444"
OUTLINE_COLOUR = "#ef4444"  # the measured outline, as for every fit
STROKE_OVERLAY_ALPHA = 120
PAINT_ALPHA = 0.55  # painted pixels on a card's picture
CROP_RADII = 1.8  # a card's picture spans this many radii around the centre
MIN_CROP_HALF = 12  # px
STRETCH_PERCENTILES = (1.0, 99.0)  # contrast of a card's picture


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


def crop_box(shape: tuple[int, int], rc: float, cc: float, radius: float
             ) -> tuple[tuple[int, int], tuple[int, int]]:
    """((r0, r1), (c0, c1)) 0-based slice bounds around the 1-based circle (rc, cc, R)."""
    h, w = int(shape[0]), int(shape[1])
    half = max(CROP_RADII * float(radius), MIN_CROP_HALF)

    def span(centre: float, size: int) -> tuple[int, int]:
        lo = int(np.clip(math.floor(centre - 1 - half), 0, size - 1))
        return lo, int(np.clip(math.ceil(centre - 1 + half) + 1, lo + 1, size))

    return span(rc, h), span(cc, w)


def stretch(image: np.ndarray) -> np.ndarray:
    """*image* scaled to [0, 1] between its 1st and 99th percentiles."""
    img = np.asarray(image, dtype=np.float32)
    lo, hi = np.percentile(img, STRETCH_PERCENTILES) if img.size else (0.0, 1.0)
    return np.clip((img - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0)


def paint_pixels(grey: np.ndarray, bubble, background, *, origin: tuple[int, int]) -> np.ndarray:
    """(H, W, 3) uint8: *grey* [0, 1] with the painted pixels (1-based full-image
    (row, col) sets) tinted; *origin* is the crop's 0-based (row, col)."""
    out = np.repeat((np.clip(grey, 0.0, 1.0) * 255.0)[..., None], 3, axis=2)
    h, w = grey.shape
    for pixels, colour in ((bubble, BUBBLE_COLOUR), (background, BACKGROUND_COLOUR)):
        if not pixels:
            continue
        px = np.array(sorted(pixels), dtype=np.int64) - 1 - np.array(origin, dtype=np.int64)
        px = px[(px[:, 0] >= 0) & (px[:, 0] < h) & (px[:, 1] >= 0) & (px[:, 1] < w)]
        out[px[:, 0], px[:, 1]] = (1 - PAINT_ALPHA) * out[px[:, 0], px[:, 1]] + PAINT_ALPHA * _rgb(colour)
    return np.clip(out, 0, 255).astype(np.uint8)


def _qimage(image: np.ndarray) -> QImage:
    """A grey [0, 1] (H, W) or an RGB uint8 (H, W, 3) array as a QImage."""
    if image.ndim == 2:
        img8 = np.ascontiguousarray((np.clip(image, 0.0, 1.0) * 255).astype(np.uint8))
        h, w = img8.shape
        return QImage(img8.tobytes(), w, h, w, QImage.Format.Format_Grayscale8).copy()
    rgb = np.ascontiguousarray(image.astype(np.uint8))
    h, w = rgb.shape[:2]
    return QImage(rgb.tobytes(), w, h, 3 * w, QImage.Format.Format_RGB888).copy()


def thumbnail(image: np.ndarray, outline: np.ndarray | None, size: tuple[int, int]) -> QPixmap:
    """A small picture: *image* (grey [0, 1] or RGB uint8) scaled into *size* (w, h),
    with *outline* (N, 2) [row, col], in the image's pixel coordinates, in red."""
    qimage = _qimage(image)
    pix = QPixmap.fromImage(qimage).scaled(size[0], size[1], Qt.AspectRatioMode.KeepAspectRatio,
                                           Qt.TransformationMode.SmoothTransformation)
    if outline is None or len(outline) < 3 or qimage.width() == 0:
        return pix
    scale = pix.width() / qimage.width()
    poly = QPolygonF([QPointF(c * scale, r * scale)
                      for r, c in np.asarray(outline, float).reshape(-1, 2)])
    painter = QPainter(pix)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(QPen(QColor(OUTLINE_COLOUR), 1.5))
    painter.drawPolygon(poly)
    painter.end()
    return pix
