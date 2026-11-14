"""Overlay items of an image panel (outlines, circles, ROI) and their colour key.

Every ``draw_*`` call adds its overlay's key line to the panel's legend: by
default the meaning of a known colour (``overlay_legend``), or the
``legend=`` text given; ``legend=""`` draws without a key line.
"""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QBrush, QColor, QImage, QPainterPath, QPen, QPixmap
from PyQt6.QtWidgets import QGraphicsPathItem, QGraphicsScene, QGraphicsView

from bubbletrack.model.conventions import roi_to_slice
from bubbletrack.ui.circle_drag import dashed_pen
from bubbletrack.ui.contour_shape import is_ordered_contour
from bubbletrack.ui.overlay_legend import (
    CIRCLE_COLOUR, OUTLINE_COLOUR, ROI_COLOUR, LegendEntry, OverlayLegend, make_entry,
)


def cosmetic_pen(colour: str, width: float) -> QPen:
    """A pen whose width is in screen pixels, so overlays stay visible at any zoom."""
    pen = QPen(QColor(colour), width)
    pen.setCosmetic(True)
    return pen


class OverlayDrawingMixin:
    """Drawing API of :class:`~bubbletrack.ui.image_panel.ImagePanel`.

    The host provides ``_scene`` and ``_view`` and calls :meth:`_init_overlays`.
    Coordinates are scene pixels, given as (row, col) = (y, x).
    """

    _scene: QGraphicsScene
    _view: QGraphicsView

    def _init_overlays(self) -> None:
        self._overlay_items: list = []
        self._legend = OverlayLegend(self._view)

    def draw_circle(self, row_c: float, col_c: float, radius: float,
                    colour: str = CIRCLE_COLOUR, width: float = 2.0, *, dashed: bool = False,
                    legend: str | None = None) -> None:
        """Draw a circle overlay. Coordinates are (row, col) = (y, x)."""
        item = self._scene.addEllipse(
            col_c - radius, row_c - radius, 2 * radius, 2 * radius,
            dashed_pen(colour, width) if dashed else cosmetic_pen(colour, width),
        )
        self._keep(item, colour, "dash" if dashed else "line", legend)

    def draw_points(self, points: np.ndarray, colour: str = OUTLINE_COLOUR, size: float = 3.0,
                    *, legend: str | None = None) -> None:
        """Draw scatter points. *points* is (N, 2) ``[row, col]``.

        All points form one path item, so thousands of edge points stay cheap.
        """
        pts = np.asarray(points, dtype=float).reshape(-1, 2)
        if len(pts) == 0:
            return
        path = QPainterPath()
        half = size / 2
        for r, c in pts:
            path.addEllipse(c - half, r - half, size, size)
        self._add_path(path, QPen(Qt.PenStyle.NoPen), QBrush(QColor(colour)),
                       make_entry(colour, "dots", legend))

    def draw_contour(self, points: np.ndarray, colour: str = OUTLINE_COLOUR,
                     width: float = 1.5, closed: bool = True, *, dashed: bool = False,
                     legend: str | None = None) -> None:
        """Draw *points* (N, 2) ``[row, col]`` as a connected polyline."""
        pts = np.asarray(points, dtype=float).reshape(-1, 2)
        if len(pts) < 2:
            return
        path = QPainterPath()
        path.moveTo(pts[0, 1], pts[0, 0])
        for r, c in pts[1:]:
            path.lineTo(c, r)
        if closed:
            path.closeSubpath()
        pen = dashed_pen(colour, width) if dashed else cosmetic_pen(colour, width)
        self._add_path(path, pen, QBrush(Qt.BrushStyle.NoBrush),
                       make_entry(colour, "dash" if dashed else "line", legend))

    def draw_fit_outline(self, points: np.ndarray | None, colour: str = OUTLINE_COLOUR,
                         *, legend: str | None = None) -> None:
        """Draw a fit's boundary: a closed polyline for an ordered contour,
        otherwise the individual edge points."""
        if points is None or len(points) == 0:
            return
        if is_ordered_contour(points):
            self.draw_contour(points, colour, legend=legend)
        else:
            self.draw_points(points, colour, 2.0, legend=legend)

    def draw_rgba_overlay(self, rgba: np.ndarray, legend: tuple[LegendEntry, ...] = ()) -> None:
        """Lay an (H, W, 4) uint8 image over the frame (scene pixel = array pixel);
        *legend* lists what its colours mean."""
        img = np.ascontiguousarray(rgba, dtype=np.uint8)
        h, w = img.shape[:2]
        qimg = QImage(img.tobytes(), w, h, 4 * w, QImage.Format.Format_RGBA8888)
        item = self._scene.addPixmap(QPixmap.fromImage(qimg))
        item.setZValue(1)
        self._overlay_items.append(item)
        for entry in legend:
            self._legend.add(entry)

    def draw_roi_rect(self, gridx: tuple[int, int], gridy: tuple[int, int],
                      colour: str = ROI_COLOUR, *, legend: str | None = None) -> None:
        """Draw an ROI rectangle (1-indexed coords)."""
        rs, cs = roi_to_slice(gridx, gridy)
        r0, r1 = rs.start, rs.stop
        c0, c1 = cs.start, cs.stop
        item = self._scene.addRect(c0, r0, c1 - c0, r1 - r0, cosmetic_pen(colour, 1.5))
        self._keep(item, colour, "box", legend)

    def add_legend_entry(self, entry: LegendEntry) -> None:
        """Explain part of the image itself (e.g. a tint); cleared with the overlays."""
        self._legend.add(entry)

    def legend_texts(self) -> list[str]:
        """The key lines currently listed, top to bottom."""
        return self._legend.texts()

    def clear_overlays(self) -> None:
        for item in self._overlay_items:
            self._scene.removeItem(item)
        self._overlay_items.clear()
        self._legend.clear()
        self.set_overlay_tip("")

    def set_overlay_tip(self, text: str) -> None:
        """Tooltip of the image area, describing an overlay (cleared with the overlays)."""
        if self._view.toolTip() != text:
            self._view.setToolTip(text)

    def overlay_tip(self) -> str:
        return self._view.toolTip()

    def _add_path(self, path: QPainterPath, pen: QPen, brush: QBrush,
                  entry: LegendEntry | None = None) -> None:
        item = QGraphicsPathItem(path)
        item.setPen(pen)
        item.setBrush(brush)
        self._scene.addItem(item)
        self._overlay_items.append(item)
        self._legend.add(entry)

    def _keep(self, item, colour: str, kind: str, legend: str | None) -> None:
        self._overlay_items.append(item)
        self._legend.add(make_entry(colour, kind, legend))
