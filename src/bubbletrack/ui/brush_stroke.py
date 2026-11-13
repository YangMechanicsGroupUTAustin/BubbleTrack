"""Press-drag-release brush gesture on an image scene (ML assist's teaching mode).

A ring follows the cursor at the brush size; while the button is down the
stroke is drawn as a translucent round-capped band.  On release the caller
gets the stroke's points in scene coordinates (x = col, y = row) and turns
them into labelled pixels.
"""

from __future__ import annotations

from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import QColor, QPainterPath, QPen
from PyQt6.QtWidgets import QGraphicsEllipseItem, QGraphicsPathItem, QGraphicsScene

STROKE_ALPHA = 110  # 0-255: the band shows the frame underneath
_Z = 11  # above the fit overlays and the circle preview


class BrushStroke:
    """One brush (colour, radius in scene pixels) and the stroke being drawn."""

    def __init__(self, scene: QGraphicsScene) -> None:
        self._scene = scene
        self.radius = 5.0
        self.colour = "#22c55e"
        self._points: list[QPointF] = []
        self._band: QGraphicsPathItem | None = None
        self._ring: QGraphicsEllipseItem | None = None

    @property
    def active(self) -> bool:
        return bool(self._points)

    @property
    def points(self) -> list[tuple[float, float]]:
        """The stroke so far, scene (x, y)."""
        return [(p.x(), p.y()) for p in self._points]

    def set_brush(self, colour: str, radius: float) -> None:
        self.colour, self.radius = colour, max(0.5, float(radius))
        if self._ring is not None:
            self._ring.setPen(self._ring_pen())
            self._place_ring(self._ring.rect().center())

    # -- hover ring ------------------------------------------------------------ #

    def hover(self, scene_pos: QPointF) -> None:
        if self._ring is None or self._ring.scene() is not self._scene:
            self._ring = self._scene.addEllipse(0, 0, 0, 0, self._ring_pen())
            self._ring.setZValue(_Z + 1)
        self._place_ring(scene_pos)

    def hide_hover(self) -> None:
        if self._ring is not None and self._ring.scene() is self._scene:
            self._scene.removeItem(self._ring)
        self._ring = None

    def _place_ring(self, c: QPointF) -> None:
        r = self.radius
        self._ring.setRect(c.x() - r, c.y() - r, 2 * r, 2 * r)

    def _ring_pen(self) -> QPen:
        pen = QPen(QColor(self.colour), 1.2)
        pen.setCosmetic(True)
        return pen

    # -- the stroke ------------------------------------------------------------- #

    def press(self, scene_pos: QPointF) -> None:
        self.cancel()
        self._points = [QPointF(scene_pos)]
        colour = QColor(self.colour)
        colour.setAlpha(STROKE_ALPHA)
        pen = QPen(colour, 2 * self.radius)  # scene width: scales with the zoom like the pixels
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        self._band = QGraphicsPathItem()
        self._band.setPen(pen)
        self._band.setZValue(_Z)
        self._scene.addItem(self._band)
        self._redraw()

    def move(self, scene_pos: QPointF) -> None:
        if not self._points:
            return
        last = self._points[-1]
        if abs(scene_pos.x() - last.x()) + abs(scene_pos.y() - last.y()) >= 0.5:
            self._points.append(QPointF(scene_pos))
            self._redraw()
        if self._ring is not None:
            self._place_ring(scene_pos)

    def release(self, scene_pos: QPointF) -> list[tuple[float, float]] | None:
        """End the stroke; its points (scene x, y), or None if none was running."""
        if not self._points:
            return None
        self.move(scene_pos)
        points = self.points
        self.cancel()
        return points

    def cancel(self) -> None:
        if self._band is not None and self._band.scene() is self._scene:
            self._scene.removeItem(self._band)
        self._band = None
        self._points = []

    def _redraw(self) -> None:
        path = QPainterPath(self._points[0])
        if len(self._points) == 1:  # a dot: a zero-length segment with round caps
            path.lineTo(self._points[0] + QPointF(0.01, 0.0))
        for p in self._points[1:]:
            path.lineTo(p)
        self._band.setPath(path)
