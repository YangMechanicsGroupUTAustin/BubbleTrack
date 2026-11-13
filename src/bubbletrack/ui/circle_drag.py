"""Press-drag-release circle gesture on an image scene, with a live dashed preview.

Press on the bubble centre, drag to its wall, release.  The preview is a
dashed circle plus a small cross at the centre; it is removed on release
(the caller draws the measured result).  A release within ``CLICK_SLOP_PX``
screen pixels of the press is a click: the user gave a centre, no size.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import QColor, QPainterPath, QPen
from PyQt6.QtWidgets import QGraphicsEllipseItem, QGraphicsPathItem, QGraphicsScene

CLICK_SLOP_PX = 3.0  # screen pixels: a smaller drag is a click
PREVIEW_COLOUR = "#f59e0b"
_CROSS_PX = 4.0  # half-size of the centre cross, scene pixels


def dashed_pen(colour: str = PREVIEW_COLOUR, width: float = 1.5) -> QPen:
    """A dashed cosmetic pen (constant screen width at any zoom)."""
    pen = QPen(QColor(colour), width)
    pen.setCosmetic(True)
    pen.setStyle(Qt.PenStyle.DashLine)
    return pen


@dataclass(frozen=True)
class DrawnCircle:
    """The finished gesture, in scene coordinates (x = col, y = row)."""

    x: float
    y: float
    radius: float
    dragged: bool  # False: a click (radius 0)


class CircleDrag:
    """State of one gesture; the view forwards press / move / release to it."""

    def __init__(self, scene: QGraphicsScene) -> None:
        self._scene = scene
        self._centre: QPointF | None = None
        self._press_view: QPointF | None = None
        self._circle: QGraphicsEllipseItem | None = None
        self._cross: QGraphicsPathItem | None = None

    @property
    def active(self) -> bool:
        return self._centre is not None

    @property
    def preview(self) -> QGraphicsEllipseItem | None:
        """The live preview circle while dragging (None otherwise)."""
        return self._circle

    def press(self, scene_pos: QPointF, view_pos: QPointF) -> None:
        self.cancel()
        self._centre, self._press_view = QPointF(scene_pos), QPointF(view_pos)
        pen = dashed_pen()
        self._circle = self._scene.addEllipse(scene_pos.x(), scene_pos.y(), 0.0, 0.0, pen)
        path = QPainterPath()
        x, y = scene_pos.x(), scene_pos.y()
        path.moveTo(x - _CROSS_PX, y)
        path.lineTo(x + _CROSS_PX, y)
        path.moveTo(x, y - _CROSS_PX)
        path.lineTo(x, y + _CROSS_PX)
        solid = QPen(pen)
        solid.setStyle(Qt.PenStyle.SolidLine)
        self._cross = self._scene.addPath(path, solid)
        for item in (self._circle, self._cross):
            item.setZValue(10)  # above the fit overlays

    def move(self, scene_pos: QPointF) -> None:
        if self._centre is None or self._circle is None:
            return
        r = self._radius(scene_pos)
        c = self._centre
        self._circle.setRect(c.x() - r, c.y() - r, 2 * r, 2 * r)

    def release(self, scene_pos: QPointF, view_pos: QPointF) -> DrawnCircle | None:
        """End the gesture and remove the preview; None if none was running."""
        if self._centre is None or self._press_view is None:
            return None
        moved = math.hypot(view_pos.x() - self._press_view.x(), view_pos.y() - self._press_view.y())
        dragged = moved >= CLICK_SLOP_PX
        centre = self._centre
        radius = self._radius(scene_pos) if dragged else 0.0
        self.cancel()
        return DrawnCircle(centre.x(), centre.y(), radius, dragged and radius > 0)

    def cancel(self) -> None:
        """Drop a running gesture and its preview."""
        for item in (self._circle, self._cross):
            if item is not None and item.scene() is self._scene:
                self._scene.removeItem(item)
        self._circle = self._cross = None
        self._centre = self._press_view = None

    def _radius(self, scene_pos: QPointF) -> float:
        c = self._centre
        return math.hypot(scene_pos.x() - c.x(), scene_pos.y() - c.y())
