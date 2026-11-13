"""QGraphicsView-based image viewer with overlay support."""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal, QRectF
from PyQt6.QtGui import (
    QBrush, QColor, QImage, QMouseEvent, QPainterPath, QPen, QPixmap,
    QResizeEvent, QWheelEvent,
)
from PyQt6.QtWidgets import (
    QGraphicsPathItem, QGraphicsPixmapItem, QGraphicsScene, QGraphicsView,
    QHBoxLayout, QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget,
)

from bubbletrack.model.conventions import roi_to_slice
from bubbletrack.ui.brush_stroke import BrushStroke
from bubbletrack.ui.circle_drag import CircleDrag, dashed_pen
from bubbletrack.ui.contour_shape import is_ordered_contour  # noqa: F401 (public here too)

_VIEW_STYLE = {
    "normal": "border:1px solid rgba(255,255,255,0.06); border-radius:8px; background:#0c0d12;",
    "point": "border:2px solid #6366f1; border-radius:8px; background:#0c0d12;",
    "circle": "border:2px solid #6366f1; border-radius:8px; background:#0c0d12;",
    "paint": "border:2px solid #22c55e; border-radius:8px; background:#0c0d12;",
    "roi": "border:2px solid #f59e0b; border-radius:8px; background:#0c0d12;",
}


def _cosmetic_pen(colour: str, width: float) -> QPen:
    """A pen whose width is in screen pixels, so overlays stay visible at any zoom."""
    pen = QPen(QColor(colour), width)
    pen.setCosmetic(True)
    return pen


class ImagePanel(QWidget):
    """Image viewer with zoom controls and overlay items.

    The zoom and pan survive frame changes: a new image of the same size
    keeps the current view; the view is refitted only for a new image size
    or after :meth:`reset_zoom`.  Until the user zooms, the image follows
    the panel size.

    Signals
    -------
    point_clicked(float, float)
        Emitted in *point-select* mode with scene (x, y) = (col, row).
    roi_selected(int, int, int, int)
        Emitted when rubber-band ROI selection finishes: (r0, r1, c0, c1) 1-indexed.
    circle_drawn(float, float, float, bool)
        Emitted in *circle* mode on release: scene (x, y) of the press (the
        centre), the radius to the release point and whether it was dragged
        (False: a click, radius 0).
    stroke_drawn(object)
        Emitted in *paint* mode on release: the stroke's scene (x, y) points.
    brush_scrolled(int)
        Mouse wheel in *paint* mode (+1 / -1 per notch; Ctrl+wheel zooms).
    """

    point_clicked = pyqtSignal(float, float)
    roi_selected = pyqtSignal(int, int, int, int)
    circle_drawn = pyqtSignal(float, float, float, bool)
    stroke_drawn = pyqtSignal(object)
    brush_scrolled = pyqtSignal(int)
    select_roi_clicked = pyqtSignal()

    def __init__(self, title: str = "Image", parent: QWidget | None = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addLayout(self._build_toolbar(title))

        self._scene = QGraphicsScene(self)
        self._view = _PanZoomView(self._scene, self)
        self._view.setStyleSheet(_VIEW_STYLE["normal"])
        self._view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._view.setMinimumSize(160, 120)
        layout.addWidget(self._view)

        self._pixmap_item: QGraphicsPixmapItem | None = None
        self._overlay_items: list = []
        self._mode = "normal"  # "normal" | "point" | "circle" | "paint" | "roi"
        self.circle_drag = CircleDrag(self._scene)
        self.brush = BrushStroke(self._scene)
        self._image_size: tuple[int, int] | None = None  # (h, w)
        self._auto_fit = True  # follow the panel size until the user zooms

    def _build_toolbar(self, title: str) -> QHBoxLayout:
        top = QHBoxLayout()
        top.setContentsMargins(8, 4, 8, 0)
        self._title_label = QLabel(title)
        self._title_label.setObjectName("sectionTitle")
        top.addWidget(self._title_label)
        top.addStretch()

        self._roi_label = QLabel("")
        self._roi_label.setObjectName("dimText")
        self._roi_label.setStyleSheet("font-size: 11px;")
        top.addWidget(self._roi_label)

        self._select_roi_btn = self._tool_button("ROI", 42, "Drag on image to select Region of Interest")
        self._select_roi_btn.clicked.connect(self.select_roi_clicked)
        self._zoom_in_btn = self._tool_button("+", 28, "Zoom in (wheel up)")
        self._zoom_in_btn.clicked.connect(self.zoom_in)
        self._zoom_out_btn = self._tool_button("−", 28, "Zoom out (wheel down)")
        self._zoom_out_btn.clicked.connect(self.zoom_out)
        self._zoom_reset_btn = self._tool_button("↺", 28, "Fit image to the panel (reset zoom)")
        self._zoom_reset_btn.clicked.connect(self.reset_zoom)
        for btn in (self._select_roi_btn, self._zoom_in_btn,
                    self._zoom_out_btn, self._zoom_reset_btn):
            top.addWidget(btn)
        return top

    @staticmethod
    def _tool_button(text: str, width: int, tip: str) -> QPushButton:
        btn = QPushButton(text)
        btn.setFixedSize(width, 28)
        btn.setObjectName("secondaryBtn")
        btn.setToolTip(tip)
        return btn

    # -- Public API --

    def set_title(self, title: str) -> None:
        self._title_label.setText(title)

    def set_image(self, img: np.ndarray):
        """Display a greyscale or binary image (float [0,1] or bool)."""
        if img.dtype == bool:
            img8 = (img.astype(np.uint8)) * 255
        elif img.dtype in (np.float32, np.float64):
            img8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        else:
            img8 = img.astype(np.uint8)
        img8 = np.ascontiguousarray(img8)
        h, w = img8.shape[:2]
        qimg = QImage(img8.tobytes(), w, h, w, QImage.Format.Format_Grayscale8)
        self._show_pixmap(QPixmap.fromImage(qimg), h, w)

    def set_image_rgb(self, img: np.ndarray):
        """Display an RGB image (H, W, 3) uint8.

        Used by overlay / wipe comparison modes that produce colour
        composites from the original grayscale and binary images.
        """
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError("set_image_rgb expects an (H, W, 3) array")
        img8 = np.ascontiguousarray(img.astype(np.uint8))
        h, w = img8.shape[:2]
        qimg = QImage(img8.tobytes(), w, h, 3 * w, QImage.Format.Format_RGB888)
        self._show_pixmap(QPixmap.fromImage(qimg), h, w)

    def _show_pixmap(self, pixmap: QPixmap, h: int, w: int) -> None:
        """Swap in a new frame; refit the view only when the image size changes."""
        self.clear_overlays()
        if self._pixmap_item is None:
            self._pixmap_item = self._scene.addPixmap(pixmap)
            self._pixmap_item.setZValue(-1)
        else:
            self._pixmap_item.setPixmap(pixmap)
        if self._image_size != (h, w):
            self._image_size = (h, w)
            self._scene.setSceneRect(QRectF(0, 0, w, h))
            self._auto_fit = True
            self._fit()

    def draw_circle(self, row_c: float, col_c: float, radius: float,
                    colour: str = "#6366f1", width: float = 2.0, *, dashed: bool = False):
        """Draw a circle overlay. Coordinates are (row, col) = (y, x)."""
        item = self._scene.addEllipse(
            col_c - radius, row_c - radius, 2 * radius, 2 * radius,
            dashed_pen(colour, width) if dashed else _cosmetic_pen(colour, width),
        )
        self._overlay_items.append(item)

    def draw_points(self, points: np.ndarray, colour: str = "#EF4444", size: float = 3.0):
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
        self._add_path(path, QPen(Qt.PenStyle.NoPen), QBrush(QColor(colour)))

    def draw_contour(self, points: np.ndarray, colour: str = "#ef4444",
                     width: float = 1.5, closed: bool = True) -> None:
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
        self._add_path(path, _cosmetic_pen(colour, width), QBrush(Qt.BrushStyle.NoBrush))

    def draw_fit_outline(self, points: np.ndarray | None, colour: str = "#ef4444") -> None:
        """Draw a fit's boundary: a closed polyline for an ordered contour,
        otherwise the individual edge points."""
        if points is None or len(points) == 0:
            return
        if is_ordered_contour(points):
            self.draw_contour(points, colour)
        else:
            self.draw_points(points, colour, 2.0)

    def _add_path(self, path: QPainterPath, pen: QPen, brush: QBrush) -> None:
        item = QGraphicsPathItem(path)
        item.setPen(pen)
        item.setBrush(brush)
        self._scene.addItem(item)
        self._overlay_items.append(item)

    def draw_rgba_overlay(self, rgba: np.ndarray) -> None:
        """Lay an (H, W, 4) uint8 image over the frame (scene pixel = array pixel)."""
        img = np.ascontiguousarray(rgba, dtype=np.uint8)
        h, w = img.shape[:2]
        qimg = QImage(img.tobytes(), w, h, 4 * w, QImage.Format.Format_RGBA8888)
        item = self._scene.addPixmap(QPixmap.fromImage(qimg))
        item.setZValue(1)
        self._overlay_items.append(item)

    def draw_roi_rect(self, gridx: tuple[int, int], gridy: tuple[int, int],
                      colour: str = "#10b981"):
        """Draw an ROI rectangle (1-indexed coords)."""
        rs, cs = roi_to_slice(gridx, gridy)
        r0, r1 = rs.start, rs.stop
        c0, c1 = cs.start, cs.stop
        item = self._scene.addRect(c0, r0, c1 - c0, r1 - r0, _cosmetic_pen(colour, 1.5))
        self._overlay_items.append(item)

    def clear_overlays(self):
        for item in self._overlay_items:
            self._scene.removeItem(item)
        self._overlay_items.clear()

    @property
    def mode(self) -> str:
        return self._mode

    def set_mode(self, mode: str):
        """Set interaction mode: 'normal', 'point', 'circle', 'paint' or 'roi'.

        | Mode   | Cursor       | Drag              | Border        |
        |--------|-------------|-------------------|---------------|
        | normal | default     | ScrollHandDrag    | subtle border |
        | point  | CrossCursor | NoDrag            | #6366f1 indigo|
        | circle | CrossCursor | draws a circle    | #6366f1 indigo|
        | paint  | CrossCursor | paints a stroke   | #22c55e green |
        | roi    | CrossCursor | RubberBandDrag    | #f59e0b amber |
        """
        self._mode = mode
        if mode != "circle":
            self.circle_drag.cancel()
        if mode != "paint":
            self.brush.cancel()
            self.brush.hide_hover()
        if mode == "roi":
            self._view.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
            self._view.setCursor(Qt.CursorShape.CrossCursor)
        elif mode in ("point", "circle", "paint"):
            self._view.setDragMode(QGraphicsView.DragMode.NoDrag)
            self._view.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self._view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self._view.unsetCursor()
        self._view.setStyleSheet(_VIEW_STYLE.get(mode, _VIEW_STYLE["normal"]))

    # -- ROI label --
    def set_roi_text(self, gridx: tuple[int, int], gridy: tuple[int, int]) -> None:
        """Update the ROI coordinate display in the toolbar."""
        self._roi_label.setText(
            f"X: {gridy[0]}–{gridy[1]}  Y: {gridx[0]}–{gridx[1]}"
        )

    # -- Zoom --
    @property
    def auto_fit(self) -> bool:
        """True while the image follows the panel size (no manual zoom)."""
        return self._auto_fit

    def zoom_in(self) -> None:
        self.zoom_by(1.25)

    def zoom_out(self) -> None:
        self.zoom_by(0.8)

    def zoom_by(self, factor: float) -> None:
        self._auto_fit = False
        self._view.scale(factor, factor)

    def reset_zoom(self) -> None:
        """Fit the whole image into the panel and follow its size again."""
        self._auto_fit = True
        self._fit()

    def _fit(self) -> None:
        if self._image_size is None:
            return
        self._view.resetTransform()
        self._view.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    # Backwards-compatible names (shortcut wiring in the controller)
    _zoom_in = zoom_in
    _zoom_out = zoom_out
    _zoom_reset = reset_zoom


class _PanZoomView(QGraphicsView):
    """QGraphicsView with wheel-zoom and pan."""

    def __init__(self, scene, panel: ImagePanel):
        super().__init__(scene)
        self._panel = panel
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setRenderHint(self.renderHints())

    def wheelEvent(self, event: QWheelEvent):
        up = event.angleDelta().y() > 0
        ctrl = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
        if self._panel._mode == "paint" and not ctrl:
            self._panel.brush_scrolled.emit(1 if up else -1)
            return
        self._panel.zoom_by(1.15 if up else 1 / 1.15)

    def leaveEvent(self, event) -> None:  # noqa: N802 (Qt override)
        self._panel.brush.hide_hover()
        super().leaveEvent(event)

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        if self._panel.auto_fit:
            self._panel._fit()

    def mousePressEvent(self, event: QMouseEvent):
        left = event.button() == Qt.MouseButton.LeftButton
        if self._panel._mode == "point" and left:
            pos = self.mapToScene(event.pos())
            self._panel.point_clicked.emit(pos.x(), pos.y())  # (col, row) in scene
            return
        if self._panel._mode == "circle" and left:
            self._panel.circle_drag.press(self.mapToScene(event.pos()), event.position())
            return
        if self._panel._mode == "paint" and left:
            self._panel.brush.press(self.mapToScene(event.pos()))
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._panel.circle_drag.active:
            self._panel.circle_drag.move(self.mapToScene(event.pos()))
            return
        if self._panel._mode == "paint":
            pos = self.mapToScene(event.pos())
            self._panel.brush.hover(pos)
            self._panel.brush.move(pos)
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        brush = self._panel.brush
        if brush.active and event.button() == Qt.MouseButton.LeftButton:
            points = brush.release(self.mapToScene(event.pos()))
            if points:
                self._panel.stroke_drawn.emit(points)
            return
        drag = self._panel.circle_drag
        if drag.active and event.button() == Qt.MouseButton.LeftButton:
            drawn = drag.release(self.mapToScene(event.pos()), event.position())
            if drawn is not None:
                self._panel.circle_drawn.emit(drawn.x, drawn.y, drawn.radius, drawn.dragged)
            return
        if self._panel._mode == "roi" and self.rubberBandRect():
            rect = self.mapToScene(self.rubberBandRect()).boundingRect()
            r0, r1 = int(rect.top()) + 1, int(rect.bottom()) + 1
            c0, c1 = int(rect.left()) + 1, int(rect.right()) + 1
            self._panel.roi_selected.emit(r0, r1, c0, c1)
            self._panel.set_mode("normal")
        super().mouseReleaseEvent(event)
