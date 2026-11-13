"""QGraphicsView-based image viewer with overlay support."""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QRectF, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QImage, QMouseEvent, QPainter, QPixmap, QResizeEvent, QWheelEvent
from PyQt6.QtWidgets import (
    QGraphicsPixmapItem, QGraphicsScene, QGraphicsView,
    QHBoxLayout, QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget,
)

from bubbletrack.ui.brush_stroke import BrushStroke
from bubbletrack.ui.circle_drag import CircleDrag
from bubbletrack.ui.contour_shape import is_ordered_contour  # noqa: F401 (public here too)
from bubbletrack.ui.image_overlays import OverlayDrawingMixin

_VIEW_STYLE = {
    "normal": "border:1px solid rgba(255,255,255,0.06); border-radius:8px; background:#0c0d12;",
    "point": "border:2px solid #6366f1; border-radius:8px; background:#0c0d12;",
    "circle": "border:2px solid #6366f1; border-radius:8px; background:#0c0d12;",
    "paint": "border:2px solid #22c55e; border-radius:8px; background:#0c0d12;",
    "roi": "border:2px solid #f59e0b; border-radius:8px; background:#0c0d12;",
}
PANEL_SIZE_HINT = (480, 360)  # px; above any toolbar's own width, so it never decides


class ImagePanel(OverlayDrawingMixin, QWidget):
    """Image viewer with zoom controls and overlay items.

    The zoom and pan survive frame changes: a new image of the same size
    keeps the current view; the view is refitted only for a new image size
    or after :meth:`reset_zoom`.  Until the user zooms, the image follows
    the panel size.  Overlays are drawn with the ``draw_*`` methods of
    :class:`~bubbletrack.ui.image_overlays.OverlayDrawingMixin`, which also
    keep the colour key in the top-left corner of the image up to date.

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
        self._init_overlays()
        self._mode = "normal"  # "normal" | "point" | "circle" | "paint" | "roi"
        self.circle_drag = CircleDrag(self._scene)
        self.brush = BrushStroke(self._scene)
        self._image_size: tuple[int, int] | None = None  # (h, w)
        self._auto_fit = True  # follow the panel size until the user zooms

    def _build_toolbar(self, title: str) -> QHBoxLayout:
        top = QHBoxLayout()
        top.setContentsMargins(8, 4, 8, 0)
        self._title_label = ElidedLabel(title)
        self._title_label.setObjectName("sectionTitle")
        top.addWidget(self._title_label)
        top.addStretch()

        self._roi_label = ElidedLabel("")  # like the title: never widens the panel
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

    def sizeHint(self) -> QSize:  # noqa: N802 (Qt override)
        """The same for every panel, whatever its title or ROI text: panels side
        by side split the row evenly."""
        least = self.minimumSizeHint()
        return QSize(max(PANEL_SIZE_HINT[0], least.width()),
                     max(PANEL_SIZE_HINT[1], least.height()))

    # -- Public API --

    def set_title(self, title: str, tooltip: str = "") -> None:
        """The panel's title (cut with "…" when narrow) and what it shows, on hover."""
        self._title_label.setText(title)
        self._title_label.setToolTip(tooltip)

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


class ElidedLabel(QLabel):
    """One-line label that ends in "…" when it is too narrow.

    ``text()`` stays whole; only the drawing is cut, and the label never asks
    for more than a few characters of width (its sizeHint is the whole text).
    """

    MIN_TEXT = "Image…"  # the narrowest it gets: about one short word

    def minimumSizeHint(self) -> QSize:  # noqa: N802 (Qt override)
        hint = super().minimumSizeHint()
        width = min(hint.width(), self.fontMetrics().horizontalAdvance(self.MIN_TEXT))
        return QSize(width, hint.height())

    def shown_text(self) -> str:
        """The text as drawn at the current width."""
        return self.fontMetrics().elidedText(self.text(), Qt.TextElideMode.ElideRight,
                                             self.contentsRect().width())

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        painter = QPainter(self)
        self.style().drawItemText(painter, self.contentsRect(),
                                  int(self.alignment() | Qt.AlignmentFlag.AlignVCenter),
                                  self.palette(), self.isEnabled(), self.shown_text(),
                                  self.foregroundRole())
        painter.end()


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
