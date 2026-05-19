"""Interactive R-t scatter chart using pyqtgraph."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QAction, QFont
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QMenu, QPushButton, QVBoxLayout, QWidget

from bubbletrack.model.constants import QUALITY_GOOD_THRESHOLD, QUALITY_WARN_THRESHOLD


class RadiusChart(QWidget):
    """Scatter plot of bubble radius vs frame number.

    Uses pyqtgraph for 10-100x faster updates compared to Matplotlib.
    Supports click-to-jump, zoom, pan, and right-click context menu.

    Signals:
        point_clicked(int): Emitted when user clicks a data point,
                           with the 0-based frame index.
        delete_requested(int): Emitted when user requests deletion of a point,
                              with the 0-based frame index.
        refit_requested(int): Emitted when user requests refitting a frame,
                             with the 0-based frame index.
        neighbor_refit_requested(int): Emitted when user requests refitting a
                                      frame using adjacent fitted frames.
    """

    point_clicked = pyqtSignal(int)
    delete_requested = pyqtSignal(int)
    refit_requested = pyqtSignal(int)
    neighbor_refit_requested = pyqtSignal(int)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setMinimumHeight(350)
        self.setMaximumHeight(500)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # Toolbar row with reset-zoom button
        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(4, 0, 4, 0)
        toolbar.addStretch()
        self._reset_btn = QPushButton("Reset Zoom")
        self._reset_btn.setObjectName("secondaryBtn")
        self._reset_btn.setFixedHeight(28)
        self._reset_btn.clicked.connect(self._reset_zoom)
        toolbar.addWidget(self._reset_btn)
        layout.addLayout(toolbar)

        # Configure pyqtgraph plot widget
        self._plot = pg.PlotWidget()
        self._plot.setBackground("#0c0d12")
        self._plot.setLabel("left", "Radius (px)")
        self._plot.setLabel("bottom", "Frame")
        self._plot.showGrid(x=True, y=True, alpha=0.3)

        # Style axes to match the app theme
        for axis_name in ("left", "bottom"):
            axis = self._plot.getAxis(axis_name)
            axis.setPen(pg.mkPen(255, 255, 255, 15))  # rgba(255,255,255,0.06)
            axis.setTextPen(pg.mkPen("#71717a"))

        # Scatter plot item — "+" markers (color-coded by quality when available)
        self._scatter = pg.ScatterPlotItem(
            pen=pg.mkPen("#EF4444", width=1.5),
            brush=pg.mkBrush("#EF4444"),
            symbol="+",
            size=10,
        )
        self._plot.addItem(self._scatter)

        # Click highlight — larger indigo circle on selected point (Issue 4c)
        self._highlight = pg.ScatterPlotItem(
            pen=pg.mkPen("#6366f1", width=2),
            brush=pg.mkBrush(99, 102, 241, 50),
            symbol="o",
            size=18,
        )
        self._plot.addItem(self._highlight)

        # Hover tooltip label (Issue 4b)
        self._hover_label = pg.TextItem(color="#e4e4e7", anchor=(0, 1))
        self._hover_label.setFont(QFont("Segoe UI", 9))
        self._hover_label.hide()
        self._plot.addItem(self._hover_label)

        # Hover tracking (Issue 4b)
        self._plot.scene().sigMouseMoved.connect(self._on_mouse_moved)

        layout.addWidget(self._plot)

        # Legend row — fit quality color key
        legend_row = QHBoxLayout()
        legend_row.setContentsMargins(8, 0, 8, 0)
        lbl_good = QLabel(
            '<span style="color:#10b981; font-size:14px; font-weight:bold;">+</span>'
            '  <span style="color:#a1a1aa; font-size:11px;">Good fit</span>'
        )
        lbl_fair = QLabel(
            '<span style="color:#f59e0b; font-size:14px; font-weight:bold;">+</span>'
            '  <span style="color:#a1a1aa; font-size:11px;">Fair fit</span>'
        )
        lbl_poor = QLabel(
            '<span style="color:#ef4444; font-size:14px; font-weight:bold;">+</span>'
            '  <span style="color:#a1a1aa; font-size:11px;">Poor fit</span>'
        )
        for lbl in (lbl_good, lbl_fair, lbl_poor):
            lbl.setStyleSheet("background: transparent; border: none;")
        legend_row.addWidget(lbl_good)
        legend_row.addWidget(lbl_fair)
        legend_row.addWidget(lbl_poor)
        legend_row.addStretch()
        layout.addLayout(legend_row)

        self._total_frames: int = 0
        self._frames: np.ndarray = np.array([])
        self._radii: np.ndarray = np.array([])
        self._quality_scores: np.ndarray | None = None

        # Right-click context menu on scatter plot
        self._setup_context_menu()

    def plot_all(
        self,
        frames: np.ndarray,
        radii: np.ndarray,
        quality_scores: np.ndarray | None = None,
    ) -> None:
        """Clear and replot all valid data points (radius > 0).

        When *quality_scores* is provided, points are color-coded by fit
        quality: green (good), amber (fair), red (poor).
        """
        self._highlight.setData([], [])  # clear stale highlight
        mask = radii > 0
        if not np.any(mask):
            self._scatter.setData([], [])
            self._frames = np.array([])
            self._radii = np.array([])
            self._quality_scores = None
            return
        self._frames = frames[mask]
        self._radii = radii[mask]

        if quality_scores is not None:
            filtered = quality_scores[mask]
            self._quality_scores = filtered
            colors = [self._quality_color(s) for s in filtered]
            brushes = [pg.mkBrush(c) for c in colors]
            pens = [pg.mkPen(c, width=1.5) for c in colors]
            self._scatter.setData(
                self._frames, self._radii, brush=brushes, pen=pens,
            )
        else:
            self._quality_scores = None
            self._scatter.setData(
                self._frames, self._radii,
                pen=pg.mkPen("#EF4444", width=1.5),
                brush=pg.mkBrush("#EF4444"),
            )

    @staticmethod
    def _quality_color(score: float) -> str:
        """Map a quality score [0, 1] to a display color."""
        if score >= QUALITY_GOOD_THRESHOLD:
            return "#10b981"  # green — reliable
        if score >= QUALITY_WARN_THRESHOLD:
            return "#f59e0b"  # amber — marginal
        return "#ef4444"      # red — unreliable

    def set_total_frames(self, n: int) -> None:
        """Fix the x-axis range to [1, n]."""
        self._total_frames = n
        if n > 0:
            self._plot.setXRange(0.5, n + 0.5, padding=0)

    def clear(self) -> None:
        """Remove all data points from the chart."""
        self._scatter.setData([], [])
        self._highlight.setData([], [])
        self._hover_label.hide()
        self._frames = np.array([])
        self._radii = np.array([])
        self._quality_scores = None

    def _setup_context_menu(self) -> None:
        """Connect right-click detection on the scatter plot scene."""
        self._plot.scene().sigMouseClicked.connect(self._on_mouse_clicked)

    def _find_nearest_point(self, x: float, y: float) -> int | None:
        """Find nearest data point to (x, y) in view coords. Return 0-based idx or None."""
        if len(self._frames) == 0:
            return None

        vb = self._plot.plotItem.vb
        view_range = vb.viewRange()
        x_range = view_range[0][1] - view_range[0][0]
        y_range = view_range[1][1] - view_range[1][0]

        if x_range <= 0 or y_range <= 0:
            return None

        # Normalize distances by axis range for uniform click tolerance
        dx = (self._frames - x) / x_range
        dy = (self._radii - y) / y_range
        dists = dx ** 2 + dy ** 2

        nearest_idx = int(np.argmin(dists))
        # Threshold: 3% of the plot area (generous click tolerance)
        if dists[nearest_idx] > 0.03 ** 2 + 0.03 ** 2:
            return None

        # Convert from filtered index to 0-based frame index
        frame_num = int(self._frames[nearest_idx])  # 1-based
        return frame_num - 1  # 0-based

    def _on_mouse_clicked(self, event) -> None:
        """Unified click handler: double-click resets zoom, left-click jumps, right-click menu."""
        if event.double():
            self._reset_zoom()
            return

        pos = self._plot.plotItem.vb.mapSceneToView(event.scenePos())
        frame_idx = self._find_nearest_point(pos.x(), pos.y())

        if event.button() == Qt.MouseButton.LeftButton:
            if frame_idx is not None:
                self.highlight_frame(frame_idx + 1)  # 1-based
                self.point_clicked.emit(frame_idx)
            return

        if event.button() == Qt.MouseButton.RightButton:
            if frame_idx is None:
                return
            menu = QMenu(self)
            frame_label = frame_idx + 1

            delete_action = QAction(f"Delete point (Frame {frame_label})", self)
            delete_action.triggered.connect(
                lambda _checked=False, i=frame_idx: self.delete_requested.emit(i),
            )
            refit_action = QAction(f"Refit frame {frame_label}", self)
            refit_action.triggered.connect(
                lambda _checked=False, i=frame_idx: self.refit_requested.emit(i),
            )
            neighbor_refit_action = QAction(
                f"Refit using neighbor frames (Frame {frame_label})",
                self,
            )
            neighbor_refit_action.triggered.connect(
                lambda _checked=False, i=frame_idx: (
                    self.neighbor_refit_requested.emit(i)
                ),
            )
            menu.addAction(delete_action)
            menu.addAction(refit_action)
            menu.addAction(neighbor_refit_action)
            menu.exec(event.screenPos().toPoint())

    # ------------------------------------------------------------------ #
    #  Reset zoom (Issue 4a)
    # ------------------------------------------------------------------ #

    def _reset_zoom(self) -> None:
        """Reset axes to original auto-range (or fixed x-range if set)."""
        self._plot.enableAutoRange()
        if self._total_frames > 0:
            self._plot.setXRange(0.5, self._total_frames + 0.5, padding=0)

    # ------------------------------------------------------------------ #
    #  Hover tooltip (Issue 4b)
    # ------------------------------------------------------------------ #

    def _on_mouse_moved(self, pos) -> None:
        """Show tooltip near the nearest data point on hover."""
        if len(self._frames) == 0:
            self._hover_label.hide()
            return

        mouse_point = self._plot.plotItem.vb.mapSceneToView(pos)
        x, y = mouse_point.x(), mouse_point.y()

        if not (np.isfinite(x) and np.isfinite(y)):
            self._hover_label.hide()
            return

        vb = self._plot.plotItem.vb
        view_range = vb.viewRange()
        x_range = view_range[0][1] - view_range[0][0]
        y_range = view_range[1][1] - view_range[1][0]

        if x_range <= 0 or y_range <= 0:
            self._hover_label.hide()
            return

        dx = (self._frames - x) / x_range
        dy = (self._radii - y) / y_range
        dists = dx ** 2 + dy ** 2

        nearest_idx = int(np.argmin(dists))
        # 2% threshold (relaxed from _find_nearest_point's 1%)
        if dists[nearest_idx] > 0.02 ** 2 + 0.02 ** 2:
            self._hover_label.hide()
            return

        frame = int(self._frames[nearest_idx])
        radius = self._radii[nearest_idx]
        if self._quality_scores is not None and nearest_idx < len(self._quality_scores):
            q_pct = int(self._quality_scores[nearest_idx] * 100)
            self._hover_label.setText(
                f"Frame: {frame}  R: {radius:.1f} px  Q: {q_pct}%",
            )
        else:
            self._hover_label.setText(f"Frame: {frame}  R: {radius:.1f} px")
        self._hover_label.setPos(self._frames[nearest_idx], self._radii[nearest_idx])
        self._hover_label.show()

    # ------------------------------------------------------------------ #
    #  Public highlight API (Issue 4c)
    # ------------------------------------------------------------------ #

    def highlight_frame(self, frame_1based: int) -> None:
        """Highlight a specific frame on the chart (for external sync)."""
        if len(self._frames) == 0:
            return
        mask = self._frames == frame_1based
        if np.any(mask):
            idx = int(np.argmax(mask))
            self._highlight.setData([self._frames[idx]], [self._radii[idx]])
        else:
            self._highlight.setData([], [])
