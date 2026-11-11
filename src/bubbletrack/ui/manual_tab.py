"""Manual mode tab: point selection, counter, progress, instructions."""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QHBoxLayout, QLabel, QProgressBar, QPushButton, QSpinBox,
    QVBoxLayout, QWidget,
)

from bubbletrack.ui.widgets import InfoBox


class ManualTab(QWidget):
    """Manual point-selection mode panel."""

    select_clicked = pyqtSignal()
    done_clicked = pyqtSignal()
    clear_clicked = pyqtSignal()
    frame_selected = pyqtSignal(int)  # 0-indexed frame number

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        # -- Frame selector --
        frame_row = QHBoxLayout()
        frame_row.addWidget(QLabel("Frame:"))
        self._frame_spin = QSpinBox()
        self._frame_spin.setMinimum(1)
        self._frame_spin.setValue(1)
        self._frame_spin.setMinimumWidth(70)
        self._frame_spin.valueChanged.connect(
            lambda v: self.frame_selected.emit(v - 1)
        )
        frame_row.addWidget(self._frame_spin)
        self._frame_total_label = QLabel("/ —")
        frame_row.addWidget(self._frame_total_label)
        frame_row.addStretch()
        layout.addLayout(frame_row)

        # Instructions
        self._info = InfoBox(
            "Click on the bubble edge to select points.\n"
            "At least 3 points are needed for circle fitting.\n"
            "Press Done or Enter when finished.",
            colour="rgba(99, 102, 241, 0.15)", border="rgba(99, 102, 241, 0.20)",
        )
        layout.addWidget(self._info)

        # Point counter with progress
        counter_row = QHBoxLayout()
        self._count_label = QLabel("Points: 0")
        self._count_label.setStyleSheet("font-weight:bold; font-size:14px;")
        counter_row.addWidget(self._count_label)
        counter_row.addStretch()
        self._status = QLabel("Need \u2265 3")
        self._status.setObjectName("dimText")
        counter_row.addWidget(self._status)
        layout.addLayout(counter_row)

        self._progress = QProgressBar()
        self._progress.setRange(0, 3)
        self._progress.setValue(0)
        self._progress.setTextVisible(False)
        layout.addWidget(self._progress)

        # Buttons
        btn_row = QHBoxLayout()
        self._select_btn = QPushButton("Select Points")
        self._select_btn.clicked.connect(self.select_clicked)
        btn_row.addWidget(self._select_btn)

        self._done_btn = QPushButton("Done")
        self._done_btn.setEnabled(False)
        self._done_btn.clicked.connect(self.done_clicked)
        btn_row.addWidget(self._done_btn)

        self._clear_btn = QPushButton("Clear")
        self._clear_btn.setObjectName("secondaryBtn")
        self._clear_btn.clicked.connect(self.clear_clicked)
        btn_row.addWidget(self._clear_btn)

        layout.addLayout(btn_row)
        layout.addStretch()

    def set_point_count(self, n: int):
        self._count_label.setText(f"Points: {n}")
        self._progress.setValue(min(n, 3))
        self._done_btn.setEnabled(n >= 3)
        if n >= 3:
            self._status.setText("Ready to fit")
            self._status.setStyleSheet("color:#10b981; font-size:11px;")
        else:
            self._status.setText(f"Need {3 - n} more")
            self._status.setStyleSheet("color:#71717a; font-size:11px;")

    def reset(self):
        self.set_point_count(0)

    def set_frame_range(self, total: int):
        """Set the max frame number without emitting a frame change.

        Shrinking the maximum clamps the value, which would otherwise emit
        frame_selected for a frame of the previous source.
        """
        self._frame_spin.blockSignals(True)
        self._frame_spin.setMaximum(total)
        self._frame_spin.blockSignals(False)
        self._frame_total_label.setText(f"/ {total}")

    def set_frame_value(self, idx: int):
        """Set frame spinbox to 1-indexed value without emitting signal."""
        self._frame_spin.blockSignals(True)
        self._frame_spin.setValue(idx + 1)
        self._frame_spin.blockSignals(False)
