"""Automatic mode tab: frame range, progress, Fit / Stop / Clear buttons."""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox, QHBoxLayout, QLabel, QProgressBar, QPushButton, QSpinBox,
    QVBoxLayout, QWidget,
)


class AutomaticTab(QWidget):
    """Automatic batch-processing panel."""

    fit_clicked = pyqtSignal()
    stop_clicked = pyqtSignal()
    clear_clicked = pyqtSignal()
    batch_folders_clicked = pyqtSignal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        # Frame range
        range_lbl = QLabel("Frame Range")
        range_lbl.setObjectName("sectionTitle")
        layout.addWidget(range_lbl)

        range_row = QHBoxLayout()
        range_row.addWidget(QLabel("From:"))
        self._start_spin = QSpinBox()
        self._start_spin.setMinimum(1)
        self._start_spin.setValue(1)
        self._start_spin.setMinimumWidth(70)
        range_row.addWidget(self._start_spin)

        range_row.addStretch()

        range_row.addWidget(QLabel("To:"))
        self._end_spin = QSpinBox()
        self._end_spin.setMinimum(1)
        self._end_spin.setValue(1)
        self._end_spin.setMinimumWidth(70)
        range_row.addWidget(self._end_spin)
        layout.addLayout(range_row)

        self._track_check = QCheckBox("Track the bubble across frames (Edge)")
        self._track_check.setChecked(True)
        self._track_check.setToolTip(
            "After Fit All, follow the bubble from frame to frame: frames whose\n"
            "result does not fit the track are re-measured near the predicted\n"
            "circle, and doubtful frames are added to the Review queue.\n"
            "Manual and locked frames are kept and anchor the track.")
        layout.addWidget(self._track_check)

        # Progress
        self._progress_label = QLabel("Ready")
        self._progress_label.setObjectName("dimText")
        layout.addWidget(self._progress_label)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setTextVisible(False)
        layout.addWidget(self._progress)

        # Buttons
        btn_row = QHBoxLayout()
        self._fit_btn = QPushButton("Fit All")
        self._fit_btn.clicked.connect(self.fit_clicked)
        btn_row.addWidget(self._fit_btn)

        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setObjectName("dangerBtn")
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self.stop_clicked)
        btn_row.addWidget(self._stop_btn)

        self._clear_btn = QPushButton("Clear")
        self._clear_btn.setObjectName("secondaryBtn")
        self._clear_btn.clicked.connect(self.clear_clicked)
        btn_row.addWidget(self._clear_btn)

        layout.addLayout(btn_row)

        # -- Batch multi-folder section --
        sep = QLabel("")
        sep.setFixedHeight(1)
        sep.setStyleSheet("background: rgba(255,255,255,0.06);")
        layout.addWidget(sep)

        batch_lbl = QLabel("Batch Multi-Folder")
        batch_lbl.setObjectName("sectionTitle")
        layout.addWidget(batch_lbl)

        self._batch_btn = QPushButton("Batch Folders...")
        self._batch_btn.setObjectName("secondaryBtn")
        self._batch_btn.clicked.connect(self.batch_folders_clicked)
        layout.addWidget(self._batch_btn)

        self._batch_status = QLabel("Select a parent directory containing experiment folders")
        self._batch_status.setObjectName("dimText")
        self._batch_status.setWordWrap(True)
        layout.addWidget(self._batch_status)

        self._batch_progress = QProgressBar()
        self._batch_progress.setRange(0, 100)
        self._batch_progress.setValue(0)
        self._batch_progress.setTextVisible(False)
        self._batch_progress.hide()
        layout.addWidget(self._batch_progress)

        layout.addStretch()

    # -- Public API --
    def set_range(self, total: int):
        """Reset the fit range to the full new source (1..total)."""
        self._start_spin.setMaximum(total)
        self._start_spin.setValue(1)
        self._end_spin.setMaximum(total)
        self._end_spin.setValue(total)

    def tracking_enabled(self) -> bool:
        return self._track_check.isChecked()

    def get_range(self) -> tuple[int, int]:
        """Return 1-indexed (start, end)."""
        return self._start_spin.value(), self._end_spin.value()

    def set_progress(self, current: int, total: int, eta: str = "") -> None:
        pct = int(current / max(total, 1) * 100)
        self._progress.setValue(pct)
        text = f"Processing: {current}/{total} ({pct}%)"
        if eta:
            text += f" — {eta}"
        self._progress_label.setText(text)

    def set_progress_text(self, text: str) -> None:
        """Describe a run step that has no frame count (e.g. tracking)."""
        self._progress_label.setText(text)

    def set_running(self, running: bool):
        self._fit_btn.setEnabled(not running)
        self._stop_btn.setEnabled(running)
        if not running:
            self._progress_label.setText("Done" if self._progress.value() > 0 else "Ready")

    def set_editing_enabled(self, enabled: bool) -> None:
        """Clear / Batch are disabled while a fit runs (Stop stays available)."""
        self._clear_btn.setEnabled(enabled)
        self._batch_btn.setEnabled(enabled)
        self._track_check.setEnabled(enabled)

    def reset_progress(self):
        self._progress.setValue(0)
        self._progress_label.setText("Ready")

    # -- Batch multi-folder API --

    def set_batch_running(self, running: bool) -> None:
        self._batch_btn.setEnabled(not running)
        self._batch_progress.setVisible(running)
        if not running:
            self._batch_progress.setValue(0)

    def set_batch_status(self, text: str) -> None:
        self._batch_status.setText(text)

    def set_batch_progress(self, current: int, total: int) -> None:
        pct = int(current / max(total, 1) * 100)
        self._batch_progress.setValue(pct)
