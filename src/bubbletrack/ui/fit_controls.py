"""The one Fit button: measures every frame; progress and Stop show while it runs."""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QHBoxLayout, QLabel, QProgressBar, QPushButton, QVBoxLayout, QWidget,
)

FIT_TIP = (
    "Measure every frame in the background.\n"
    "Locked frames are kept; you are asked about frames fixed by hand.\n"
    "To refit only some frames, select them on the R(t) chart and right-click."
)


class FitControls(QWidget):
    """Primary Fit button; a progress line with Stop appears only during a run."""

    fit_clicked = pyqtSignal()
    stop_clicked = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("fitControls")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.fit_button = QPushButton("Fit")
        self.fit_button.setObjectName("primaryBtn")
        self.fit_button.setToolTip(FIT_TIP)
        self.fit_button.clicked.connect(self.fit_clicked)
        layout.addWidget(self.fit_button)

        self.progress_row = QWidget()
        self.progress_row.setObjectName("fitProgress")
        rows = QVBoxLayout(self.progress_row)
        rows.setContentsMargins(0, 0, 0, 0)
        rows.setSpacing(4)
        self._progress_label = QLabel("")
        self._progress_label.setObjectName("dimText")
        self._progress_label.setWordWrap(True)
        rows.addWidget(self._progress_label)
        bar_row = QHBoxLayout()
        bar_row.setSpacing(8)
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setTextVisible(False)
        bar_row.addWidget(self._progress, 1)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setObjectName("dangerBtn")
        self.stop_button.setToolTip("Stop after the frames being measured; results so far are kept.")
        self.stop_button.clicked.connect(self.stop_clicked)
        bar_row.addWidget(self.stop_button)
        rows.addLayout(bar_row)
        layout.addWidget(self.progress_row)
        self.set_running(False)

    # -- run state ---------------------------------------------------------- #

    def set_running(self, running: bool) -> None:
        """While running: Fit disabled, progress line and Stop shown."""
        self.fit_button.setEnabled(not running)
        self.stop_button.setEnabled(running)
        self.progress_row.setVisible(running)
        if running:
            self._progress.setValue(0)
            self._progress_label.setText("Starting...")

    def is_running(self) -> bool:
        return not self.progress_row.isHidden()

    def set_progress(self, current: int, total: int, eta: str = "") -> None:
        pct = int(current / max(total, 1) * 100)
        self._progress.setValue(pct)
        text = f"Fitting frame {current}/{total} ({pct}%)"
        self._progress_label.setText(text + (f" — {eta}" if eta else ""))

    def set_progress_text(self, text: str) -> None:
        """Describe a run step that has no frame count (e.g. tracking)."""
        self._progress_label.setText(text)

    def progress_text(self) -> str:
        return self._progress_label.text()

    def reset_progress(self) -> None:
        self._progress.setValue(0)
        self._progress_label.setText("")
