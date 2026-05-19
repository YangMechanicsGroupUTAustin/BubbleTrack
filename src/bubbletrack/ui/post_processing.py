"""Post-processing panel: export controls, FPS / scale inputs."""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QGridLayout, QHBoxLayout, QLabel,
    QPushButton, QSpinBox, QVBoxLayout, QWidget,
)

from bubbletrack.ui.widgets import CollapsibleSection


class PostProcessing(QWidget):
    """Collapsible export panel with FPS, scale, fit-length inputs."""

    export_r_data_clicked = pyqtSignal()
    export_rof_t_clicked = pyqtSignal()
    generate_report_clicked = pyqtSignal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        section = CollapsibleSection("Post Processing", collapsed=False)

        # -- Physical conversion parameters --
        grid = QGridLayout()
        grid.setSpacing(6)

        # FPS with unit selector
        grid.addWidget(QLabel("FPS:"), 0, 0)
        self._fps_spin = QDoubleSpinBox()
        self._fps_spin.setRange(0.001, 99999)
        self._fps_spin.setDecimals(1)
        self._fps_spin.setValue(10)
        grid.addWidget(self._fps_spin, 0, 1)
        self._fps_unit = QComboBox()
        self._fps_unit.addItems(["Hz", "k", "M"])
        self._fps_unit.setCurrentIndex(2)  # default M
        grid.addWidget(self._fps_unit, 0, 2)

        # um/px
        grid.addWidget(QLabel("\u00b5m/px:"), 1, 0)
        self._scale_spin = QDoubleSpinBox()
        self._scale_spin.setRange(0.001, 9999)
        self._scale_spin.setDecimals(3)
        self._scale_spin.setValue(3.2)
        grid.addWidget(self._scale_spin, 1, 1, 1, 2)

        # Rmax fit length
        grid.addWidget(QLabel("Rmax Fit:"), 2, 0)
        self._fit_len_spin = QSpinBox()
        self._fit_len_spin.setRange(3, 999)
        self._fit_len_spin.setSingleStep(2)
        self._fit_len_spin.setValue(11)
        grid.addWidget(self._fit_len_spin, 2, 1, 1, 2)

        # Export format selector
        grid.addWidget(QLabel("Format:"), 3, 0)
        self._format_combo = QComboBox()
        self._format_combo.addItems(["MAT", "CSV", "Excel"])
        self._format_combo.setCurrentIndex(0)
        grid.addWidget(self._format_combo, 3, 1, 1, 2)

        container = QWidget()
        container.setLayout(grid)
        section.add_widget(container)

        # -- Export buttons --
        btn_row = QHBoxLayout()
        self._export_r_btn = QPushButton("Export Pixel Data")
        self._export_r_btn.clicked.connect(self.export_r_data_clicked)
        btn_row.addWidget(self._export_r_btn)

        self._export_rt_btn = QPushButton("Export Physical Data")
        self._export_rt_btn.clicked.connect(self.export_rof_t_clicked)
        btn_row.addWidget(self._export_rt_btn)

        btn_container = QWidget()
        btn_container.setLayout(btn_row)
        section.add_widget(btn_container)

        # PDF report button
        self._report_btn = QPushButton("Generate Report")
        self._report_btn.clicked.connect(self.generate_report_clicked)
        section.add_widget(self._report_btn)

        # Status label
        self._status = QLabel("")
        self._status.setObjectName("dimText")
        section.add_widget(self._status)

        layout.addWidget(section)

    def get_fps(self) -> float:
        multipliers = {"Hz": 1, "k": 1_000, "M": 1_000_000}
        unit = self._fps_unit.currentText()
        return self._fps_spin.value() * multipliers.get(unit, 1)

    def get_scale(self) -> float:
        return self._scale_spin.value()

    def get_fit_length(self) -> int:
        return self._fit_len_spin.value()

    def get_format(self) -> str:
        """Return selected export format: ``"mat"``, ``"csv"``, or ``"xlsx"``."""
        mapping = {"MAT": "mat", "CSV": "csv", "Excel": "xlsx"}
        return mapping.get(self._format_combo.currentText(), "mat")

    def set_physical_params(self, fps: float, scale: float, fit_length: int) -> None:
        """Set persisted physical parameters without emitting signals."""
        if fps >= 1_000_000:
            self._fps_unit.setCurrentText("M")
            self._fps_spin.setValue(fps / 1_000_000)
        elif fps >= 1_000:
            self._fps_unit.setCurrentText("k")
            self._fps_spin.setValue(fps / 1_000)
        else:
            self._fps_unit.setCurrentText("Hz")
            self._fps_spin.setValue(fps)
        self._scale_spin.setValue(scale)
        self._fit_len_spin.setValue(fit_length)

    def set_status(self, text: str, success: bool = True):
        colour = "#10b981" if success else "#ef4444"
        self._status.setStyleSheet(f"color:{colour}; font-size:11px;")
        self._status.setText(text)
