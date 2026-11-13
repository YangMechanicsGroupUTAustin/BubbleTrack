"""Export dialog: what to export, the recording's physical scale, format and file.

The physical values (frame rate, um/px, Rmax fit length) belong to the
recording: edits apply at once (``physical_params_changed``) and are kept when
the dialog is closed without exporting, as the old side-panel fields were.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QButtonGroup, QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog,
    QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
    QPushButton, QRadioButton, QSpinBox, QVBoxLayout, QWidget,
)

UNITS = ("physical", "pixel")
FORMATS = (("MAT", "mat"), ("CSV", "csv"), ("Excel", "xlsx"))
FORMAT_EXT = {"mat": ".mat", "csv": ".csv", "xlsx": ".xlsx"}
FORMAT_FILTERS = {
    "mat": "MAT Files (*.mat);;All Files (*)",
    "csv": "CSV Files (*.csv);;All Files (*)",
    "xlsx": "Excel Files (*.xlsx);;All Files (*)",
}
_FPS_UNITS = (("Hz", 1.0), ("k", 1e3), ("M", 1e6))


def with_format_extension(path: str, fmt: str) -> str:
    """*path* ending in the extension of *fmt* (replacing another known one)."""
    if not path:
        return path
    ext = FORMAT_EXT[fmt]
    p = Path(path)
    if p.suffix.lower() == ext:
        return path
    if p.suffix.lower() in FORMAT_EXT.values():
        return str(p.with_suffix(ext))
    return path + ext


def report_path_for(path: str) -> str:
    """The PDF report saved next to an export: ``<stem>_report.pdf``."""
    p = Path(path)
    return str(p.with_name(f"{p.stem}_report.pdf"))


class ExportDialog(QDialog):
    """Choose units, scale, format and file; Export asks the controller to write it."""

    export_clicked = pyqtSignal()
    physical_params_changed = pyqtSignal()

    def __init__(self, *, fps: float, um2px: float, fit_length: int, fmt: str = "mat",
                 units: str = "physical", default_path: Callable[[str, str], str],
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export Results")
        self.setMinimumWidth(520)
        self._default_path = default_path
        self._path_edited = False  # the user typed or browsed a path: keep it
        self._browsed = ""  # confirmed in the save dialog (it asks about replacing)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.addWidget(self._build_units())
        layout.addWidget(self._build_recording())
        layout.addLayout(self._build_output())
        self._status = QLabel("")
        self._status.setObjectName("dimText")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)
        layout.addLayout(self._build_buttons())

        self.set_physical_params(fps, um2px, fit_length)
        self.set_units(units if units in UNITS else "physical")
        self.set_format(fmt)
        self._refresh_path()
        for control in (self._fps_spin, self._scale_spin, self._fit_len_spin):
            control.valueChanged.connect(self.physical_params_changed)
        self._fps_unit.currentIndexChanged.connect(self.physical_params_changed)

    # -- construction ------------------------------------------------------ #

    def _build_units(self) -> QGroupBox:
        box = QGroupBox("Data")
        rows = QVBoxLayout(box)
        self._physical = QRadioButton("Physical units: R(t) in µm, time from the frame rate")
        self._pixel = QRadioButton("Pixels: R per frame in px")
        self._units = QButtonGroup(self)
        for button in (self._physical, self._pixel):
            self._units.addButton(button)
            rows.addWidget(button)
        self._units.buttonToggled.connect(lambda _b, on: on and self._refresh_path())
        return box

    def _build_recording(self) -> QGroupBox:
        box = QGroupBox("Recording")
        grid = QGridLayout(box)
        grid.setSpacing(6)
        self._fps_spin = QDoubleSpinBox()
        self._fps_spin.setRange(0.001, 99999)
        self._fps_spin.setDecimals(3)  # e.g. 1.25 M must be enterable
        self._fps_unit = QComboBox()
        for label, _scale in _FPS_UNITS:
            self._fps_unit.addItem(label)
        grid.addWidget(QLabel("Frame rate:"), 0, 0)
        grid.addWidget(self._fps_spin, 0, 1)
        grid.addWidget(self._fps_unit, 0, 2)
        grid.addWidget(QLabel("fps"), 0, 3)
        self._scale_spin = QDoubleSpinBox()
        self._scale_spin.setRange(0.001, 9999)
        self._scale_spin.setDecimals(3)
        grid.addWidget(QLabel("Scale:"), 1, 0)
        grid.addWidget(self._scale_spin, 1, 1, 1, 2)
        grid.addWidget(QLabel("µm/px"), 1, 3)
        self._fit_len_spin = QSpinBox()
        self._fit_len_spin.setRange(3, 999)
        self._fit_len_spin.setSingleStep(2)
        self._fit_len_spin.setToolTip("Frames around the maximum used to fit Rmax and its time.")
        grid.addWidget(QLabel("Rmax fit:"), 2, 0)
        grid.addWidget(self._fit_len_spin, 2, 1, 1, 2)
        grid.addWidget(QLabel("frames"), 2, 3)
        grid.setColumnStretch(1, 1)
        return box

    def _build_output(self) -> QGridLayout:
        grid = QGridLayout()
        grid.setSpacing(6)
        self._format_combo = QComboBox()
        for label, fmt in FORMATS:
            self._format_combo.addItem(label, fmt)
        self._format_combo.currentIndexChanged.connect(lambda _i: self._refresh_path())
        grid.addWidget(QLabel("Format:"), 0, 0)
        grid.addWidget(self._format_combo, 0, 1)
        self._report = QCheckBox("Also save a PDF report (R(t) chart and fit summary)")
        grid.addWidget(self._report, 1, 1, 1, 2)
        self._path = QLineEdit()
        self._path.textEdited.connect(self._on_path_edited)
        browse = QPushButton("Browse…")
        browse.setObjectName("secondaryBtn")
        browse.clicked.connect(self._browse)
        grid.addWidget(QLabel("Save to:"), 2, 0)
        grid.addWidget(self._path, 2, 1)
        grid.addWidget(browse, 2, 2)
        grid.setColumnStretch(1, 1)
        return grid

    def _build_buttons(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.addStretch()
        self.export_button = QPushButton("Export")
        self.export_button.setDefault(True)
        self.export_button.clicked.connect(self.export_clicked)
        close = QPushButton("Close")
        close.setObjectName("secondaryBtn")
        close.clicked.connect(self.reject)
        row.addWidget(self.export_button)
        row.addWidget(close)
        return row

    # -- values ------------------------------------------------------------ #

    def get_units(self) -> str:
        return "pixel" if self._pixel.isChecked() else "physical"

    def set_units(self, units: str) -> None:
        (self._pixel if units == "pixel" else self._physical).setChecked(True)

    def get_format(self) -> str:
        return self._format_combo.currentData()

    def set_format(self, fmt: str) -> bool:
        index = self._format_combo.findData(fmt)
        if index >= 0:
            self._format_combo.setCurrentIndex(index)
        return index >= 0

    def get_fps(self) -> float:
        scale = dict(_FPS_UNITS)[self._fps_unit.currentText()]
        return self._fps_spin.value() * scale

    def get_scale(self) -> float:
        return self._scale_spin.value()

    def get_fit_length(self) -> int:
        return self._fit_len_spin.value()

    def set_physical_params(self, fps: float, scale: float, fit_length: int) -> None:
        """Show the values without emitting physical_params_changed."""
        controls = (self._fps_spin, self._fps_unit, self._scale_spin, self._fit_len_spin)
        for control in controls:
            control.blockSignals(True)
        try:
            label, factor = next(((lbl, f) for lbl, f in reversed(_FPS_UNITS) if fps >= f),
                                 _FPS_UNITS[0])
            self._fps_unit.setCurrentText(label)
            self._fps_spin.setValue(fps / factor)
            self._scale_spin.setValue(scale)
            self._fit_len_spin.setValue(int(fit_length))
        finally:
            for control in controls:
                control.blockSignals(False)

    def report_enabled(self) -> bool:
        return self._report.isChecked()

    def set_report_enabled(self, on: bool) -> None:
        self._report.setChecked(bool(on))

    def get_path(self) -> str:
        """The file to write, ending in the chosen format's extension."""
        return with_format_extension(self._path.text().strip(), self.get_format())

    def set_path(self, path: str) -> None:
        """Use *path* as typed by the user (no automatic renaming afterwards)."""
        self._path.setText(path)
        self._path_edited = True

    def set_status(self, text: str, success: bool = True, colour: str = "") -> None:
        colour = colour or ("#10b981" if success else "#ef4444")
        self._status.setStyleSheet(f"color:{colour}; font-size:11px;")
        self._status.setText(text)

    def status_text(self) -> str:
        return self._status.text()

    # -- the file ------------------------------------------------------------ #

    def _on_path_edited(self, _text: str) -> None:
        self._path_edited = True

    def _refresh_path(self) -> None:
        """A default name follows units and format; a chosen one keeps its name."""
        if self._path_edited and self._path.text().strip():
            self._path.setText(with_format_extension(self._path.text().strip(), self.get_format()))
            return
        self._path.setText(self._default_path(self.get_units(), self.get_format()))
        self._path_edited = False

    def _browse(self) -> None:
        fmt = self.get_format()
        path, _ = QFileDialog.getSaveFileName(self, "Export to", self.get_path(), FORMAT_FILTERS[fmt])
        if path:
            self.set_path(with_format_extension(path, fmt))
            self._browsed = self.get_path()

    def confirm_overwrite(self, paths: list[str]) -> bool:
        """Ask once before replacing existing files (the save dialog already asked)."""
        existing = [p for p in paths if p and Path(p).exists() and p != self._browsed]
        if not existing:
            return True
        names = "\n".join(Path(p).name for p in existing)
        answer = QMessageBox.question(
            self, "Replace existing file?", f"These files already exist:\n{names}\n\nReplace them?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        return answer == QMessageBox.StandardButton.Yes
