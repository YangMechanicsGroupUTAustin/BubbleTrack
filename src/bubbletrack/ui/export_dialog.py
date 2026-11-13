"""Export dialog: what to export, the recording's physical scale, format and file.

The physical values (frame rate, um/px, Rmax fit length) belong to the
recording: edits apply at once (``physical_params_changed``) and are kept when
the dialog is closed without exporting, as the old side-panel fields were.

Frame rate and scale that were not set for this recording (*unset*) are
marked "Not set for this recording — check" and physical export waits until
they are: typing a value sets it, "These are right" keeps the values shown.
Pixels need neither.
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
PHYSICAL_FIELDS = frozenset({"fps", "scale"})
NOT_SET_TEXT = "⚠ Not set for this recording — check"
BLOCKED_TEXT = ("Check the frame rate and scale of this recording first "
                "(or choose Pixels).")
RMAX_FIT_TEXT = "frames around the largest radius, used to find Rmax and its time (t = 0)"
_MARK_STYLE = "border: 1px solid #f59e0b;"
_WARN = "#f59e0b"


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
                 unset: frozenset[str] = frozenset(), parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export Results")
        self.setMinimumWidth(520)
        self._default_path = default_path
        self._unset = frozenset(unset) & PHYSICAL_FIELDS
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
        self._fps_spin.valueChanged.connect(lambda _v: self._mark_set({"fps"}))
        self._fps_unit.currentIndexChanged.connect(lambda _i: self._mark_set({"fps"}))
        self._scale_spin.valueChanged.connect(lambda _v: self._mark_set({"scale"}))
        self._fit_len_spin.valueChanged.connect(self.physical_params_changed)
        self._show_unset()

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
        self._units.buttonToggled.connect(lambda _b, on: on and self._on_units_changed())
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
        grid.addWidget(self._build_unset_line(), 2, 1, 1, 3)
        self._fit_len_spin = QSpinBox()
        self._fit_len_spin.setRange(3, 999)
        self._fit_len_spin.setSingleStep(2)
        self._fit_len_spin.setToolTip("Frames around the largest radius used to fit Rmax\n"
                                      "and its time, the zero of the exported time axis.")
        grid.addWidget(QLabel("Rmax fit:"), 3, 0)
        grid.addWidget(self._fit_len_spin, 3, 1, 1, 2)
        grid.addWidget(QLabel("frames"), 3, 3)
        rmax_note = QLabel(RMAX_FIT_TEXT)
        rmax_note.setObjectName("dimText")
        rmax_note.setWordWrap(True)
        grid.addWidget(rmax_note, 4, 1, 1, 3)
        grid.setColumnStretch(1, 1)
        return box

    def _build_unset_line(self) -> QWidget:
        """'⚠ Not set for this recording — check  [These are right]'."""
        self._unset_line = QWidget()
        row = QHBoxLayout(self._unset_line)
        row.setContentsMargins(0, 0, 0, 0)
        self._unset_label = QLabel(NOT_SET_TEXT)
        self._unset_label.setStyleSheet(f"color:{_WARN};")
        self._unset_label.setWordWrap(True)
        self._confirm = QPushButton("These are right")
        self._confirm.setObjectName("secondaryBtn")
        self._confirm.setToolTip("Keep the frame rate and scale shown for this recording")
        self._confirm.clicked.connect(self.confirm_values)
        row.addWidget(self._unset_label, 1)
        row.addWidget(self._confirm)
        return self._unset_line

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

    # -- frame rate and scale set for this recording? ------------------------ #

    def unset_fields(self) -> frozenset[str]:
        """'fps' / 'scale' not yet set for this recording."""
        return self._unset

    def confirm_values(self) -> None:
        """'These are right': the values shown are this recording's."""
        self._mark_set(PHYSICAL_FIELDS)

    def physical_ready(self) -> bool:
        return not self._unset

    def _mark_set(self, fields: set[str] | frozenset[str]) -> None:
        self._unset = self._unset - frozenset(fields)
        self._show_unset()
        self.physical_params_changed.emit()

    def _show_unset(self) -> None:
        """Mark the unset fields; physical export waits for them."""
        for field_name, spin in (("fps", self._fps_spin), ("scale", self._scale_spin)):
            spin.setStyleSheet(_MARK_STYLE if field_name in self._unset else "")
        self._unset_line.setVisible(bool(self._unset))
        self._refresh_export_button()

    def _on_units_changed(self) -> None:
        self._refresh_path()
        self._refresh_export_button()

    def _refresh_export_button(self) -> None:
        blocked = self.get_units() == "physical" and bool(self._unset)
        self.export_button.setEnabled(not blocked)
        if blocked:
            self.set_status(BLOCKED_TEXT, colour=_WARN)
        elif self.status_text() == BLOCKED_TEXT:
            self.set_status("")

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
