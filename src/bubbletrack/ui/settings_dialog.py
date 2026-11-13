"""File > Settings: switches that bring back parts the clean main window hides."""

from __future__ import annotations

from collections.abc import Mapping

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QDialog, QDialogButtonBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget,
)

from bubbletrack.ui.widgets import ToggleSwitch

# (setting name in model.config.UI_SETTINGS, title, explanation)
SETTINGS: tuple[tuple[str, str, str], ...] = (
    ("show_classic_methods", "Classic detection methods",
     "Offer Intensity mask, Gradient rings and Dark region area next to Auto,\n"
     "with their controls (Threshold, Remove, Invert Mask, Auto Tune, ...).\n"
     "These methods will be phased out; Auto is recommended."),
    ("show_all_view_modes", "All image views",
     "Offer Image, Overlay and Wipe next to Image + Mask above the images."),
)


class SettingsDialog(QDialog):
    """One switch per setting; every change is applied at once (no restart)."""

    setting_changed = pyqtSignal(str, bool)

    def __init__(self, values: Mapping[str, bool], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(460)
        layout = QVBoxLayout(self)
        layout.setSpacing(18)
        self._switches: dict[str, ToggleSwitch] = {}
        for name, title, text in SETTINGS:
            layout.addWidget(self._setting_row(name, title, text, bool(values.get(name, False))))
        layout.addStretch()

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _setting_row(self, name: str, title: str, text: str, on: bool) -> QWidget:
        """Title and switch on one line, the explanation below."""
        row = QWidget()
        rows = QVBoxLayout(row)
        rows.setContentsMargins(0, 0, 0, 0)
        rows.setSpacing(4)
        head = QHBoxLayout()
        head.addWidget(QLabel(f"<b>{title}</b>"))
        head.addStretch()
        switch = ToggleSwitch()
        switch.setChecked(on)
        switch.setAccessibleName(title)
        switch.toggled.connect(lambda checked, key=name: self.setting_changed.emit(key, bool(checked)))
        head.addWidget(switch)
        rows.addLayout(head)
        detail = QLabel(text)
        detail.setObjectName("dimText")
        detail.setWordWrap(True)
        rows.addWidget(detail)
        self._switches[name] = switch
        return row

    def switch(self, name: str) -> ToggleSwitch:
        return self._switches[name]

    def value(self, name: str) -> bool:
        return self._switches[name].isChecked()

    def set_value(self, name: str, value: bool) -> None:
        """Show *value* without emitting setting_changed."""
        switch = self._switches[name]
        switch.blockSignals(True)
        switch.setChecked(bool(value))
        switch.blockSignals(False)
