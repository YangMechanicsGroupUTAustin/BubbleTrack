"""Dark top navigation bar."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QWidget


class HeaderBar(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("HeaderBar")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 0, 16, 0)

        # App title
        title = QLabel("Bubble Radius Fitting")
        title.setStyleSheet("font-size:15px; font-weight:bold;")
        layout.addWidget(title)

        # Status indicator
        self._status_colour = "#10b981"
        self._status_dot = QLabel("\u25CF")
        self._status_dot.setStyleSheet("color:#10b981; font-size:10px;")
        layout.addWidget(self._status_dot)
        self._status_label = QLabel("Ready")
        self._status_label.setStyleSheet("color:#a1a1aa; font-size:12px; font-weight:normal;")
        layout.addWidget(self._status_label)

        layout.addStretch()

    def set_status(self, text: str, colour: str = "#10b981"):
        self._status_label.setText(text)
        self._status_colour = colour
        self._status_dot.setStyleSheet(f"color:{colour}; font-size:10px;")

    def status_text(self) -> str:
        return self._status_label.text()

    def status(self) -> tuple[str, str]:
        """(text, colour) on show, e.g. to put a run's summary back."""
        return self._status_label.text(), self._status_colour
