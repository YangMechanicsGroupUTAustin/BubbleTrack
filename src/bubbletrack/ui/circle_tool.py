"""Manual tab page of the "Drag a circle" tool: how to use it, and the
non-blocking prompt shown when no wall matching the circle was found."""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from bubbletrack.ui.widgets import InfoBox

CIRCLE_HELP = (
    "Press on the middle of the bubble, drag to its wall and release.\n"
    "Half to twice the size is fine: BubbleTrack measures the exact wall itself.\n"
    "A click (no drag) uses the size of the nearest fitted frame.\n"
    "Ctrl+Z undoes the last fix."
)
USE_CIRCLE_TEXT = "Use my circle as a manual fit"


class CircleToolPage(QWidget):
    """Instructions plus a hidden prompt: 'Use my circle as a manual fit' / 'Cancel'."""

    use_circle_clicked = pyqtSignal()
    cancel_clicked = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(InfoBox(
            CIRCLE_HELP,
            colour="rgba(99, 102, 241, 0.15)", border="rgba(99, 102, 241, 0.20)",
        ))
        layout.addWidget(self._build_prompt())
        self.hide_prompt()

    def _build_prompt(self) -> QFrame:
        self._prompt = QFrame()
        self._prompt.setObjectName("circlePrompt")
        self._prompt.setStyleSheet(
            "QFrame#circlePrompt { background: rgba(245, 158, 11, 0.12);"
            " border: 1px solid rgba(245, 158, 11, 0.35); border-radius: 8px; }"
        )
        box = QVBoxLayout(self._prompt)
        box.setContentsMargins(10, 8, 10, 8)
        box.setSpacing(6)
        self._prompt_text = QLabel()
        self._prompt_text.setWordWrap(True)
        self._prompt_text.setStyleSheet("color:#fcd34d; border:none;")
        box.addWidget(self._prompt_text)
        row = QHBoxLayout()
        self._use_btn = QPushButton(USE_CIRCLE_TEXT)
        self._use_btn.setToolTip("Store the circle you drew as this frame's result (source MANUAL)")
        self._use_btn.clicked.connect(self.use_circle_clicked)
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setObjectName("secondaryBtn")
        self._cancel_btn.setToolTip("Leave the frame as it was")
        self._cancel_btn.clicked.connect(self.cancel_clicked)
        row.addWidget(self._use_btn, 1)
        row.addWidget(self._cancel_btn)
        box.addLayout(row)
        return self._prompt

    def show_prompt(self, text: str, offer_manual: bool = True) -> None:
        self._prompt_text.setText(text)
        self._use_btn.setVisible(offer_manual)
        self._prompt.show()

    def hide_prompt(self) -> None:
        self._prompt.hide()

    def prompt_visible(self) -> bool:
        return not self._prompt.isHidden()

    def prompt_text(self) -> str:
        return self._prompt_text.text()

    def offers_manual(self) -> bool:
        return not self._use_btn.isHidden()
