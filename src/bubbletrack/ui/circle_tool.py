"""The two fix tools' names and texts, and the non-blocking prompt shown when
Auto found no wall matching a drawn circle ("Keep my circle" / "Cancel")."""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

TOOL_CIRCLE, TOOL_POINTS = "circle", "points"

CIRCLE_HINT = "Drag from the bubble's centre to its wall; Auto finds the exact wall."
CIRCLE_TIP = (
    "Half to twice the size is fine: BubbleTrack measures the exact wall itself.\n"
    "A click (no drag) uses the size of the nearest fitted frame.\n"
    "Ctrl+Z undoes the last fix."
)
POINTS_TIP = "Click at least 3 points on the bubble wall, then Done (Enter). Ctrl+Z undoes a point."
CIRCLE_UNAVAILABLE_TIP = (
    "Drag a circle uses the Auto (reference frame) engine.\n"
    "Choose \"Auto (reference frame)\" as the detection mode to use it."
)
USE_CIRCLE_TEXT = "Keep my circle"
USE_CIRCLE_TIP = ("Store the circle you drew as this frame's result\n"
                  "(a manual fit, locked; Ctrl+Z undoes)")


class CirclePrompt(QFrame):
    """Hidden until a drawn circle is refused: what happened and why, then
    'Keep my circle' / 'Cancel'."""

    use_circle_clicked = pyqtSignal()
    cancel_clicked = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("circlePrompt")
        self.setStyleSheet(
            "QFrame#circlePrompt { background: rgba(245, 158, 11, 0.12);"
            " border: 1px solid rgba(245, 158, 11, 0.35); border-radius: 8px; }"
        )
        box = QVBoxLayout(self)
        box.setContentsMargins(10, 8, 10, 8)
        box.setSpacing(6)
        self._text = QLabel()
        self._text.setWordWrap(True)
        self._text.setStyleSheet("color:#fcd34d; border:none; background:transparent;")
        box.addWidget(self._text)
        row = QHBoxLayout()
        self._use_btn = QPushButton(USE_CIRCLE_TEXT)
        self._use_btn.setToolTip(USE_CIRCLE_TIP)
        self._use_btn.clicked.connect(self.use_circle_clicked)
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setObjectName("secondaryBtn")
        self._cancel_btn.setToolTip("Leave the frame as it was (Esc)")
        self._cancel_btn.clicked.connect(self.cancel_clicked)
        row.addWidget(self._use_btn, 1)
        row.addWidget(self._cancel_btn)
        box.addLayout(row)
        self.hide_prompt()

    def show_prompt(self, text: str, offer_manual: bool = True) -> None:
        self._text.setText(text)
        self._use_btn.setVisible(offer_manual)
        self.show()

    def hide_prompt(self) -> None:
        self.hide()

    def prompt_visible(self) -> bool:
        return not self.isHidden()

    def prompt_text(self) -> str:
        return self._text.text()

    def offers_manual(self) -> bool:
        return not self._use_btn.isHidden()
