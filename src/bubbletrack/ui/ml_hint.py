"""The short, non-blocking hint 'many frames failed: try ML assist' (Review dock).

The text says what failed in a few words; the button says what to do.
"""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QToolButton, QWidget


class MLHint(QFrame):
    """One line of text, an 'ML assist…' button and a close cross."""

    open_clicked = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("mlHint")
        self.setStyleSheet("QFrame#mlHint { background: rgba(245, 158, 11, 0.12);"
                           " border: 1px solid rgba(245, 158, 11, 0.35); border-radius: 6px; }")
        row = QHBoxLayout(self)
        row.setContentsMargins(8, 4, 4, 4)
        self._text = QLabel()
        self._text.setWordWrap(True)
        self._text.setStyleSheet("color:#fcd34d; border:none;")
        self._text.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self._open = QPushButton("Try ML assist…")
        self._open.setObjectName("secondaryBtn")
        self._open.setToolTip("ML assist can help on hard recordings: confirm a few frames\n"
                              "or paint a few strokes to teach it, then apply it")
        self._open.clicked.connect(self.open_clicked)
        close = QToolButton()
        close.setText("×")
        close.setToolTip("Hide this hint")
        close.setAutoRaise(True)
        close.clicked.connect(self.hide)
        row.addWidget(self._text, 1)
        row.addWidget(self._open)
        row.addWidget(close)
        self.hide()

    def show_text(self, text: str) -> None:
        self._text.setText(text)
        self.show()

    def text(self) -> str:
        return self._text.text()
