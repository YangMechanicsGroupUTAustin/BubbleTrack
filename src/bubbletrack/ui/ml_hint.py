"""The short, non-blocking ML assist line in the Review dock.

* After a run where many frames fail: what failed in a few words, and
  "Try ML assist…" (it opens the teaching mode on step 1).
* After Train and apply: what ML assist did ("ML assist measured 41 of 52
  frames to check; 9 are listed in Review"), with no button: the next step
  is the list below.
"""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QToolButton, QWidget

_HINT_STYLE = ("QFrame#mlHint { background: rgba(245, 158, 11, 0.12);"
               " border: 1px solid rgba(245, 158, 11, 0.35); border-radius: 6px; }")
_NOTE_STYLE = ("QFrame#mlHint { background: rgba(16, 185, 129, 0.12);"
               " border: 1px solid rgba(16, 185, 129, 0.35); border-radius: 6px; }")
_HINT_TEXT, _NOTE_TEXT = "color:#fcd34d; border:none;", "color:#6ee7b7; border:none;"
OPEN_TIP = ("ML assist helps Auto on hard recordings. Confirm two frames where\n"
            "Auto's outline is right; ML assist learns from them and measures\n"
            "the frames to check.")


class MLHint(QFrame):
    """One line of text, an 'ML assist…' button (hints only) and a close cross."""

    open_clicked = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("mlHint")
        row = QHBoxLayout(self)
        row.setContentsMargins(8, 4, 4, 4)
        self._text = QLabel()
        self._text.setWordWrap(True)
        self._text.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self._open = QPushButton("Try ML assist…")
        self._open.setObjectName("secondaryBtn")
        self._open.setToolTip(OPEN_TIP)
        self._open.clicked.connect(self.open_clicked)
        close = QToolButton()
        close.setText("×")
        close.setToolTip("Hide this line")
        close.setAutoRaise(True)
        close.clicked.connect(self.hide)
        row.addWidget(self._text, 1)
        row.addWidget(self._open)
        row.addWidget(close)
        self.hide()

    def show_text(self, text: str) -> None:
        """A hint: many frames failed, ML assist can help."""
        self._show(text, hint=True)

    def show_note(self, text: str) -> None:
        """What Train and apply did (no button: Review's list is the next step)."""
        self._show(text, hint=False)

    def _show(self, text: str, *, hint: bool) -> None:
        self.setStyleSheet(_HINT_STYLE if hint else _NOTE_STYLE)
        self._text.setStyleSheet(_HINT_TEXT if hint else _NOTE_TEXT)
        self._text.setText(text)
        self._open.setVisible(hint)
        self.show()

    def text(self) -> str:
        return self._text.text()

    def offers_teaching(self) -> bool:
        return not self._open.isHidden()
