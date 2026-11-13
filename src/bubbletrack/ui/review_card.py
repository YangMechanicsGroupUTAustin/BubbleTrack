"""The Review dock's lower half: the frame on screen and what to do with it.

::

    Frame 125 · _135                      6 of 61
    No result, but Auto has a guess (dashed outline).
    Accept it if it sits on the bubble wall.
    [ Accept the guess                          A ]   <- recommended: filled
    [ Fix with a circle                         F ]
    [ No bubble here                            N ]
    [ Accept all 61       ⇧A ] [ Skip   S ] [ ⋯ ]
    ✓ Frame 124 accepted. Ctrl+Z undoes     [Undo]

The card shows; ReviewController decides (``Advice`` picks the filled button).
"""

from __future__ import annotations

from typing import NamedTuple

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QMenu, QPushButton, QSizePolicy, QToolButton, QVBoxLayout,
    QWidget,
)

FRAME_ACTIONS = ("accept", "fix", "no_bubble")


class CardAdvice(NamedTuple):
    """What the card says about the frame on screen (from ``review_advice``)."""

    text: str  # what is wrong and what to do
    action: str  # the recommended (filled) action
    guess: bool = False  # the frame shows Auto's guess: Accept stores it
    can_accept: bool = True  # False on an empty frame: nothing to accept
_KEYS = {"accept": "A", "fix": "F", "no_bubble": "N", "accept_all": "⇧A", "skip": "S"}
_TIPS = {
    "accept": "Keep this frame's result (or store Auto's dashed guess) and lock it, so Fit All "
              "keeps it. Ctrl+Z undoes.",
    "fix": "Draw the bubble yourself: press on its centre and drag to its wall (Auto measures "
           "the exact wall), or click edge points. Esc cancels.",
    "no_bubble": "There is no bubble on this frame: it stays empty and locked (exported as no "
                 "bubble). Ctrl+Z undoes.",
    "accept_all": "Accept every frame of this row as it is, Auto's guesses included, and lock "
                  "them. Ctrl+Z undoes.",
    "skip": "Leave this row as it is (nothing is stored) and go to the next one.",
}
_DIM, _FAINT, _OK = "#a1a1aa", "#71717a", "#34d399"


class ActionButton(QPushButton):
    """A full-width action with its key at the right; filled when recommended."""

    def __init__(self, action: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.action = action
        self.setToolTip(f"{_TIPS[action]} (key {_KEYS[action]})")
        self._key = QLabel(_KEYS[action], self)
        self._key.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 10, 0)
        row.addStretch()
        row.addWidget(self._key)
        self._primary: bool | None = None
        self.set_primary(False)

    def set_primary(self, on: bool) -> None:
        if on == self._primary:
            return
        self._primary = on
        self.setObjectName("" if on else "secondaryBtn")  # the app stylesheet's two looks
        self.setStyleSheet("QPushButton { text-align: left; padding: 3px 10px; "
                           "min-height: 24px; }")
        self.style().unpolish(self)
        self.style().polish(self)
        colour = "rgba(255, 255, 255, 0.70)" if on else _FAINT
        self._key.setStyleSheet(f"color:{colour}; font-size:10px; background:transparent;")

    def is_primary(self) -> bool:
        return bool(self._primary)


class ReviewCard(QWidget):
    """Where the shown frame is, what is wrong, and the actions (see module doc)."""

    action_requested = pyqtSignal(str)  # accept | fix | no_bubble | accept_all | skip | no_bubble_all
    export_case_requested = pyqtSignal()  # More > Export these frames as a hard case

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._row_count = 0  # frames of the selected row (0: none)
        self._fixing = False
        self._can_accept = True
        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(5)
        rule = QFrame()  # sets the frame on screen apart from the list above
        rule.setFixedHeight(1)
        rule.setStyleSheet("background: rgba(255, 255, 255, 0.08);")
        box.addWidget(rule)
        box.addSpacing(2)
        head = QHBoxLayout()
        self.title = QLabel()
        self.title.setObjectName("reviewFrameTitle")
        self.title.setStyleSheet("font-weight:600;")
        self.position = QLabel()
        self.position.setStyleSheet(f"color:{_FAINT}; font-size:11px;")
        head.addWidget(self.title, 1)
        head.addWidget(self.position)
        box.addLayout(head)
        self.advice = QLabel()
        self.advice.setObjectName("reviewAdvice")
        self.advice.setWordWrap(True)
        self.advice.setStyleSheet(f"color:{_DIM}; font-size:11px;")
        self.advice.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        box.addWidget(self.advice)
        self.buttons: dict[str, ActionButton] = {}
        for action in FRAME_ACTIONS:
            box.addWidget(self._add_button(action))
        box.addLayout(self._build_row_actions())

    def _add_button(self, action: str) -> ActionButton:
        btn = ActionButton(action)
        btn.clicked.connect(lambda _checked, a=action: self.action_requested.emit(a))
        self.buttons[action] = btn
        return btn

    def _build_row_actions(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(4)
        row.addWidget(self._add_button("accept_all"), 3)
        row.addWidget(self._add_button("skip"), 2)
        self.more = QToolButton()
        self.more.setText("⋯")
        self.more.setToolTip("More for this row")
        self.more.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.more.setStyleSheet("QToolButton { padding: 2px 8px; } "
                                "QToolButton::menu-indicator { image: none; }")
        self._more_menu = QMenu(self.more)
        self._more_menu.setToolTipsVisible(True)
        self._no_bubble_all = self._more_menu.addAction("")
        self._no_bubble_all.setToolTip("There is no bubble on any frame of this row "
                                       "(Shift+N). Ctrl+Z undoes.")
        self._no_bubble_all.triggered.connect(lambda: self.action_requested.emit("no_bubble_all"))
        case = self._more_menu.addAction("Export these frames as a hard case...")
        case.setToolTip("Save these frames, the settings and your fixes as one .zip "
                        "to send to the maintainer")
        case.triggered.connect(self.export_case_requested)
        self.more.setMenu(self._more_menu)
        row.addWidget(self.more)
        return row

    # -- content ----------------------------------------------------------- #

    def show_frame(self, title: str, position: str, advice: CardAdvice, *, row_count: int,
                   fix_label: str, why: str = "") -> None:
        """Fill the card for the frame on screen (*advice.action*: the filled button).

        *row_count*: frames of the selected row (0 = no row: no row actions).
        """
        self.title.setText(title)
        self.position.setText(position)
        self.advice.setText(advice.text)
        self.advice.setToolTip(why)
        self.buttons["accept"].setText("Accept the guess" if advice.guess
                                       else "Accept this frame")
        self.buttons["fix"].setText(fix_label)
        self.buttons["no_bubble"].setText("No bubble here")
        self._can_accept = advice.can_accept
        self._row_count = row_count
        recommended = advice.action
        self.buttons["accept_all"].setText(f"Accept all {row_count}")
        self.buttons["skip"].setText("Skip row" if row_count > 1 else "Skip")
        self._no_bubble_all.setText(f"No bubble on all {row_count} frames\tShift+N")
        self._no_bubble_all.setVisible(row_count > 1)
        for action, btn in self.buttons.items():
            btn.set_primary(action == recommended)
        self._show_actions()

    def set_fixing(self, on: bool) -> None:
        """While a fix runs, its strip is the one place to act: the buttons step aside."""
        self._fixing = on
        self._show_actions()

    def _show_actions(self) -> None:
        on, rows = not self._fixing, self._row_count
        for action in FRAME_ACTIONS:
            self.buttons[action].setVisible(on)
        # Nothing to accept on an empty frame (its A key means "no bubble", like N).
        self.buttons["accept"].setVisible(on and self._can_accept)
        self.buttons["accept_all"].setVisible(on and rows > 1)
        self.buttons["skip"].setVisible(on and rows > 0)
        self.more.setVisible(on and rows > 0)

    def recommended(self) -> str | None:
        return next((a for a, b in self.buttons.items() if b.is_primary()), None)


class FeedbackLine(QWidget):
    """What the last action did, with Undo ('✓ Frame 124 accepted.  [Undo]')."""

    undo_requested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        self._text = QLabel()
        self._text.setObjectName("reviewFeedback")
        self._text.setWordWrap(True)
        self._text.setStyleSheet(f"color:{_OK}; font-size:11px;")
        self._text.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        row.addWidget(self._text, 1)
        self._undo = QPushButton("Undo")
        self._undo.setObjectName("secondaryBtn")
        self._undo.setStyleSheet("QPushButton { padding: 1px 8px; min-height: 20px; }")
        self._undo.setToolTip("Undo the last action (Ctrl+Z)")
        self._undo.clicked.connect(self.undo_requested)
        row.addWidget(self._undo, 0, Qt.AlignmentFlag.AlignTop)
        self.clear()

    def show_text(self, text: str, *, undo: bool = True) -> None:
        self._text.setText(f"✓ {text}")
        self._undo.setVisible(undo)
        self.show()

    def clear(self) -> None:
        self._text.setText("")
        self.hide()

    def text(self) -> str:
        return self._text.text()

    def offers_undo(self) -> bool:
        return not self.isHidden() and not self._undo.isHidden()


class DonePage(QWidget):
    """The list's place when nothing waits: a message, and Export when all is done."""

    export_requested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        box = QVBoxLayout(self)
        box.setContentsMargins(16, 16, 16, 16)
        box.addStretch()
        self._text = QLabel()
        self._text.setObjectName("reviewEmpty")
        self._text.setWordWrap(True)
        self._text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._text.setStyleSheet(f"color:{_DIM};")
        box.addWidget(self._text)
        self._export = QPushButton("Export R(t)...")
        self._export.setToolTip("Save the radius curve of every frame")
        self._export.clicked.connect(self.export_requested)
        box.addWidget(self._export, 0, Qt.AlignmentFlag.AlignHCenter)
        box.addStretch()

    def set_message(self, text: str, *, offer_export: bool) -> None:
        self._text.setText(text)
        self._export.setVisible(offer_export)

    def text(self) -> str:
        return self._text.text()

    def offers_export(self) -> bool:
        return not self._export.isHidden()
