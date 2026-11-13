"""Fix strip: the Review dock's tools for fixing the frame on screen.

Shown only while a fix runs (Fix, F)::

    Fix frame 45                                  [Cancel]
    Press on the bubble centre, drag to its wall ...
    [Click edge points instead]

The default tool is "Drag a circle"; "Click edge points" is the fallback
(and the only tool outside the Auto engine).  When Auto finds no wall
matching a drawn circle, the prompt replaces the hint ("Use my circle as a
manual fit" / "Cancel").  The strip only reports requests; ``FixMode``
(controller) decides what they do.
"""

from __future__ import annotations

from typing import Protocol

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget,
)

from bubbletrack.ui.circle_tool import (
    CIRCLE_HINT, CIRCLE_TIP, CIRCLE_UNAVAILABLE_TIP, POINTS_TIP, TOOL_CIRCLE, TOOL_POINTS,
    CirclePrompt,
)

MIN_POINTS = 3  # a circle fit needs three edge points


class FixView(Protocol):
    """What the fix controllers (circle, points) tell the user."""

    def set_point_count(self, n: int) -> None: ...
    def set_circle_available(self, available: bool) -> None: ...
    def show_circle_prompt(self, text: str, offer_manual: bool = True) -> None: ...
    def hide_circle_prompt(self) -> None: ...


class NoFixView:
    """Stand-in until the Review dock attaches its strip (tests, start-up)."""

    def set_point_count(self, n: int) -> None:
        pass

    def set_circle_available(self, available: bool) -> None:
        pass

    def show_circle_prompt(self, text: str, offer_manual: bool = True) -> None:
        pass

    def hide_circle_prompt(self) -> None:
        pass


def points_hint(n: int) -> str:
    """How many points are clicked, and what is still needed before Done."""
    if n >= MIN_POINTS:
        return f"{n} points on the wall: Done fits a circle through them"
    return f"Click at least {MIN_POINTS} points on the bubble wall · {n} so far"


def _button(text: str, tip: str, *, secondary: bool = True) -> QPushButton:
    btn = QPushButton(text)
    btn.setToolTip(tip)
    if secondary:
        btn.setObjectName("secondaryBtn")
    return btn


class FixStrip(QWidget):
    """The fix tools of one frame (see module doc); hidden while not fixing."""

    points_requested = pyqtSignal()  # "Click edge points instead"
    circle_requested = pyqtSignal()  # "Drag a circle instead"
    done_clicked = pyqtSignal()  # fit the clicked points
    cancel_clicked = pyqtSignal()  # leave the fix; the frame stays as it was
    use_circle_clicked = pyqtSignal()  # store the refused circle as a manual fit
    active_changed = pyqtSignal(bool)  # shown (a fix started) / hidden (it ended)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 0, 8, 8)  # in line with the Review panel above
        card = QFrame()
        card.setObjectName("fixStrip")
        card.setStyleSheet(
            "QFrame#fixStrip { background: rgba(99, 102, 241, 0.10);"
            " border: 1px solid rgba(99, 102, 241, 0.45); border-radius: 8px; }"
            " QFrame#fixStrip QLabel { background: transparent; border: none; }"
        )
        outer.addWidget(card)
        self._frame: int | None = None
        self._tool: str | None = None
        self._circle_available = True
        self._points = 0
        box = QVBoxLayout(card)
        box.setContentsMargins(10, 8, 10, 8)
        box.setSpacing(6)
        box.addLayout(self._build_title())
        self._hint = QLabel()
        self._hint.setObjectName("fixHint")
        self._hint.setWordWrap(True)
        self._hint.setStyleSheet("color:#a1a1aa; font-size:11px;")
        self._hint.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        box.addWidget(self._hint)
        self.prompt = CirclePrompt()
        self.prompt.use_circle_clicked.connect(self.use_circle_clicked)
        self.prompt.cancel_clicked.connect(self.cancel_clicked)
        box.addWidget(self.prompt)
        box.addLayout(self._build_tools())
        self.hide()

    def _build_title(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self._title = QLabel()
        self._title.setObjectName("fixTitle")
        self._title.setStyleSheet("font-weight:600;")
        row.addWidget(self._title, 1)
        self._cancel_btn = _button("Cancel (Esc)", "Stop fixing; the frame stays as it was")
        self._cancel_btn.clicked.connect(self.cancel_clicked)
        row.addWidget(self._cancel_btn)
        return row

    def _build_tools(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(4)
        self._done_btn = _button("Done", "Fit a circle through the clicked points (Enter)",
                                 secondary=False)
        self._done_btn.clicked.connect(self.done_clicked)
        self._points_btn = _button("Click edge points instead",
                                   "Fix this frame by clicking points on the bubble wall")
        self._points_btn.clicked.connect(self.points_requested)
        self._circle_btn = _button("Drag a circle instead", "Press on the bubble centre and "
                                   "drag to its wall: Auto measures the exact wall")
        self._circle_btn.clicked.connect(self.circle_requested)
        for btn in (self._done_btn, self._points_btn, self._circle_btn):
            row.addWidget(btn)
        row.addStretch()
        return row

    # -- the fix ----------------------------------------------------------- #

    def frame(self) -> int | None:
        """The frame being fixed (0-based), None while hidden."""
        return self._frame

    def tool(self) -> str | None:
        """TOOL_CIRCLE or TOOL_POINTS while fixing, None while hidden."""
        return self._tool

    def show_tool(self, frame: int, tool: str) -> None:
        """Show the strip for *frame* with *tool* in use."""
        if tool not in (TOOL_CIRCLE, TOOL_POINTS):
            raise ValueError(f"unknown fix tool {tool!r}")
        started = self._frame is None
        self._frame, self._tool = frame, tool
        self._points = 0
        self.prompt.hide_prompt()
        self._title.setText(f"Fixing frame {frame + 1}")
        self._update()
        self.show()
        if started:
            self.active_changed.emit(True)

    def end(self) -> None:
        """Hide the strip (the fix is over)."""
        ended = self._frame is not None
        self._frame = self._tool = None
        self.prompt.hide_prompt()
        self.hide()
        if ended:
            self.active_changed.emit(False)

    # -- FixView ----------------------------------------------------------- #

    def set_point_count(self, n: int) -> None:
        self._points = max(int(n), 0)
        self._update()

    def set_circle_available(self, available: bool) -> None:
        self._circle_available = bool(available)
        self._update()

    def show_circle_prompt(self, text: str, offer_manual: bool = True) -> None:
        self.prompt.show_prompt(text, offer_manual)
        self._update()

    def hide_circle_prompt(self) -> None:
        self.prompt.hide_prompt()
        self._update()

    # -- inspection (tests) ------------------------------------------------ #

    def hint_text(self) -> str:
        return self._hint.text()

    def prompt_visible(self) -> bool:
        return self.prompt.prompt_visible()

    def prompt_text(self) -> str:
        return self.prompt.prompt_text()

    def offers_manual(self) -> bool:
        return self.prompt.offers_manual()

    # -- view -------------------------------------------------------------- #

    def _update(self) -> None:
        circle, prompt = self._tool == TOOL_CIRCLE, self.prompt.prompt_visible()
        if circle and not self._circle_available:
            self._hint.setText(CIRCLE_UNAVAILABLE_TIP.replace("\n", " "))
        else:
            self._hint.setText(CIRCLE_HINT if circle else points_hint(self._points))
        self._hint.setToolTip(CIRCLE_TIP if circle else POINTS_TIP)
        self._hint.setVisible(not prompt)
        self._cancel_btn.setVisible(not prompt)  # the prompt has its own Cancel
        self._done_btn.setVisible(self._tool == TOOL_POINTS)
        self._done_btn.setEnabled(self._points >= MIN_POINTS)
        self._points_btn.setVisible(circle)
        self._circle_btn.setVisible(self._tool == TOOL_POINTS)
        self._circle_btn.setEnabled(self._circle_available)
        self._circle_btn.setToolTip(
            "Press on the bubble centre and drag to its wall: Auto measures the exact wall"
            if self._circle_available else CIRCLE_UNAVAILABLE_TIP)
