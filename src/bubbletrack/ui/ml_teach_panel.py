"""ML assist's teaching panel (a dock): two good frames, then Train and apply.

::

    Teach ML assist with two good frames
    Step 1 of 2: confirm 2 frames where the outline is right — 1 of 2 done
    Is the red outline on the bubble wall?
    [picture]            [picture]
    Frame 40 · R = 38 px  Frame 120 · Auto's guess, R = 21 px
    ✓ Confirmed           [Looks right ✓]
                          [Looks wrong ✗]
    One more: judge the other frame, which differs from frame 40.
    ▸ Paint examples by hand (optional)
    [progress] [Stop]
    [Train and apply]  [Close]

Painting by hand is folded away; opened, it holds the brushes (B / G / E),
the brush size, Undo stroke and Clear frame.  The panel only shows things and
reports requests; the controller decides.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QMouseEvent, QPixmap
from PyQt6.QtWidgets import (
    QButtonGroup, QFrame, QGridLayout, QHBoxLayout, QLabel, QProgressBar, QPushButton,
    QSizePolicy, QSpinBox, QVBoxLayout, QWidget,
)

TOOLS = ("bubble", "background", "eraser")
TOOL_KEYS = {"bubble": "B", "background": "G", "eraser": "E"}
TOOL_LABELS = {"bubble": "Bubble", "background": "Background", "eraser": "Eraser"}
TOOL_TIPS = {
    "bubble": "Paint on the bubble, its bright core included (B)",
    "background": "Paint on what is not the bubble: speckle, walls, debris (G)",
    "eraser": "Remove painted strokes (E)",
}
TITLE = "Teach ML assist with two good frames"
QUESTION = "Is the red outline on the bubble wall?"
PAINT_TITLE = "Paint examples by hand (optional)"
PAINT_HELP = ("Green on the bubble, red on what is not. A painted frame counts as an example "
              "once it has both. Nothing is trained until Train and apply.")
THUMB_SIZE = (256, 200)  # the whole frame, as wide as the dock allows (height follows)
_OK, _WARN, _DIM = "#10b981", "#f59e0b", "#a1a1aa"
_CHECK, _CROSS = "✓", "✗"
LOOKS_RIGHT = f"Looks right {_CHECK}"
LOOKS_WRONG = f"Looks wrong {_CROSS}"
CONFIRMED = f"{_CHECK} Confirmed"
# The active brush stands out (the shared button style has no checked look).
_TOOL_STYLE = ("QPushButton:checked { background:#6366f1; color:#ffffff;"
               " border:1px solid #818cf8; font-weight:bold; }")


@dataclass(frozen=True, eq=False)
class CardView:
    """What one card shows: a frame to judge, or a confirmed example."""

    frame: int
    caption: str
    confirmed: bool = False
    pixmap: QPixmap | None = None


class _Thumb(QLabel):
    clicked = pyqtSignal()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 (Qt override)
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class _Card(QFrame):
    """One frame: picture, caption, then Looks right / Looks wrong or "✓ Confirmed"."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("mlCard")
        self.setStyleSheet("QFrame#mlCard { border:1px solid rgba(255,255,255,0.08);"
                           " border-radius:6px; }")
        box = QVBoxLayout(self)
        box.setContentsMargins(4, 4, 4, 4)
        box.setSpacing(3)
        self.thumb = _Thumb()
        self.thumb.setFixedSize(*THUMB_SIZE)
        self.thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.thumb.setStyleSheet("background:#0c0d12; border:none;")
        self.thumb.setCursor(Qt.CursorShape.PointingHandCursor)
        self.thumb.setToolTip("Show this frame in the image")
        self.caption = QLabel()
        self.caption.setWordWrap(True)
        self.caption.setStyleSheet(f"color:{_DIM}; font-size:11px; border:none;")
        self.status = QLabel(CONFIRMED)
        self.status.setStyleSheet(f"color:{_OK}; font-weight:bold; border:none;")
        self.right = QPushButton(LOOKS_RIGHT)
        self.right.setToolTip("The red outline sits on the bubble wall: confirm this frame\n"
                              "(it is accepted and locked, and ML assist learns from it)")
        self.wrong = QPushButton(LOOKS_WRONG)
        self.wrong.setObjectName("secondaryBtn")
        self.wrong.setToolTip("The outline is off: show another frame instead\n"
                              "(nothing is changed on this one)")
        for widget in (self.thumb, self.caption, self.status):
            box.addWidget(widget)
        buttons = QHBoxLayout()
        buttons.setSpacing(4)
        buttons.addWidget(self.right)
        buttons.addWidget(self.wrong)
        box.addLayout(buttons)
        self.frame: int | None = None
        self.confirmed = False

    def show_view(self, view: CardView | None, *, enabled: bool = True) -> None:
        self.frame = None if view is None else view.frame
        self.setVisible(view is not None)
        if view is None:
            return
        self.confirmed = view.confirmed
        if view.pixmap is not None:
            self.thumb.setPixmap(view.pixmap)
            self.thumb.setFixedHeight(view.pixmap.height())  # no empty band above / below
        else:
            self.thumb.clear()
            self.thumb.setText("…")
        self.caption.setText(view.caption)
        self.status.setVisible(view.confirmed)
        for button in (self.right, self.wrong):
            button.setVisible(not view.confirmed)
            button.setEnabled(enabled)


class MLTeachPanel(QWidget):
    """What the teaching mode shows; signals report what the user asked for."""

    confirm_requested = pyqtSignal(int)  # Looks right on a frame
    reject_requested = pyqtSignal(int)  # Looks wrong on a frame
    frame_activated = pyqtSignal(int)
    train_clicked = pyqtSignal()
    close_clicked = pyqtSignal()
    stop_clicked = pyqtSignal()
    paint_toggled = pyqtSignal(bool)
    tool_changed = pyqtSignal(str)  # one of TOOLS
    brush_size_changed = pyqtSignal(int)
    undo_clicked = pyqtSignal()
    clear_frame_clicked = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("MLTeachPanel")
        self.setMinimumWidth(290)
        self._train_ok = False
        self._running = False
        self._next_warn = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 8)
        layout.setSpacing(6)
        title = QLabel(TITLE)
        title.setObjectName("sectionTitle")
        layout.addWidget(title)
        self._step = self._line("font-weight:600;")
        self._question = self._line(f"color:{_DIM};")
        self._question.setText(QUESTION)
        layout.addWidget(self._step)
        layout.addWidget(self._question)
        layout.addLayout(self._build_cards())
        self._next = self._line()
        layout.addWidget(self._next)
        layout.addWidget(self._build_paint())
        layout.addLayout(self._build_progress())
        layout.addStretch(1)
        layout.addLayout(self._build_bottom())
        self.set_running(False)

    # -- construction ------------------------------------------------------- #

    @staticmethod
    def _line(style: str = "") -> QLabel:
        label = QLabel()
        label.setWordWrap(True)
        label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        if style:
            label.setStyleSheet(style)
        return label

    def _build_cards(self) -> QVBoxLayout:
        """The two cards one above the other: each picture is the whole frame, full width."""
        row = QVBoxLayout()
        row.setSpacing(6)
        self._cards = [_Card(), _Card()]
        for card in self._cards:
            card.thumb.clicked.connect(lambda c=card: self._emit(self.frame_activated, c))
            card.right.clicked.connect(lambda _on, c=card: self._emit(self.confirm_requested, c))
            card.wrong.clicked.connect(lambda _on, c=card: self._emit(self.reject_requested, c))
            row.addWidget(card)
            card.hide()
        row.addStretch(1)
        return row

    def _build_paint(self) -> QWidget:
        box = QWidget()
        outer = QVBoxLayout(box)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)
        self.paint_toggle = QPushButton()
        self.paint_toggle.setObjectName("collapseToggle")
        self.paint_toggle.setCheckable(True)
        self.paint_toggle.setToolTip("Brushes for frames without a good outline (rarely needed)")
        self.paint_toggle.toggled.connect(self._on_paint_toggled)
        outer.addWidget(self.paint_toggle)
        self._paint_body = QWidget()
        body = QVBoxLayout(self._paint_body)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(4)
        help_label = self._line(f"color:{_DIM};")
        help_label.setText(PAINT_HELP)
        body.addWidget(help_label)
        body.addLayout(self._build_tools())
        body.addLayout(self._build_edit_row())
        self._paint_summary = self._line(f"color:{_DIM};")
        body.addWidget(self._paint_summary)
        outer.addWidget(self._paint_body)
        self._on_paint_toggled(False)
        return box

    def _build_tools(self) -> QGridLayout:
        grid = QGridLayout()
        grid.setSpacing(4)
        self._tool_group = QButtonGroup(self)
        self._tool_group.setExclusive(True)
        self._tool_buttons: dict[str, QPushButton] = {}
        for i, tool in enumerate(TOOLS):
            btn = QPushButton(f"{TOOL_LABELS[tool]} ({TOOL_KEYS[tool]})")
            btn.setCheckable(True)
            btn.setObjectName("secondaryBtn")
            btn.setStyleSheet(_TOOL_STYLE)
            btn.setToolTip(TOOL_TIPS[tool])
            btn.clicked.connect(lambda _checked, t=tool: self.tool_changed.emit(t))
            self._tool_group.addButton(btn)
            self._tool_buttons[tool] = btn
            grid.addWidget(btn, 0, i)
        size_row = QHBoxLayout()
        size_row.addWidget(QLabel("Brush size:"))
        self._size = QSpinBox()
        self._size.setRange(1, 80)
        self._size.setSuffix(" px")
        self._size.setToolTip("Brush radius; the mouse wheel over the image changes it\n"
                              "(Ctrl+wheel zooms)")
        self._size.valueChanged.connect(self.brush_size_changed)
        size_row.addWidget(self._size)
        size_row.addStretch(1)
        grid.addLayout(size_row, 1, 0, 1, 3)
        return grid

    def _build_edit_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self._undo_btn = QPushButton("Undo stroke")
        self._undo_btn.setObjectName("secondaryBtn")
        self._undo_btn.setToolTip("Undo the last stroke (Ctrl+Z)")
        self._undo_btn.clicked.connect(self.undo_clicked)
        self._clear_btn = QPushButton("Clear frame")
        self._clear_btn.setObjectName("secondaryBtn")
        self._clear_btn.setToolTip("Remove every stroke on the shown frame")
        self._clear_btn.clicked.connect(self.clear_frame_clicked)
        row.addWidget(self._undo_btn)
        row.addWidget(self._clear_btn)
        return row

    def _build_progress(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setTextVisible(False)
        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setObjectName("dangerBtn")
        self._stop_btn.setToolTip("Stop: the frames measured so far keep ML assist's results")
        self._stop_btn.clicked.connect(self.stop_clicked)
        row.addWidget(self._progress, 1)
        row.addWidget(self._stop_btn)
        return row

    def _build_bottom(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self.train_button = QPushButton("Train and apply")
        self.train_button.clicked.connect(self.train_clicked)
        self._close_btn = QPushButton("Close")
        self._close_btn.setObjectName("secondaryBtn")
        self._close_btn.setToolTip("Leave ML assist (your confirmed frames and strokes are kept)")
        self._close_btn.clicked.connect(self.close_clicked)
        row.addWidget(self.train_button, 1)
        row.addWidget(self._close_btn)
        return row

    # -- the step, the cards, the next step --------------------------------- #

    def set_step(self, text: str) -> None:
        self._step.setText(text)

    def step_text(self) -> str:
        return self._step.text()

    def question_text(self) -> str:
        return self._question.text()

    def set_cards(self, views: Sequence[CardView | None]) -> None:
        """Up to two cards; None hides a card.  The question shows while one asks."""
        views = list(views)[:len(self._cards)]
        views += [None] * (len(self._cards) - len(views))
        for card, view in zip(self._cards, views):
            card.show_view(view, enabled=not self._running)
        self._question.setVisible(any(v is not None and not v.confirmed for v in views))

    def card_frames(self) -> list[int]:
        return [c.frame for c in self._cards if c.frame is not None]

    def _card(self, frame: int) -> _Card | None:
        return next((c for c in self._cards if c.frame == frame), None)

    def card_caption(self, frame: int) -> str:
        card = self._card(frame)
        return "" if card is None else card.caption.text()

    def card_confirmed(self, frame: int) -> bool:
        card = self._card(frame)
        return card is not None and card.confirmed

    def _emit(self, signal, card: _Card) -> None:
        if card.frame is not None:
            signal.emit(card.frame)

    def set_next(self, text: str, warn: bool = False) -> None:
        self._next.setText(text)
        self._next_warn = bool(warn)
        self._next.setStyleSheet(f"color:{_WARN};" if warn else "")

    def next_text(self) -> str:
        return self._next.text()

    def next_is_warning(self) -> bool:
        return self._next_warn

    def set_train(self, enabled: bool, tip: str) -> None:
        self._train_ok = bool(enabled)
        self.train_button.setToolTip(tip)
        self.train_button.setEnabled(self._train_ok and not self._running)

    # -- painting by hand ----------------------------------------------------- #

    def set_painting(self, open_: bool) -> None:
        """Open or fold "Paint examples by hand" (as a click would)."""
        self.paint_toggle.setChecked(bool(open_))

    def painting_open(self) -> bool:
        return self.paint_toggle.isChecked()

    def _on_paint_toggled(self, open_: bool) -> None:
        self._paint_body.setVisible(open_)
        self.paint_toggle.setText(f"  {'▾' if open_ else '▸'} {PAINT_TITLE}")
        self.paint_toggled.emit(bool(open_))

    def set_paint_summary(self, text: str) -> None:
        self._paint_summary.setText(text)

    def paint_summary(self) -> str:
        return self._paint_summary.text()

    def set_tool(self, tool: str) -> None:
        self._tool_buttons[tool].setChecked(True)

    def tool(self) -> str:
        return next((t for t, b in self._tool_buttons.items() if b.isChecked()), TOOLS[0])

    def set_brush_size(self, radius: int) -> None:
        self._size.blockSignals(True)
        self._size.setValue(int(radius))
        self._size.blockSignals(False)

    def brush_size(self) -> int:
        return int(self._size.value())

    # -- runs ------------------------------------------------------------------ #

    def set_running(self, running: bool) -> None:
        """While a run writes results: no judging or painting, Stop shown."""
        self._running = bool(running)
        for widget in (*self._tool_buttons.values(), self._size, self._undo_btn, self._clear_btn,
                       *(b for c in self._cards for b in (c.right, c.wrong))):
            widget.setEnabled(not running)
        self.train_button.setEnabled(self._train_ok and not running)
        self._progress.setVisible(running)
        self._stop_btn.setVisible(running)
        if not running:
            self._progress.setValue(0)

    def set_progress(self, current: int, total: int) -> None:
        self._progress.setValue(int(100 * current / max(total, 1)))
