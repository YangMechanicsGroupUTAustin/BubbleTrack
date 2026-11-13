"""ML assist's teaching panel (a dock): suggested frames, brushes, quality, Apply.

::

    Paint a few strokes: green on the bubble, red on what is not ...
    Look at these frames
    [preview]      [preview]
    Frame 12 ✓        Frame 40
    [Looks right ✓]  [Looks right ✓]
    [Bubble (B)] [Background (G)] [Eraser (E)]   Size [ 5 ] px
    [Undo stroke] [Clear frame]
    2 verified frames, 3 strokes — ready
    Frame 12: Auto (ML-guided), R = 40.2 px
    [Apply to all frames]  [Close]

The panel only shows things and reports requests; the controller decides.
"""

from __future__ import annotations

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
HELP_TEXT = ("Paint a few strokes on a frame: green on the bubble, red on what is not. "
             "Frames you verified count automatically. The right panel shows what ML sees.")
THUMB_SIZE = (132, 88)
_OK, _WARN, _DIM = "#10b981", "#f59e0b", "#a1a1aa"
_CHECK = "✓"
LOOKS_RIGHT = f"Looks right {_CHECK}"
IN_REVIEW = "will be listed in Review for you to check"
# The active brush stands out (the shared button style has no checked look).
_TOOL_STYLE = ("QPushButton:checked { background:#6366f1; color:#ffffff;"
               " border:1px solid #818cf8; font-weight:bold; }")


class _Thumb(QLabel):
    clicked = pyqtSignal()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 (Qt override)
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class _Card(QFrame):
    """One suggested frame: preview, caption and 'Looks right ✓'."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("mlCard")
        self.setStyleSheet("QFrame#mlCard { border:1px solid rgba(255,255,255,0.08); border-radius:6px; }")
        box = QVBoxLayout(self)
        box.setContentsMargins(4, 4, 4, 4)
        box.setSpacing(3)
        self.thumb = _Thumb()
        self.thumb.setFixedSize(*THUMB_SIZE)
        self.thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.thumb.setStyleSheet("background:#0c0d12; border:none;")
        self.thumb.setCursor(Qt.CursorShape.PointingHandCursor)
        self.thumb.setToolTip("Show this frame")
        self.caption = QLabel()
        self.caption.setStyleSheet(f"color:{_DIM}; font-size:11px; border:none;")
        self.correct = QPushButton(LOOKS_RIGHT)
        self.correct.setObjectName("secondaryBtn")
        self.correct.setToolTip("The result on this frame is right: accept (lock) it,\n"
                                "and ML assist learns from it")
        for widget in (self.thumb, self.caption, self.correct):
            box.addWidget(widget)
        self.frame: int | None = None


class MLTeachPanel(QWidget):
    """What the teaching mode shows; signals report what the user asked for."""

    tool_changed = pyqtSignal(str)  # one of TOOLS
    brush_size_changed = pyqtSignal(int)
    undo_clicked = pyqtSignal()
    clear_frame_clicked = pyqtSignal()
    apply_clicked = pyqtSignal()
    close_clicked = pyqtSignal()
    stop_clicked = pyqtSignal()
    frame_activated = pyqtSignal(int)
    accept_requested = pyqtSignal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("MLTeachPanel")
        self.setMinimumWidth(290)
        self._ready = False
        self._running = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 8)
        layout.setSpacing(8)
        help_label = QLabel(HELP_TEXT)
        help_label.setWordWrap(True)
        help_label.setStyleSheet(f"color:{_DIM};")
        layout.addWidget(help_label)
        layout.addWidget(self._section("Look at these frames"))
        layout.addLayout(self._build_cards())
        layout.addLayout(self._build_tools())
        layout.addLayout(self._build_edit_row())
        self._quality = self._line()
        self._preview = self._line()
        layout.addWidget(self._quality)
        layout.addWidget(self._preview)
        layout.addLayout(self._build_progress())
        layout.addStretch(1)
        layout.addLayout(self._build_bottom())
        self.set_running(False)

    # -- construction ------------------------------------------------------- #

    @staticmethod
    def _section(text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("sectionTitle")
        return label

    @staticmethod
    def _line() -> QLabel:
        label = QLabel()
        label.setWordWrap(True)
        label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        return label

    def _build_cards(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(6)
        self._cards = [_Card(), _Card()]
        for card in self._cards:
            card.thumb.clicked.connect(lambda c=card: self._activate(c))
            card.correct.clicked.connect(lambda _checked, c=card: self._accept(c))
            row.addWidget(card)
            card.hide()
        row.addStretch(1)
        return row

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
        self._stop_btn.clicked.connect(self.stop_clicked)
        row.addWidget(self._progress, 1)
        row.addWidget(self._stop_btn)
        return row

    def _build_bottom(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self._apply_btn = QPushButton("Apply to all frames")
        self._apply_btn.setToolTip("Measure every frame you have not confirmed with ML assist\n"
                                   "(manual, circle-fixed and locked frames are kept).\n"
                                   f"ML's own outlines {IN_REVIEW}.")
        self._apply_btn.clicked.connect(self.apply_clicked)
        self._close_btn = QPushButton("Close")
        self._close_btn.setObjectName("secondaryBtn")
        self._close_btn.setToolTip("Leave the teaching mode (your strokes are kept)")
        self._close_btn.clicked.connect(self.close_clicked)
        row.addWidget(self._apply_btn, 1)
        row.addWidget(self._close_btn)
        return row

    # -- suggested frames ---------------------------------------------------- #

    def set_suggestions(self, frames) -> None:
        frames = list(frames)[:len(self._cards)]
        for card, frame in zip(self._cards, frames + [None] * len(self._cards)):
            card.frame = frame
            card.setVisible(frame is not None)
            if frame is not None:
                card.thumb.clear()
                card.thumb.setText("…")
                card.caption.setText(f"Frame {frame + 1}")

    def suggestions(self) -> list[int]:
        return [c.frame for c in self._cards if c.frame is not None]

    def set_card(self, frame: int, *, caption: str | None = None, pixmap: QPixmap | None = None,
                 verified: bool | None = None, can_accept: bool | None = None) -> None:
        card = next((c for c in self._cards if c.frame == frame), None)
        if card is None:
            return
        if pixmap is not None:
            card.thumb.setPixmap(pixmap)
        if caption is not None or verified is not None:
            text = caption if caption is not None else card.caption.text().rstrip(f" {_CHECK}")
            card.caption.setText(text + (f" {_CHECK}" if verified else ""))
            card.caption.setToolTip("Verified: an example for ML assist" if verified else "")
        if can_accept is not None:
            card.correct.setEnabled(can_accept)

    def card_caption(self, frame: int) -> str:
        card = next((c for c in self._cards if c.frame == frame), None)
        return "" if card is None else card.caption.text()

    def _activate(self, card: _Card) -> None:
        if card.frame is not None:
            self.frame_activated.emit(card.frame)

    def _accept(self, card: _Card) -> None:
        if card.frame is not None:
            self.accept_requested.emit(card.frame)

    # -- tools, lines and run state ---------------------------------------------- #

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

    def set_quality(self, text: str, ready: bool) -> None:
        self._quality.setText(text)
        self._quality.setStyleSheet(f"color:{_OK if ready else _WARN};")
        self._ready = bool(ready)
        self._apply_btn.setEnabled(self._ready and not self._running)

    def set_quality_tip(self, text: str) -> None:
        self._quality.setToolTip(text)

    def quality_text(self) -> str:
        return self._quality.text()

    def set_preview(self, text: str, warn: bool = False) -> None:
        self._preview.setText(text)
        self._preview.setStyleSheet(f"color:{_WARN if warn else _DIM};")

    def preview_text(self) -> str:
        return self._preview.text()

    def set_running(self, running: bool) -> None:
        """While a run (Fit All, Apply, re-track) writes results: no painting, Stop shown."""
        self._running = running
        for widget in (*self._tool_buttons.values(), self._size, self._undo_btn, self._clear_btn,
                       *(c.correct for c in self._cards)):
            widget.setEnabled(not running)
        self._apply_btn.setEnabled(self._ready and not running)
        self._progress.setVisible(running)
        self._stop_btn.setVisible(running)
        if not running:
            self._progress.setValue(0)

    def set_progress(self, current: int, total: int) -> None:
        self._progress.setValue(int(100 * current / max(total, 1)))
