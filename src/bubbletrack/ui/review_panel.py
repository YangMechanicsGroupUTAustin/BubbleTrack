"""Review panel: the list of suspicious frames and one-key actions on them."""

from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QGridLayout, QLabel, QListWidget, QListWidgetItem, QPushButton,
    QVBoxLayout, QWidget,
)

if TYPE_CHECKING:
    from bubbletrack.controller.review_controller import ReviewItem

_TRAILING_DIGITS = re.compile(r"(\d+)$")
_SEP = " · "
_HIGH, _MEDIUM = 80, 55  # severity bands for the item colour
_RED, _AMBER = "#ef4444", "#f59e0b"


class ReviewEntry(NamedTuple):
    frame: int  # 0-based
    text: str
    severity: int


def file_number(path: str, index: int) -> str:
    """Trailing digits of the image file name, else the 1-based frame index."""
    if path.startswith("video:"):
        return str(index + 1)
    stem = os.path.splitext(os.path.basename(path))[0]
    match = _TRAILING_DIGITS.search(stem)
    return match.group(1) if match else str(index + 1)


def _reason_text(reason: str, confidence: float) -> str:
    if reason == "low_confidence":
        if np.isfinite(confidence):
            return f"low confidence {confidence:.2f}"
        return "low confidence"
    return reason


def item_text(item: ReviewItem, images) -> str:
    """'Frame 23 (file _033) · low confidence 0.21 · non_circular'."""
    head = f"Frame {item.frame + 1}"
    if 0 <= item.frame < len(images):
        head += f" (file _{file_number(images[item.frame], item.frame)})"
    return _SEP.join([head] + [_reason_text(r, item.confidence) for r in item.reasons])


class ReviewPanel(QWidget):
    """List of frames to review with Accept / Refit / Manual / Delete / Skip.

    The panel only shows entries and reports requests; ReviewController
    decides what they do.  The selected entry is always the frame shown in
    the main view, so actions never hit a frame the user is not looking at.
    """

    ACTIONS = ("accept", "refit", "refit_neighbors", "manual", "delete", "skip")
    _BUTTONS = (  # (action, label, tooltip), laid out in pairs
        ("accept", "Accept (A)", "Keep this result and lock it against Fit All"),
        ("skip", "Skip (S)", "Leave as is and drop from the queue"),
        ("refit", "Refit (R)", "Run detection again on this frame"),
        ("refit_neighbors", "Neighbours (N)", "Refit using neighbouring fits as a prior"),
        ("manual", "Manual (M)", "Click edge points on this frame in the Manual tab"),
        ("delete", "Delete (D)", "Remove this frame's result"),
    )

    count_changed = pyqtSignal(int)
    frame_activated = pyqtSignal(int)  # the user picked an entry: show that frame
    action_requested = pyqtSignal(str)  # one of ACTIONS

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("ReviewPanel")
        self.setMinimumWidth(260)
        self._entries: list[ReviewEntry] = []
        self._count = -1

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        self._count_label = QLabel()
        self._count_label.setStyleSheet("font-weight:bold;")
        layout.addWidget(self._count_label)

        self._list = QListWidget()
        self._list.setObjectName("ReviewList")
        self._list.setWordWrap(True)
        self._list.setSpacing(2)
        self._list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._list.currentRowChanged.connect(self._on_row_changed)
        layout.addWidget(self._list, 1)

        grid = QGridLayout()
        grid.setSpacing(4)
        self._action_buttons: dict[str, QPushButton] = {}
        for i, (action, label, tip) in enumerate(self._BUTTONS):
            btn = QPushButton(label)
            btn.setToolTip(tip)
            btn.clicked.connect(lambda _checked, a=action: self.action_requested.emit(a))
            grid.addWidget(btn, i // 2, i % 2)
            self._action_buttons[action] = btn
        self._prev_btn = QPushButton("▲ Prev (K)")
        self._next_btn = QPushButton("Next (J) ▼")
        self._prev_btn.clicked.connect(lambda: self.step(-1))
        self._next_btn.clicked.connect(lambda: self.step(1))
        nav_row = len(self._BUTTONS) // 2
        grid.addWidget(self._prev_btn, nav_row, 0)
        grid.addWidget(self._next_btn, nav_row, 1)
        layout.addLayout(grid)

        self._update_controls()

    # -- entries ----------------------------------------------------------- #

    def set_entries(self, entries: list[ReviewEntry]) -> None:
        """Show *entries*, keeping the selected frame selected if still listed."""
        entries = list(entries)
        if entries == self._entries:
            return
        keep = self.current_frame()
        self._entries = entries
        self._list.blockSignals(True)
        self._list.clear()
        for entry in entries:
            item = QListWidgetItem(entry.text)
            if entry.severity >= _HIGH:
                item.setForeground(QColor(_RED))
            elif entry.severity >= _MEDIUM:
                item.setForeground(QColor(_AMBER))
            self._list.addItem(item)
        self._list.setCurrentRow(self._row_of(keep))
        self._list.blockSignals(False)
        self._update_controls()

    def frames(self) -> list[int]:
        return [entry.frame for entry in self._entries]

    def count(self) -> int:
        return len(self._entries)

    def texts(self) -> list[str]:
        return [entry.text for entry in self._entries]

    # -- selection --------------------------------------------------------- #

    def current_frame(self) -> int | None:
        row = self._list.currentRow()
        return self._entries[row].frame if 0 <= row < len(self._entries) else None

    def select_frame(self, frame: int | None) -> bool:
        """Silently select *frame*'s entry (or nothing); True if it is listed."""
        row = self._row_of(frame)
        self._list.blockSignals(True)
        self._list.setCurrentRow(row)
        self._list.blockSignals(False)
        self._update_controls()
        return row >= 0

    def step(self, delta: int) -> None:
        """Select the next (+1) or previous (-1) entry and show its frame."""
        if not self._entries:
            return
        row = self._list.currentRow()
        target = 0 if row < 0 else min(max(row + delta, 0), len(self._entries) - 1)
        if target != row:
            self._list.setCurrentRow(target)  # emits frame_activated

    def _row_of(self, frame: int | None) -> int:
        if frame is None:
            return -1
        return next((i for i, e in enumerate(self._entries) if e.frame == frame), -1)

    def _on_row_changed(self, row: int) -> None:
        self._update_controls()
        if 0 <= row < len(self._entries):
            self.frame_activated.emit(self._entries[row].frame)

    def _update_controls(self) -> None:
        selected = self.current_frame() is not None
        for btn in self._action_buttons.values():
            btn.setEnabled(selected)
        self._prev_btn.setEnabled(bool(self._entries))
        self._next_btn.setEnabled(bool(self._entries))
        n = len(self._entries)
        self._count_label.setText(
            f"{n} frame{'s' if n != 1 else ''} to review" if n else "Nothing to review",
        )
        if n != self._count:
            self._count = n
            self.count_changed.emit(n)
