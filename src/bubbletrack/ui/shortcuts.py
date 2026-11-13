"""Keyboard shortcuts for BubbleTrack."""

from __future__ import annotations

from collections.abc import Callable

from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QAbstractSpinBox, QApplication, QComboBox, QLineEdit, QPlainTextEdit,
    QTextEdit, QWidget,
)

# Review queue keys: letters (with Shift for the whole row), so they never
# collide with the global bindings below (arrows, + = - 0, Space, Ctrl+Z),
# Ctrl+S / Ctrl+O or ML assist's teaching keys (B, G, E).  Up/Down move
# through the list while it has focus (Qt's own keys).
REVIEW_BINDINGS: dict[str, str] = {
    "A": "accept",
    "Shift+A": "accept_all",
    "F": "fix",
    "N": "no_bubble",
    "Shift+N": "no_bubble_all",
    "S": "skip",
    "J": "next",
    "K": "prev",
}
# Live only while a fix runs (the Review dock's fix strip).
FIX_BINDINGS: dict[str, str] = {
    "Esc": "fix_cancel",
    "Return": "fix_done",
}

_TEXT_INPUTS = (QLineEdit, QTextEdit, QPlainTextEdit, QAbstractSpinBox, QComboBox)


def text_input_has_focus() -> bool:
    """True while the user is typing into (or keyboard-searching) a field."""
    return isinstance(QApplication.focusWidget(), _TEXT_INPUTS)


class ReviewShortcuts:
    """The review keys (*bindings*), live only while *is_active()* holds.

    ``sync()`` disables them outright (so the keys stay free) whenever the
    review is unavailable; each key also re-checks on activation and does
    nothing while a text field has focus.
    """

    def __init__(self, window: QWidget, callbacks: dict[str, Callable[[], None]],
                 is_active: Callable[[], bool],
                 bindings: dict[str, str] | None = None) -> None:
        self._is_active = is_active
        self.shortcuts: dict[str, QShortcut] = {}
        for key, action in (REVIEW_BINDINGS if bindings is None else bindings).items():
            if action not in callbacks:
                continue
            sc = QShortcut(QKeySequence(key), window)
            sc.activated.connect(lambda cb=callbacks[action]: self._fire(cb))
            self.shortcuts[action] = sc
        self.sync()

    def sync(self) -> None:
        active = bool(self._is_active())
        for sc in self.shortcuts.values():
            sc.setEnabled(active)

    def _fire(self, callback: Callable[[], None]) -> None:
        if self._is_active() and not text_input_has_focus():
            callback()


def setup_shortcuts(window: QWidget, callbacks: dict) -> list[QShortcut]:
    """Register keyboard shortcuts.

    Parameters
    ----------
    window : QWidget
        The top-level widget that owns the shortcuts.
    callbacks : dict
        Maps action names to callables:
        ``"frame_prev"``, ``"frame_next"``, ``"zoom_in"``,
        ``"zoom_out"``, ``"zoom_reset"``, ``"undo"``, ``"play_pause"``.

    Returns
    -------
    list[QShortcut]
        Created shortcuts (caller should keep a reference to prevent GC).
    """
    bindings = {
        "Left": "frame_prev",
        "Right": "frame_next",
        "+": "zoom_in",
        "=": "zoom_in",  # for keyboards without numpad
        "-": "zoom_out",
        "0": "zoom_reset",
        "Ctrl+Z": "undo",
        "Space": "play_pause",
    }
    shortcuts: list[QShortcut] = []
    for key, action in bindings.items():
        if action in callbacks:
            sc = QShortcut(QKeySequence(key), window)
            sc.activated.connect(callbacks[action])
            shortcuts.append(sc)
    return shortcuts
