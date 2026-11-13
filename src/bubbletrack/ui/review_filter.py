"""'All kinds ▾': show only some kinds of problem in the Review list."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QMenu, QToolButton, QWidget

from bubbletrack.controller.review_queue import ReviewItem
from bubbletrack.controller.review_segments import CATEGORIES, OTHER, category_label, category_of

_WIDEST = "8 of 10 kinds ▾"  # the longest text the button shows
_ORDER = [c.key for c in CATEGORIES] + [OTHER.key]


def kinds_present(items: Iterable[ReviewItem]) -> list[tuple[str, int]]:
    """(category, frames) of every kind among *items*, in menu order (a row
    can hold frames of several kinds: the filter counts frames)."""
    totals: dict[str, int] = {}
    for item in items:
        key = category_of(item)
        totals[key] = totals.get(key, 0) + 1
    return [(k, totals[k]) for k in _ORDER if k in totals]


class KindFilter(QToolButton):
    """A checkable menu of the kinds present; hidden while only one kind is listed."""

    hidden_changed = pyqtSignal(object)  # frozenset of the hidden category keys

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("reviewFilter")
        self.setToolTip("Show only some kinds of problem")
        self.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.menu_ = QMenu(self)
        self.menu_.aboutToShow.connect(self.fill_menu)
        self.setMenu(self.menu_)
        self._present: list[tuple[str, int]] = []
        self._hidden: frozenset[str] = frozenset()
        # One width for every text, so the summary beside it never shifts.
        self.setText(_WIDEST)
        self.setMinimumWidth(self.sizeHint().width())

    def update_kinds(self, present: Sequence[tuple[str, int]], hidden: frozenset[str]) -> None:
        """*present*: (category, frames) of every kind listed, in menu order."""
        self._present, self._hidden = list(present), hidden
        shown = sum(1 for key, _ in self._present if key not in hidden)
        self.setVisible(len(self._present) > 1 or bool(hidden))
        self.setText("All kinds ▾" if shown == len(self._present)
                     else f"{shown} of {len(self._present)} kinds ▾")

    def fill_menu(self) -> None:
        menu = self.menu_
        menu.clear()
        menu.addAction("Show all kinds", lambda: self.hidden_changed.emit(frozenset()))
        menu.addSeparator()
        for key, frames in self._present:
            action = menu.addAction(f"{category_label(key)}  ({frames})")
            action.setCheckable(True)
            action.setChecked(key not in self._hidden)
            action.toggled.connect(lambda on, k=key: self.hidden_changed.emit(
                self._hidden - {k} if on else self._hidden | {k}))
