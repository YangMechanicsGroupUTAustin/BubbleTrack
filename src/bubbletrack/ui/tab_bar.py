"""Side-panel mode: measuring (normal) or fixing a frame by hand.

The side panel has no tabs any more.  This non-visual switch keeps the old
TabBar interface (``select`` / ``current_name`` / ``tab_changed``) for the
code that still opens manual fixing by name, e.g. the Review panel's Manual
and Circle actions (``select(MANUAL)``); the side panel then shows the fix
tools until the user goes back.

``PRETUNE`` is the historic id of the measuring mode; users see
:func:`mode_label` (status bar "Mode: Measure").
"""

from __future__ import annotations

from PyQt6.QtCore import QObject, pyqtSignal

PRETUNE, MANUAL = "Pre-tune", "Manual"
_LABELS = {PRETUNE: "Measure", MANUAL: "Manual"}


def mode_label(name: str) -> str:
    """What the status bar calls the mode *name*."""
    return _LABELS.get(name, name)


class TabBar(QObject):
    """Which side-panel mode is active; emits the index of the new mode."""

    tab_changed = pyqtSignal(int)

    TAB_NAMES = (PRETUNE, MANUAL)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._index = 0

    @classmethod
    def index_of(cls, name: str) -> int:
        return cls.TAB_NAMES.index(name)

    @classmethod
    def name_of(cls, idx: int) -> str:
        return cls.TAB_NAMES[idx] if 0 <= idx < len(cls.TAB_NAMES) else ""

    def select(self, name: str) -> None:
        """Switch to the mode called *name*, emitting :attr:`tab_changed`."""
        self._index = self.index_of(name)
        self.tab_changed.emit(self._index)

    def current_index(self) -> int:
        return self._index

    def current_name(self) -> str:
        return self.name_of(self._index)
