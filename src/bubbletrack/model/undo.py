"""Generic undo/redo stack."""

from __future__ import annotations

from typing import Any


class UndoStack:
    """Generic undo/redo stack with configurable max size.

    Each ``push()`` stores a snapshot.  ``undo()`` pops the most recent
    snapshot and moves it to the redo list.  ``redo()`` reverses the last
    undo.  A new ``push()`` always clears the redo history.
    """

    def __init__(self, max_size: int = 100) -> None:
        self._undo: list[Any] = []
        self._redo: list[Any] = []
        self._max = max_size

    def push(self, state: Any) -> None:
        """Save *state* and clear redo history."""
        self._undo.append(state)
        if len(self._undo) > self._max:
            self._undo.pop(0)
        self._redo.clear()

    def undo(self) -> Any | None:
        """Pop and return the most recent state, or ``None`` if empty."""
        if not self._undo:
            return None
        item = self._undo.pop()
        self._redo.append(item)
        return item

    def redo(self) -> Any | None:
        """Re-apply the last undone state, or ``None`` if empty."""
        if not self._redo:
            return None
        item = self._redo.pop()
        self._undo.append(item)
        return item

    def peek(self) -> Any | None:
        """The state ``undo()`` would return next, without popping it."""
        return self._undo[-1] if self._undo else None

    def can_undo(self) -> bool:
        return len(self._undo) > 0

    def can_redo(self) -> bool:
        return len(self._redo) > 0

    def clear(self) -> None:
        """Discard all undo and redo history."""
        self._undo.clear()
        self._redo.clear()
