"""A word-wrapped label of a fixed number of lines.

Used where the text's length must not change the layout (the Review
dock's feedback line, below the buttons): a longer text is cut, ending in
'…', and shows whole in the tooltip.
"""

from __future__ import annotations

from PyQt6.QtCore import QEvent, QRect, Qt
from PyQt6.QtGui import QFontMetrics
from PyQt6.QtWidgets import QLabel, QSizePolicy, QWidget

ELLIPSIS = "…"
_WRAP = Qt.TextFlag.TextWordWrap.value


def _wrapped_height(metrics: QFontMetrics, width: int, text: str) -> int:
    return metrics.boundingRect(QRect(0, 0, width, 100_000), _WRAP, text).height()


def fit_lines(text: str, metrics: QFontMetrics, width: int, lines: int) -> str:
    """*text* as word-wrapped *lines* lines of *width* px hold it: cut, ending
    in '…', when it is longer."""
    limit = _wrapped_height(metrics, width, "\n".join("X" * lines))
    if _wrapped_height(metrics, width, text) <= limit:
        return text
    lo, hi = 0, len(text)  # the longest start that fits with the ellipsis
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _wrapped_height(metrics, width, text[:mid].rstrip() + ELLIPSIS) <= limit:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo].rstrip() + ELLIPSIS


class LinesLabel(QLabel):
    """A word-wrapped plain text of at most *lines* lines: a longer one ends
    in '…' and shows whole in the tooltip."""

    def __init__(self, lines: int, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._lines, self._full = lines, ""
        self.setTextFormat(Qt.TextFormat.PlainText)
        self.setWordWrap(True)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)

    def set_full_text(self, text: str) -> None:
        self._full = text
        self._fit()

    def full_text(self) -> str:
        return self._full

    def lines_height(self) -> int:
        """The height of *lines* lines: the most it ever needs."""
        self.ensurePolished()
        return _wrapped_height(self.fontMetrics(), 10_000, "\n".join("X" * self._lines))

    def resizeEvent(self, event) -> None:  # noqa: N802  (Qt's name)
        super().resizeEvent(event)
        self._fit()

    def changeEvent(self, event) -> None:  # noqa: N802  (Qt's name)
        super().changeEvent(event)
        if event.type() == QEvent.Type.FontChange:
            self._fit()

    def _fit(self) -> None:
        width = self.contentsRect().width()
        shown = fit_lines(self._full, self.fontMetrics(), width, self._lines) \
            if width > 0 and self._full else self._full
        if shown != self.text():
            self.setText(shown)
        self.setToolTip(self._full if shown != self._full else "")
