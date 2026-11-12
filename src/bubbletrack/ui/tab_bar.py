"""Three-tab selector in workflow order: Pre-tune / Automatic / Manual."""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QHBoxLayout, QPushButton, QWidget

PRETUNE, AUTOMATIC, MANUAL = "Pre-tune", "Automatic", "Manual"


class TabBar(QWidget):
    """Horizontal tab bar that emits the selected tab index.

    Indices follow :attr:`TAB_NAMES` (the workflow order: tune parameters,
    fit every frame, then fix individual frames by hand).  Use
    :meth:`index_of` instead of hard-coded numbers.
    """

    tab_changed = pyqtSignal(int)

    TAB_NAMES = (PRETUNE, AUTOMATIC, MANUAL)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("TabBar")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._buttons: list[QPushButton] = []
        for i, name in enumerate(self.TAB_NAMES):
            btn = QPushButton(name)
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked, idx=i: self._select(idx))
            layout.addWidget(btn)
            self._buttons.append(btn)

        self._select(0)

    @classmethod
    def index_of(cls, name: str) -> int:
        """Index of the tab called *name* (e.g. ``TabBar.index_of("Manual")``)."""
        return cls.TAB_NAMES.index(name)

    @classmethod
    def name_of(cls, idx: int) -> str:
        return cls.TAB_NAMES[idx] if 0 <= idx < len(cls.TAB_NAMES) else ""

    def select(self, name: str) -> None:
        """Switch to the tab called *name*, emitting :attr:`tab_changed`."""
        self._select(self.index_of(name))

    def _select(self, idx: int):
        for i, btn in enumerate(self._buttons):
            btn.setChecked(i == idx)
            btn.setProperty("active", i == idx)
            btn.style().unpolish(btn)
            btn.style().polish(btn)
        self.tab_changed.emit(idx)

    def current_index(self) -> int:
        for i, btn in enumerate(self._buttons):
            if btn.isChecked():
                return i
        return 0

    def current_name(self) -> str:
        return self.name_of(self.current_index())
