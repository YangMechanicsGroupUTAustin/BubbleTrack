"""Former Manual tab: fixing frames moved into the Review dock (Fix, F).

Kept only as a placeholder page while the left panel still builds a Manual
tab: it says where fixing went and keeps the signals and setters that older
wiring connects to (none of them is emitted).  Nothing of the fix tools
depends on it; delete this module together with the left panel's
``manual_tab`` attribute.
"""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QVBoxLayout, QWidget

from bubbletrack.ui.widgets import InfoBox

MOVED_TEXT = ("Fixing a frame now happens in the Review panel: select the frame "
              "and press Fix (F) to drag a circle or click edge points.")


class ManualTab(QWidget):
    """Placeholder: points to the Review dock's Fix."""

    select_clicked = pyqtSignal()
    done_clicked = pyqtSignal()
    clear_clicked = pyqtSignal()
    frame_selected = pyqtSignal(int)
    tool_changed = pyqtSignal(str)
    use_circle_clicked = pyqtSignal()
    cancel_circle_clicked = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(InfoBox(
            MOVED_TEXT,
            colour="rgba(99, 102, 241, 0.15)", border="rgba(99, 102, 241, 0.20)",
        ))
        layout.addStretch()

    def reset(self) -> None:
        """Nothing to reset (kept for older callers)."""

    def set_frame_range(self, total: int) -> None:
        """No frame control any more (kept for older callers)."""

    def set_frame_value(self, idx: int) -> None:
        """No frame control any more (kept for older callers)."""
