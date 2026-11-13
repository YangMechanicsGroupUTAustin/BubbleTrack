"""Side panel, top to bottom: image source, the Auto settings, Fit, Export.

No tabs: Auto measures every frame with one Fit; frames are fixed through
the Review panel; Export opens a dialog.  Classic methods appear only when
Settings asks for them (see MeasurePanel).
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QPushButton, QScrollArea, QVBoxLayout, QWidget

from bubbletrack.ui.fit_controls import FitControls
from bubbletrack.ui.image_source import ImageSource
from bubbletrack.ui.measure_panel import MeasurePanel

PANEL_WIDTH = 340


class LeftPanel(QWidget):
    """Fixed-width collapsible sidebar.

    The content scrolls; Fit and Export are pinned below it and never scroll
    away.  Fixing a frame happens in the Review dock.
    """

    export_clicked = pyqtSignal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("LeftPanel")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setFixedWidth(PANEL_WIDTH)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        content = QWidget()
        content.setObjectName("LeftPanelContent")
        content.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        self.image_source = ImageSource()
        layout.addWidget(self.image_source)
        self.measure_panel = MeasurePanel()
        layout.addWidget(self.measure_panel)
        layout.addStretch()
        self.scroll_area.setWidget(content)
        outer.addWidget(self.scroll_area, 1)
        outer.addWidget(self._build_footer())

    @property
    def pretune_tab(self) -> MeasurePanel:
        """Former name of :attr:`measure_panel` (kept for older callers)."""
        return self.measure_panel

    def _build_footer(self) -> QWidget:
        """Fit (with progress / Stop while running) and Export, always in view."""
        self.pinned_footer = QWidget()
        self.pinned_footer.setObjectName("PinnedFooter")
        self.pinned_footer.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        footer = QVBoxLayout(self.pinned_footer)
        footer.setContentsMargins(12, 8, 12, 10)
        footer.setSpacing(8)
        self.fit_controls = FitControls()
        footer.addWidget(self.fit_controls)
        self.export_button = QPushButton("Export…")
        self.export_button.setObjectName("secondaryBtn")
        self.export_button.setToolTip("Save R(t) as MAT, CSV or Excel (pixels or physical units)")
        self.export_button.clicked.connect(self.export_clicked)
        footer.addWidget(self.export_button)
        return self.pinned_footer
