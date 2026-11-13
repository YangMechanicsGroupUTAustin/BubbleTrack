"""Side panel, top to bottom: image source, the Auto settings, Fit, Export.

No tabs: Auto measures every frame with one Fit; frames are fixed through
the Review panel; Export opens a dialog.  Classic methods appear only when
Settings asks for them (see MeasurePanel).
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QGroupBox, QPushButton, QScrollArea, QVBoxLayout, QWidget,
)

from bubbletrack.ui.fit_controls import FitControls
from bubbletrack.ui.image_source import ImageSource
from bubbletrack.ui.manual_tab import ManualTab
from bubbletrack.ui.measure_panel import MeasurePanel
from bubbletrack.ui.tab_bar import MANUAL, PRETUNE, TabBar

PANEL_WIDTH = 340


class LeftPanel(QWidget):
    """Fixed-width collapsible sidebar.

    The content scrolls; Fit and Export are pinned below it and never scroll
    away.  ``tab_bar`` and ``manual_tab`` remain for manual fixing opened from
    the Review panel: while the mode is Manual the fix tools replace the
    Auto settings (Back returns).
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
        self.tab_bar = TabBar(self)  # not shown: the measuring / manual-fix mode
        self.manual_tab = ManualTab()
        layout.addWidget(self._build_manual_section())
        layout.addStretch()
        self.scroll_area.setWidget(content)
        outer.addWidget(self.scroll_area, 1)
        outer.addWidget(self._build_footer())

        self.tab_bar.tab_changed.connect(self._on_mode_changed)
        self._on_mode_changed(self.tab_bar.current_index())

    @property
    def pretune_tab(self) -> MeasurePanel:
        """Former name of :attr:`measure_panel` (kept for older callers)."""
        return self.measure_panel

    def _build_manual_section(self) -> QGroupBox:
        self.manual_section = QGroupBox("Fix frame by hand")
        layout = QVBoxLayout(self.manual_section)
        layout.addWidget(self.manual_tab)
        self.manual_back_button = QPushButton("Back")
        self.manual_back_button.setObjectName("secondaryBtn")
        self.manual_back_button.setToolTip("Close the fix tools and return to the Auto settings")
        self.manual_back_button.clicked.connect(lambda: self.tab_bar.select(PRETUNE))
        layout.addWidget(self.manual_back_button)
        return self.manual_section

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

    def _on_mode_changed(self, idx: int) -> None:
        manual = TabBar.name_of(idx) == MANUAL
        self.manual_section.setVisible(manual)
        self.measure_panel.setVisible(not manual)
