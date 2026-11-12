"""Left sidebar panel — assembles image source, tabs, and post-processing."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QScrollArea, QSizePolicy, QStackedWidget, QVBoxLayout, QWidget,
)

from bubbletrack.ui.image_source import ImageSource
from bubbletrack.ui.tab_bar import PRETUNE, TabBar
from bubbletrack.ui.pretune_tab import PreTuneTab
from bubbletrack.ui.manual_tab import ManualTab
from bubbletrack.ui.automatic_tab import AutomaticTab
from bubbletrack.ui.post_processing import PostProcessing

PANEL_WIDTH = 340


class LeftPanel(QWidget):
    """Fixed-width collapsible sidebar.

    Everything scrolls except the Pre-tune action buttons, which are pinned
    below the scroll area (visible while the Pre-tune tab is active).
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("LeftPanel")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setFixedWidth(PANEL_WIDTH)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Scrollable content
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        content = QWidget()
        content.setObjectName("LeftPanelContent")
        content.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # Sections
        self.image_source = ImageSource()
        layout.addWidget(self.image_source)

        self.tab_bar = TabBar()
        layout.addWidget(self.tab_bar)

        # Stacked tabs, in TabBar.TAB_NAMES order
        self._stack = QStackedWidget()
        self.pretune_tab = PreTuneTab()
        self.manual_tab = ManualTab()
        self.automatic_tab = AutomaticTab()
        pages = {"Pre-tune": self.pretune_tab, "Automatic": self.automatic_tab,
                 "Manual": self.manual_tab}
        for name in TabBar.TAB_NAMES:
            self._stack.addWidget(pages[name])
        layout.addWidget(self._stack)

        # Post processing
        self.post_processing = PostProcessing()
        layout.addWidget(self.post_processing)

        layout.addStretch()
        self.scroll_area.setWidget(content)
        outer.addWidget(self.scroll_area, 1)

        # Pinned footer: Fit Current Frame / Auto Tune never scroll away
        self.pinned_footer = QWidget()
        self.pinned_footer.setObjectName("PinnedFooter")
        self.pinned_footer.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        footer_layout = QVBoxLayout(self.pinned_footer)
        footer_layout.setContentsMargins(12, 8, 12, 10)
        footer_layout.addWidget(self.pretune_tab.action_bar)
        outer.addWidget(self.pinned_footer)

        self.tab_bar.tab_changed.connect(self._on_tab_changed)
        self._on_tab_changed(self.tab_bar.current_index())

    def _on_tab_changed(self, idx: int) -> None:
        self._stack.setCurrentIndex(idx)
        # Size the stack to the visible page only, so a short tab does not
        # leave the tallest tab's height as empty space above Post Processing.
        for i in range(self._stack.count()):
            vertical = QSizePolicy.Policy.Preferred if i == idx else QSizePolicy.Policy.Ignored
            self._stack.widget(i).setSizePolicy(QSizePolicy.Policy.Preferred, vertical)
        self._stack.updateGeometry()
        self.pinned_footer.setVisible(TabBar.name_of(idx) == PRETUNE)
