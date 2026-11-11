"""Main application window — assembles all panels."""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import (
    QHBoxLayout, QMainWindow, QMenu, QMenuBar, QPushButton,
    QVBoxLayout, QWidget, QSizePolicy,
)

from bubbletrack.ui.header_bar import HeaderBar
from bubbletrack.ui.image_compare import CompareMode, CompareToolbar
from bubbletrack.ui.status_bar import StatusBar
from bubbletrack.ui.left_panel import LeftPanel
from bubbletrack.ui.image_panel import ImagePanel
from bubbletrack.ui.radius_chart import RadiusChart
from bubbletrack.ui.frame_scrubber import FrameScrubber


class MainWindow(QMainWindow):
    """Top-level window following the Figma 3-panel design."""

    # Session persistence signals (handled by the controller)
    save_session_requested = pyqtSignal()
    load_session_requested = pyqtSignal()
    compare_mode_changed = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bubble Radius Fitting")
        self.setMinimumSize(1280, 960)

        # Menubar
        self._build_menubar()

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Header
        self.header = HeaderBar()
        root.addWidget(self.header)

        # Main body
        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)

        # Left sidebar
        self.left_panel = LeftPanel()
        body.addWidget(self.left_panel)

        # Sidebar toggle
        self._sidebar_toggle = QPushButton("\u25C0")
        self._sidebar_toggle.setObjectName("sidebarToggle")
        self._sidebar_toggle.setFixedWidth(20)
        self._sidebar_toggle.setSizePolicy(
            QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding,
        )
        self._sidebar_toggle.clicked.connect(self._toggle_sidebar)
        body.addWidget(self._sidebar_toggle)

        # Right content area
        right = QVBoxLayout()
        right.setContentsMargins(12, 12, 12, 8)
        right.setSpacing(8)

        # Compare-mode toolbar (Side by Side | Overlay | Wipe)
        self.compare_toolbar = CompareToolbar()
        self.compare_toolbar.mode_changed.connect(self._on_compare_mode)
        right.addWidget(self.compare_toolbar)

        # Image viewers row
        images_row = QHBoxLayout()
        images_row.setSpacing(8)
        self.original_panel = ImagePanel("Original Image")
        self.binary_panel = ImagePanel("Binary Mask")
        images_row.addWidget(self.original_panel, 1)
        images_row.addWidget(self.binary_panel, 1)
        right.addLayout(images_row, 1)

        # Frame scrubber
        self.frame_scrubber = FrameScrubber()
        right.addWidget(self.frame_scrubber)

        # Radius chart
        self.radius_chart = RadiusChart()
        right.addWidget(self.radius_chart)

        body.addLayout(right, 1)
        root.addLayout(body, 1)

        # Status bar
        self.status_bar = StatusBar()
        root.addWidget(self.status_bar)

    # -- Menubar -------------------------------------------------------- #

    def _build_menubar(self) -> None:
        """Create the application menubar with File > Save/Load Session."""
        menubar: QMenuBar = self.menuBar()
        file_menu: QMenu = menubar.addMenu("&File")

        save_action = QAction("&Save Session...", self)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.triggered.connect(self.save_session_requested.emit)
        file_menu.addAction(save_action)

        load_action = QAction("&Load Session...", self)
        load_action.setShortcut(QKeySequence.StandardKey.Open)
        load_action.triggered.connect(self.load_session_requested.emit)
        file_menu.addAction(load_action)

        self.save_session_action = save_action
        self.load_session_action = load_action

    # -- Close guard ------------------------------------------------------ #

    def set_close_guard(self, guard) -> None:
        """Install a callable returning False to veto closing the window."""
        self._close_guard = guard

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        guard = getattr(self, "_close_guard", None)
        if guard is not None and not guard():
            event.ignore()
            return
        super().closeEvent(event)

    # -- Compare mode helpers --

    @property
    def compare_mode(self) -> CompareMode:
        """Return the currently active image comparison mode."""
        return self.compare_toolbar.current_mode

    def _on_compare_mode(self, mode_value: str) -> None:
        """Handle compare-mode toolbar clicks."""
        mode = CompareMode(mode_value)
        if mode is CompareMode.SIDE_BY_SIDE:
            self.binary_panel.setVisible(True)
        else:
            # Overlay & Wipe show a single combined image
            self.binary_panel.setVisible(False)
        self.compare_mode_changed.emit(mode_value)

    def _toggle_sidebar(self):
        visible = self.left_panel.isVisible()
        self.left_panel.setVisible(not visible)
        self._sidebar_toggle.setText("\u25B6" if visible else "\u25C0")
