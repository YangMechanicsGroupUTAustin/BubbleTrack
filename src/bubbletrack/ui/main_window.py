"""Main application window — assembles all panels."""

from __future__ import annotations

from PyQt6.QtCore import QSettings, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QAction, QGuiApplication, QKeySequence
from PyQt6.QtWidgets import (
    QHBoxLayout, QMainWindow, QMenu, QMenuBar, QPushButton, QSizePolicy,
    QSplitter, QVBoxLayout, QWidget,
)

from bubbletrack.ui.frame_scrubber import FrameScrubber
from bubbletrack.ui.header_bar import HeaderBar
from bubbletrack.ui.image_compare import CompareMode, CompareToolbar
from bubbletrack.ui.image_panel import ImagePanel
from bubbletrack.ui.left_panel import LeftPanel
from bubbletrack.ui.radius_chart import RadiusChart
from bubbletrack.ui.status_bar import StatusBar

# Fits 1280x800 screens; the layout itself needs less (left panel scrolls).
MIN_WINDOW_SIZE = (1024, 680)
DEFAULT_WINDOW_SIZE = (1600, 1000)
WINDOW_FRAME_MARGIN = 48  # title bar / dock allowance when sizing to the screen
DEFAULT_SPLIT = (0.62, 0.38)  # images : R(t) chart
DEFAULT_SPLIT_WITH_STRIP = (0.52, 0.48)  # the key-frame strip shares the chart pane
SPLITTER_SETTINGS_KEY = "main_window/content_splitter_sizes"
SPLITTER_SAVE_DELAY_MS = 400

_PANEL_TITLES = {
    CompareMode.IMAGE: "Image",
    CompareMode.SIDE_BY_SIDE: "Image",
    CompareMode.OVERLAY: "Image + mask overlay",
    CompareMode.WIPE: "Image | mask wipe",
}


def layout_settings() -> QSettings:
    """Where window layout (splitter sizes) is remembered."""
    return QSettings("BubbleTrack", "BubbleTrack")


class MainWindow(QMainWindow):
    """Top-level window: sidebar | (images / R(t) chart in a splitter)."""

    # Session persistence signals (handled by the controller)
    save_session_requested = pyqtSignal()
    load_session_requested = pyqtSignal()
    compare_mode_changed = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bubble Radius Fitting")
        self.setMinimumSize(*MIN_WINDOW_SIZE)
        self._size_to_screen()

        self._build_menubar()

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.header = HeaderBar()
        root.addWidget(self.header)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)

        self.left_panel = LeftPanel()
        body.addWidget(self.left_panel)

        self._sidebar_toggle = QPushButton("◀")
        self._sidebar_toggle.setObjectName("sidebarToggle")
        self._sidebar_toggle.setToolTip("Hide / show the side panel")
        self._sidebar_toggle.setFixedWidth(20)
        self._sidebar_toggle.setSizePolicy(
            QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding,
        )
        self._sidebar_toggle.clicked.connect(self._toggle_sidebar)
        body.addWidget(self._sidebar_toggle)

        body.addLayout(self._build_content(), 1)
        root.addLayout(body, 1)

        self.status_bar = StatusBar()
        root.addWidget(self.status_bar)

        self._splitter_restored = False
        self._splitter_save_timer = QTimer(self)
        self._splitter_save_timer.setSingleShot(True)
        self._splitter_save_timer.setInterval(SPLITTER_SAVE_DELAY_MS)
        self._splitter_save_timer.timeout.connect(self.save_layout)
        self.content_splitter.splitterMoved.connect(
            lambda _pos, _idx: self._splitter_save_timer.start(),
        )
        self._on_compare_mode(self.compare_mode.value, emit=False)

    def _size_to_screen(self) -> None:
        """Open at a comfortable size that fits the available screen."""
        width, height = DEFAULT_WINDOW_SIZE
        screen = QGuiApplication.primaryScreen()
        if screen is not None:
            avail = screen.availableGeometry()
            width = min(width, avail.width())
            height = min(height, avail.height() - WINDOW_FRAME_MARGIN)
        self.resize(max(width, MIN_WINDOW_SIZE[0]), max(height, MIN_WINDOW_SIZE[1]))

    def _build_content(self) -> QVBoxLayout:
        """Right side: view-mode toolbar, then images / chart in a splitter."""
        right = QVBoxLayout()
        right.setContentsMargins(12, 8, 12, 6)
        right.setSpacing(6)

        self.compare_toolbar = CompareToolbar()
        self.compare_toolbar.mode_changed.connect(self._on_compare_mode)
        right.addWidget(self.compare_toolbar)

        # Top pane: image viewers + frame scrubber
        image_area = QWidget()
        image_layout = QVBoxLayout(image_area)
        image_layout.setContentsMargins(0, 0, 0, 0)
        image_layout.setSpacing(4)
        images_row = QHBoxLayout()
        images_row.setSpacing(8)
        self.original_panel = ImagePanel("Image")
        self.binary_panel = ImagePanel("Binary Mask")
        images_row.addWidget(self.original_panel, 1)
        images_row.addWidget(self.binary_panel, 1)
        image_layout.addLayout(images_row, 1)
        self.frame_scrubber = FrameScrubber()
        image_layout.addWidget(self.frame_scrubber)

        # Bottom pane: optional key-frame strip + R(t) chart
        chart_area = QWidget()
        self._chart_layout = QVBoxLayout(chart_area)
        self._chart_layout.setContentsMargins(0, 0, 0, 0)
        self._chart_layout.setSpacing(4)
        self.radius_chart = RadiusChart()
        self._chart_layout.addWidget(self.radius_chart, 1)
        self.filmstrip = None

        self.content_splitter = QSplitter(Qt.Orientation.Vertical)
        self.content_splitter.setObjectName("contentSplitter")
        self.content_splitter.setChildrenCollapsible(False)
        self.content_splitter.setHandleWidth(8)
        self.content_splitter.addWidget(image_area)
        self.content_splitter.addWidget(chart_area)
        self.content_splitter.setStretchFactor(0, 3)
        self.content_splitter.setStretchFactor(1, 2)
        right.addWidget(self.content_splitter, 1)
        return right

    # -- Menubar -------------------------------------------------------- #

    def _build_menubar(self) -> None:
        """File > Save/Load Session; View > layout helpers."""
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

        view_menu: QMenu = menubar.addMenu("&View")
        self.reset_zoom_action = QAction("Reset Image Zoom\t0", self)
        self.reset_zoom_action.triggered.connect(self.reset_image_zoom)
        view_menu.addAction(self.reset_zoom_action)
        self.reset_layout_action = QAction("Reset Panel Sizes", self)
        self.reset_layout_action.triggered.connect(self.reset_layout)
        view_menu.addAction(self.reset_layout_action)

    # -- Layout --------------------------------------------------------- #

    def set_filmstrip(self, widget: QWidget) -> None:
        """Place the key-frame strip above the R(t) chart (one strip at most)."""
        if self.filmstrip is not None:
            self._chart_layout.removeWidget(self.filmstrip)
            self.filmstrip.deleteLater()
        self.filmstrip = widget
        self._chart_layout.insertWidget(0, widget)

    def reset_image_zoom(self) -> None:
        self.original_panel.reset_zoom()
        self.binary_panel.reset_zoom()

    def reset_layout(self) -> None:
        """Give images and chart their default share of the height."""
        total = sum(self.content_splitter.sizes()) or self.content_splitter.height()
        split = DEFAULT_SPLIT if self.filmstrip is None else DEFAULT_SPLIT_WITH_STRIP
        self.content_splitter.setSizes([int(total * f) for f in split])

    def save_layout(self) -> None:
        settings = layout_settings()
        settings.setValue(SPLITTER_SETTINGS_KEY, [int(s) for s in self.content_splitter.sizes()])
        settings.sync()

    def restore_layout(self) -> bool:
        """Apply remembered splitter sizes; False when none are stored."""
        stored = layout_settings().value(SPLITTER_SETTINGS_KEY)
        try:
            sizes = [int(s) for s in stored] if isinstance(stored, (list, tuple)) else []
        except (TypeError, ValueError):
            sizes = []
        if len(sizes) != self.content_splitter.count() or min(sizes, default=0) <= 0:
            return False
        self.content_splitter.setSizes(sizes)
        return True

    def showEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().showEvent(event)
        if not self._splitter_restored:
            self._splitter_restored = True
            if not self.restore_layout():
                self.reset_layout()

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

    def _on_compare_mode(self, mode_value: str, emit: bool = True) -> None:
        """Show the mask panel only side by side; other modes use one image."""
        mode = CompareMode(mode_value)
        self.binary_panel.setVisible(mode is CompareMode.SIDE_BY_SIDE)
        self.original_panel.set_title(_PANEL_TITLES[mode])
        if emit:
            self.compare_mode_changed.emit(mode_value)

    def _toggle_sidebar(self):
        visible = self.left_panel.isVisible()
        self.left_panel.setVisible(not visible)
        self._sidebar_toggle.setText("▶" if visible else "◀")
