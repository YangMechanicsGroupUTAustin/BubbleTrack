"""Install the Review dock: panel and fix strip, View menu toggle, chart hook."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QDockWidget, QMainWindow, QMenu, QVBoxLayout, QWidget

from bubbletrack.ui.fix_strip import FixStrip
from bubbletrack.ui.review_panel import ReviewPanel


def _view_menu(window: QMainWindow) -> QMenu:
    menubar = window.menuBar()
    for action in menubar.actions():
        menu = action.menu()
        if menu is not None and action.text().replace("&", "") == "View":
            return menu
    return menubar.addMenu("&View")


def install_review(window: QMainWindow, app_controller):
    """Add the Review dock (and a View menu toggle) to *window*; returns its
    ``ReviewController``.

    Idempotent: the app controller installs it at start-up, and a second
    call returns that instance instead of adding another dock and key set.
    """
    from bubbletrack.controller.review_controller import ReviewController

    existing = getattr(app_controller, "review", None)
    if isinstance(existing, ReviewController):
        return existing
    panel, strip = ReviewPanel(), FixStrip()
    body = QWidget()
    body.setObjectName("ReviewBody")
    layout = QVBoxLayout(body)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)
    layout.addWidget(panel, 1)
    layout.addWidget(strip)  # outside the panel: usable while a re-track runs
    dock = QDockWidget("Review", window)
    dock.setObjectName("ReviewDock")
    # Not floatable: the review keys are window shortcuts of the main window.
    dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetClosable
                     | QDockWidget.DockWidgetFeature.DockWidgetMovable)
    dock.setWidget(body)
    window.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
    toggle = dock.toggleViewAction()
    toggle.setText("&Review Queue")
    _view_menu(window).addAction(toggle)
    review = ReviewController(window, app_controller, panel, dock, strip)
    # The chart's "Fix frame N..." (a point's right-click menu) opens the same fix.
    window.radius_chart.fix_requested.connect(review.start_fix)
    return review
