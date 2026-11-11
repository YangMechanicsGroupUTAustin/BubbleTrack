"""Application entry point — creates QApplication, loads QSS, shows window."""

from __future__ import annotations

import multiprocessing
import os
import sys
from pathlib import Path

from PyQt6.QtCore import QObject, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QIcon, QPalette
from PyQt6.QtWidgets import QApplication, QMessageBox, QWidget

from bubbletrack.controller.controller import AppController
from bubbletrack.crash_handler import CrashHandler
from bubbletrack.logging_config import setup_logging
from bubbletrack.model.config import is_onboarding_done, save_config, set_onboarding_done
from bubbletrack.ui.main_window import MainWindow
from bubbletrack.ui.welcome_dialog import WelcomeDialog

APP_DIR = Path.home() / ".bubbletrack"
LOG_DIR = APP_DIR / "logs"
RECOVERY_DIR = APP_DIR / "recovery"


class _CrashNotifier(QObject):
    """Shows uncaught-error messages on the GUI thread.

    ``notify`` may be called from any thread; the signal is queued to the
    thread this object lives in (the GUI thread).
    """

    crashed = pyqtSignal(str, str)

    def __init__(self, parent_widget: QWidget, log_path: Path) -> None:
        super().__init__(parent_widget)  # owned by the window, lives on the GUI thread
        self._parent = parent_widget
        self._log_path = log_path
        self._box: QMessageBox | None = None
        self.crashed.connect(self._show)

    def notify(self, summary: str, recovery_path: str | None) -> None:
        self.crashed.emit(summary, recovery_path or "")

    def _show(self, summary: str, recovery_path: str) -> None:
        if self._box is not None and self._box.isVisible():
            return  # one dialog at a time; every error is in the log
        lines = [
            "An unexpected error occurred. BubbleTrack is still running,",
            "but please save or export your results and restart it.",
            "",
            summary,
            "",
        ]
        if recovery_path:
            lines.append(f"Your results were saved to:\n{recovery_path}\n"
                         "(File > Load Session to reopen them.)")
        lines.append(f"Details: {self._log_path}")
        self._box = QMessageBox(
            QMessageBox.Icon.Warning, "Unexpected error", "\n".join(lines),
            parent=self._parent,
        )
        self._box.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self._box.finished.connect(self._forget_box)
        self._box.open()

    def _forget_box(self) -> None:
        self._box = None


def _install_crash_handler(window: MainWindow, controller: AppController) -> CrashHandler:
    notifier = _CrashNotifier(window, LOG_DIR / "bubbletrack.log")
    handler = CrashHandler(
        state_provider=controller._get_state,
        notify=notifier.notify,
        recovery_dir=RECOVERY_DIR,
    )
    handler.install()
    return handler


def main():
    setup_logging(log_dir=LOG_DIR)

    app = QApplication(sys.argv)

    # Force Fusion style for consistent dark-theme rendering across platforms
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor("#0f1117"))
    palette.setColor(QPalette.ColorRole.WindowText, QColor("#e4e4e7"))
    palette.setColor(QPalette.ColorRole.Base, QColor("#1a1b23"))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor("#13141b"))
    palette.setColor(QPalette.ColorRole.Text, QColor("#e4e4e7"))
    palette.setColor(QPalette.ColorRole.Button, QColor("#1a1b23"))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor("#a1a1aa"))
    palette.setColor(QPalette.ColorRole.Highlight, QColor("#6366f1"))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#e4e4e7"))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor("#1a1b23"))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor("#e4e4e7"))
    app.setPalette(palette)

    # Resolve resources path (works both normally and when frozen by PyInstaller)
    if getattr(sys, 'frozen', False):
        res = Path(sys._MEIPASS) / "bubbletrack" / "resources"
    else:
        res = Path(__file__).parent / "resources"
    qss_path = res / "style.qss"
    if qss_path.exists():
        app.setStyleSheet(qss_path.read_text(encoding="utf-8"))

    # Application icon
    icon_path = res / "icon.ico"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))

    window = MainWindow()
    controller = AppController(window)
    _install_crash_handler(window, controller)

    # Auto-save user parameters on application exit
    app.aboutToQuit.connect(lambda: save_config(controller._get_state()))

    # Show onboarding dialog on first launch
    if not is_onboarding_done():
        dlg = WelcomeDialog(window)
        dlg.exec()
        if dlg.dont_show_again:
            set_onboarding_done()

    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    # Must be called before main() so PyInstaller child processes
    # exit immediately instead of re-executing the GUI.
    multiprocessing.freeze_support()
    main()
