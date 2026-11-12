"""Application entry point — creates QApplication, loads QSS, shows window."""

from __future__ import annotations

import multiprocessing
import sys
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from PyQt6.QtCore import QObject, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QIcon, QPalette
from PyQt6.QtWidgets import QApplication, QMessageBox, QWidget

from bubbletrack.controller.controller import AppController
from bubbletrack.controller.session_controller import claim_pending_recovery
from bubbletrack.crash_handler import CrashHandler
from bubbletrack.logging_config import setup_logging
from bubbletrack.model.config import (
    is_onboarding_done,
    record_clean_exit,
    save_config,
    set_onboarding_done,
)
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


# (parent, session path, on_restore) -> None; must not block.
RestorePrompt = Callable[[QWidget | None, Path, Callable[[], None]], None]


def ask_restore_session(parent: QWidget | None, path: Path,
                        on_restore: Callable[[], None]) -> None:
    """Ask "Restore previous session?" without blocking; Restore calls *on_restore*."""
    try:
        saved = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    except OSError:
        saved = "unknown time"
    box = QMessageBox(
        QMessageBox.Icon.Question, "Restore previous session?",
        "BubbleTrack did not close normally last time.\n\n"
        f"Restore the session autosaved at {saved}?\n\n"
        f"Either way it stays available as:\n{path}\n(File > Load Session).",
        parent=parent,
    )
    restore = box.addButton("Restore", QMessageBox.ButtonRole.AcceptRole)
    box.addButton("Discard", QMessageBox.ButtonRole.RejectRole)
    box.setDefaultButton(restore)
    box.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

    def _chosen(button) -> None:
        if button == restore:
            QTimer.singleShot(0, on_restore)  # after the box has closed

    box.buttonClicked.connect(_chosen)
    box.open()


def offer_session_restore(window: QWidget | None, session_ctrl, *,
                          prompt: RestorePrompt = ask_restore_session) -> Path | None:
    """Offer the autosave of a session that did not exit cleanly; return its path.

    Returns None (and asks nothing) when there is nothing to recover.
    """
    path = claim_pending_recovery()
    if path is None:
        return None
    prompt(window, path, lambda: session_ctrl.load(str(path)))
    return path


def on_about_to_quit(state) -> None:
    """Persist the parameters, then record the clean exit (after the close autosave)."""
    save_config(state)
    record_clean_exit()


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

    # Save parameters and record a clean exit; the window's close guard has
    # already written the session autosave, so it is older than this stamp.
    app.aboutToQuit.connect(lambda: on_about_to_quit(controller._get_state()))

    # Show onboarding dialog on first launch
    if not is_onboarding_done():
        dlg = WelcomeDialog(window)
        dlg.exec()
        if dlg.dont_show_again:
            set_onboarding_done()

    window.show()
    offer_session_restore(window, controller.session_ctrl)
    sys.exit(app.exec())


if __name__ == "__main__":
    # Must be called before main() so PyInstaller child processes
    # exit immediately instead of re-executing the GUI.
    multiprocessing.freeze_support()
    main()
