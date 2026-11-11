"""End-to-end: an exception escaping a Qt slot must not kill the app.

Without the installed hook PyQt6 aborts the process (qFatal), so this test
passing at all proves the abort is prevented.
"""

import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication, QMessageBox

from bubbletrack import app as app_module
from bubbletrack.controller.controller import AppController
from bubbletrack.ui.main_window import MainWindow


def _wait_for(predicate, timeout_s: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        QApplication.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_slot_exception_shows_dialog_and_app_survives(tmp_path, monkeypatch):
    qapp = QApplication.instance() or QApplication([])
    monkeypatch.setattr(app_module, "RECOVERY_DIR", tmp_path / "recovery")
    window = MainWindow()
    controller = AppController(window)
    handler = app_module._install_crash_handler(window, controller)
    try:
        def failing_slot():
            raise RuntimeError("slot failure for test")

        QTimer.singleShot(0, failing_slot)

        def dialog_visible():
            return any(
                box.isVisible() and "slot failure for test" in box.text()
                for box in window.findChildren(QMessageBox)
            )

        assert _wait_for(dialog_visible), "error dialog was not shown"

        # The app is still alive and responsive after the error.
        ran = []
        QTimer.singleShot(0, lambda: ran.append(True))
        assert _wait_for(lambda: bool(ran))
    finally:
        handler.uninstall()
        for box in window.findChildren(QMessageBox):
            box.close()
        window.close()
        qapp.processEvents()
