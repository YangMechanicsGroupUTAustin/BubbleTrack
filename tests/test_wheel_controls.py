"""Wheel interactions for tuning and post-processing controls."""

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QPoint, QPointF, Qt
from PyQt6.QtGui import QWheelEvent
from PyQt6.QtWidgets import QApplication

from bubbletrack.ui.post_processing import PostProcessing
from bubbletrack.ui.pretune_tab import PreTuneTab


def wheel_up():
    return QWheelEvent(
        QPointF(10, 10), QPointF(10, 10), QPoint(), QPoint(0, 120),
        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase, False,
    )


class WheelControlsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_threshold_and_remove_change_by_one(self):
        tab = PreTuneTab()
        for control in (tab._threshold, tab._removing_factor):
            before = control.value()
            self.app.sendEvent(control._slider, wheel_up())
            self.assertEqual(control.value(), before + 1)

    def test_post_processing_ignores_wheel(self):
        panel = PostProcessing()
        for control in panel._no_wheel_controls:
            before = (control.currentIndex() if hasattr(control, "currentIndex")
                      else control.value())
            self.app.sendEvent(control, wheel_up())
            after = (control.currentIndex() if hasattr(control, "currentIndex")
                     else control.value())
            self.assertEqual(after, before)


if __name__ == "__main__":
    unittest.main()
