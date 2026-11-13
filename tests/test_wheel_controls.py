"""Wheel interactions for the classic tuning controls."""

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QPoint, QPointF, Qt
from PyQt6.QtGui import QWheelEvent
from PyQt6.QtWidgets import QApplication

from bubbletrack.ui.classic_controls import ClassicControls


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
        controls = ClassicControls()
        for control in (controls._threshold, controls._removing_factor):
            before = control.value()
            self.app.sendEvent(control._slider, wheel_up())
            self.assertEqual(control.value(), before + 1)


if __name__ == "__main__":
    unittest.main()
