"""Accept all's question: Cancel is the quiet default, never the bulk accept."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QMessageBox

from bubbletrack.ui.review_confirm import ACCEPT_ALL_TEXT, accept_all_box

QSS = Path(__file__).resolve().parents[1] / "src" / "bubbletrack" / "resources" / "style.qss"


def _buttons(box: QMessageBox):
    by_text = {b.text().replace("&", ""): b for b in box.buttons()}
    return by_text[ACCEPT_ALL_TEXT], by_text["Cancel"]


def test_a_buttons_key_hint_steps_aside_when_the_text_needs_the_room(qapp):
    from bubbletrack.ui.review_card import ActionButton

    button = ActionButton("no_bubble_all")
    button.setText("No bubble on all 3")
    button.resize(400, 28)
    button.show()
    qapp.processEvents()
    assert button.key_shown()
    button.resize(110, 28)  # a narrow dock (1280 x 800)
    qapp.processEvents()
    assert not button.key_shown()
    assert "⇧N" in button.toolTip()  # the key is still told
    button.close()


def test_enter_and_escape_both_cancel(qapp):
    box = accept_all_box(None, "Accept 229 frames?", empty=3)
    accept, cancel = _buttons(box)
    assert box.defaultButton() is cancel and box.escapeButton() is cancel
    assert box.buttonRole(accept) == QMessageBox.ButtonRole.AcceptRole


def test_cancel_looks_secondary_and_accept_all_primary(qapp):
    qapp.setStyleSheet(QSS.read_text(encoding="utf-8"))
    try:
        box = accept_all_box(None, "Accept 229 frames?")
        box.show()
        qapp.processEvents()
        accept, cancel = _buttons(box)
        assert cancel.objectName() == "secondaryBtn"

        def fill(button) -> QColor:
            image = button.grab().toImage()
            return image.pixelColor(image.width() // 2, 3)
        assert fill(accept).name() != fill(cancel).name()
        box.close()
    finally:
        qapp.setStyleSheet("")
