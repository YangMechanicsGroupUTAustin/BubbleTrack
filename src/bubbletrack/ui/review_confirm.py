"""Accept all's one question: storing Auto's guesses nobody has looked at."""

from __future__ import annotations

from PyQt6.QtWidgets import QMessageBox, QWidget

ACCEPT_ALL_TEXT = "Accept all"
DETAIL = ("Each guess becomes a locked result, like Accept on that frame. "
          "Ctrl+Z undoes the whole Accept all.")


def accept_all_box(parent: QWidget | None, question: str) -> QMessageBox:
    """The question as a message box (not shown): Accept all / Cancel."""
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Question)
    box.setWindowTitle("Accept all")
    box.setText(question)
    box.setInformativeText(DETAIL)
    accept = box.addButton(ACCEPT_ALL_TEXT, QMessageBox.ButtonRole.AcceptRole)
    cancel = box.addButton(QMessageBox.StandardButton.Cancel)
    cancel.setObjectName("secondaryBtn")  # the app stylesheet's quiet button
    box.setDefaultButton(accept)
    return box


def ask_accept_all(parent: QWidget | None, question: str) -> bool:
    """Ask *question*; True if the user chose Accept all."""
    box = accept_all_box(parent, question)
    box.exec()
    clicked = box.clickedButton()
    return clicked is not None and box.buttonRole(clicked) == QMessageBox.ButtonRole.AcceptRole
