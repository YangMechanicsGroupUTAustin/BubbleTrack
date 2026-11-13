"""Accept all's one question: storing Auto's guesses nobody has looked at."""

from __future__ import annotations

from PyQt6.QtWidgets import QMessageBox, QPushButton, QWidget

ACCEPT_ALL_TEXT = "Accept all"
DETAIL = ("Each guess becomes a locked result, like Accept on that frame. "
          "Ctrl+Z undoes the whole Accept all.")


def accept_all_detail(empty: int) -> str:
    """The question's small print; *empty*: frames with no result and no guess."""
    if not empty:
        return DETAIL
    frames = "1 empty frame stays" if empty == 1 else f"{empty} empty frames stay"
    return f"{DETAIL} {frames} for you to fix or mark."


def accept_all_box(parent: QWidget | None, question: str, empty: int = 0) -> QMessageBox:
    """The question as a message box (not shown): Accept all / Cancel.

    Cancel is the default (Enter) and the escape: storing unseen guesses
    takes a deliberate click.  It is created with the stylesheet's quiet
    look (a name set after polishing would not restyle it).
    """
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Question)
    box.setWindowTitle("Accept all")
    box.setText(question)
    box.setInformativeText(accept_all_detail(empty))
    box.addButton(ACCEPT_ALL_TEXT, QMessageBox.ButtonRole.AcceptRole)
    cancel = QPushButton("Cancel")
    cancel.setObjectName("secondaryBtn")  # the app stylesheet's quiet button
    box.addButton(cancel, QMessageBox.ButtonRole.RejectRole)
    box.setDefaultButton(cancel)
    box.setEscapeButton(cancel)
    return box


def ask_accept_all(parent: QWidget | None, question: str, *, empty: int = 0) -> bool:
    """Ask *question*; True if the user chose Accept all."""
    box = accept_all_box(parent, question, empty)
    box.exec()
    clicked = box.clickedButton()
    return clicked is not None and box.buttonRole(clicked) == QMessageBox.ButtonRole.AcceptRole
