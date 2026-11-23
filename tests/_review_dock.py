"""Helpers for tests of where the Review dock's buttons sit."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QPoint, QRect, Qt

QSS = Path(__file__).resolve().parents[1] / "src" / "bubbletrack" / "resources" / "style.qss"


def settle(qapp) -> None:
    for _ in range(4):
        qapp.processEvents()


def size_dock(qapp, window, rc, width: int, height: int, dock_width: int) -> None:
    """Show *window* at *width* x *height* with the Review dock *dock_width* wide."""
    window.resize(width, height)
    window.show()
    settle(qapp)
    window.resizeDocks([rc.dock], [dock_width], Qt.Orientation.Horizontal)
    settle(qapp)


def button_rects(rc) -> dict[str, QRect]:
    """Where each card button sits in the dock (a hidden one: its kept slot)."""
    card, dock = rc.panel.card, rc.dock
    buttons = card.buttons
    slot = buttons["no_bubble_all"] if buttons["no_bubble_all"].isVisible() \
        else buttons["accept_all"]
    named = {"accept": buttons["accept"], "fix": buttons["fix"],
             "no_bubble": buttons["no_bubble"], "row slot": slot, "more": card.more}
    return {name: QRect(w.mapTo(dock, QPoint(0, 0)), w.size()) for name, w in named.items()}
