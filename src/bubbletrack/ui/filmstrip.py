"""Filmstrip: a collapsible row of key-frame thumbnails; a click jumps there."""

from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtGui import QIcon, QPixmap
from PyQt6.QtWidgets import (
    QHBoxLayout, QLabel, QPushButton, QScrollArea, QSizePolicy, QToolButton,
    QVBoxLayout, QWidget,
)

THUMB_HEIGHT = 64
THUMB_MAX_WIDTH = 132


@dataclass(frozen=True)
class Thumbnail:
    """One key frame as shown in the strip."""

    frame: int        # 0-based frame index
    pixmap: QPixmap   # image with the fitted outline already drawn
    caption: str      # short text under the image, e.g. "Rmax · 21"
    tooltip: str = ""


class Filmstrip(QWidget):
    """Horizontal strip of key-frame thumbnails.

    Signals
    -------
    frame_clicked(int)
        0-based frame index of the clicked thumbnail.
    expanded_changed(bool)
        The strip was expanded (True) or collapsed (False).
    """

    frame_clicked = pyqtSignal(int)
    expanded_changed = pyqtSignal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("Filmstrip")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        self._toggle = QPushButton()
        self._toggle.setObjectName("filmstripToggle")
        self._toggle.setCheckable(True)
        self._toggle.setChecked(True)
        self._toggle.setToolTip("Show / hide the key frames")
        self._toggle.toggled.connect(self._on_toggled)
        header.addWidget(self._toggle)
        self._hint = QLabel("first fit · Rmax · collapses · last · flagged — click to jump")
        self._hint.setObjectName("dimText")
        header.addWidget(self._hint)
        header.addStretch()
        layout.addLayout(header)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        strip = QWidget()
        strip.setObjectName("FilmstripStrip")
        self._row = QHBoxLayout(strip)
        self._row.setContentsMargins(0, 0, 0, 0)
        self._row.setSpacing(6)
        self._placeholder = QLabel("Key frames appear here once frames are fitted.")
        self._placeholder.setObjectName("dimText")
        self._row.addWidget(self._placeholder)
        self._row.addStretch()
        self._scroll.setWidget(strip)
        self._scroll.setFixedHeight(THUMB_HEIGHT + 34)
        layout.addWidget(self._scroll)

        self._buttons: list[QToolButton] = []
        self._current: int | None = None
        self._update_toggle_text()

    # -- public API ------------------------------------------------------- #

    def set_thumbnails(self, thumbs: list[Thumbnail]) -> None:
        """Replace the strip's content (an empty list shows the placeholder)."""
        for btn in self._buttons:
            self._row.removeWidget(btn)
            btn.deleteLater()
        self._buttons = []
        for i, thumb in enumerate(thumbs):
            btn = self._make_button(thumb)
            self._row.insertWidget(i, btn)
            self._buttons.append(btn)
        self._placeholder.setVisible(not thumbs)
        self._update_toggle_text()
        self.set_current(self._current)

    def set_current(self, frame: int | None) -> None:
        """Mark the thumbnail of the shown frame (if it is a key frame)."""
        self._current = frame
        for btn in self._buttons:
            btn.setChecked(btn.property("frame") == frame)

    def frames(self) -> list[int]:
        """0-based frame indices of the thumbnails, left to right."""
        return [int(btn.property("frame")) for btn in self._buttons]

    def buttons(self) -> list[QToolButton]:
        return list(self._buttons)

    def is_expanded(self) -> bool:
        return self._toggle.isChecked()

    def set_expanded(self, expanded: bool) -> None:
        self._toggle.setChecked(expanded)

    # -- internals -------------------------------------------------------- #

    def _make_button(self, thumb: Thumbnail) -> QToolButton:
        btn = QToolButton()
        btn.setObjectName("thumbBtn")
        btn.setCheckable(True)
        btn.setAutoRaise(True)
        btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        btn.setIcon(QIcon(thumb.pixmap))
        btn.setIconSize(QSize(thumb.pixmap.width(), thumb.pixmap.height()))
        btn.setText(thumb.caption)
        btn.setToolTip(thumb.tooltip or thumb.caption)
        btn.setProperty("frame", thumb.frame)
        btn.clicked.connect(lambda _checked=False, f=thumb.frame: self._on_clicked(f))
        return btn

    def _on_clicked(self, frame: int) -> None:
        self.set_current(frame)
        self.frame_clicked.emit(frame)

    def _on_toggled(self, expanded: bool) -> None:
        self._scroll.setVisible(expanded)
        self._hint.setVisible(expanded)
        self._update_toggle_text()
        self.expanded_changed.emit(expanded)

    def _update_toggle_text(self) -> None:
        arrow = "▾" if self._toggle.isChecked() else "▸"
        count = f" ({len(self._buttons)})" if self._buttons else ""
        self._toggle.setText(f"{arrow} Key frames{count}")
