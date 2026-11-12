"""Image viewing modes: raw image, or raw image compared with the binary mask."""

from __future__ import annotations

from enum import Enum

import numpy as np
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QPushButton,
    QWidget,
)


class CompareMode(Enum):
    """Supported image viewing modes.

    ``IMAGE`` (the default) shows only the raw frame with the fitted outline;
    the other modes add the binary mask beside, over, or wiped into it.
    """

    IMAGE = "image"
    SIDE_BY_SIDE = "side_by_side"
    OVERLAY = "overlay"
    WIPE = "wipe"


_MODE_LABELS = {
    CompareMode.IMAGE: ("Image", "Raw frame with the fitted outline (mask hidden)"),
    CompareMode.SIDE_BY_SIDE: ("Image + Mask", "Raw frame and binary mask next to each other"),
    CompareMode.OVERLAY: ("Overlay", "Binary mask tinted green over the raw frame"),
    CompareMode.WIPE: ("Wipe", "Left half raw frame, right half binary mask"),
}


class CompareToolbar(QWidget):
    """Segmented control choosing the image viewing mode.

    The active mode is shown as a filled (checked) segment.

    Signals
    -------
    mode_changed(str)
        Emitted with a :class:`CompareMode` *value* string whenever the
        user clicks a different mode button.
    """

    mode_changed = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("CompareToolbar")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._buttons: dict[CompareMode, QPushButton] = {}

        for i, mode in enumerate(CompareMode):
            text, tip = _MODE_LABELS[mode]
            btn = QPushButton(text)
            btn.setObjectName("segmentBtn")
            btn.setCheckable(True)
            btn.setToolTip(tip)
            btn.setProperty("compare_mode", mode.value)
            btn.setProperty("segment", _segment_position(i, len(CompareMode)))
            self._group.addButton(btn)
            self._buttons[mode] = btn
            layout.addWidget(btn)
        self._buttons[CompareMode.IMAGE].setChecked(True)
        layout.addStretch()

        self._group.buttonClicked.connect(self._on_click)

    @property
    def current_mode(self) -> CompareMode:
        """Return the currently selected :class:`CompareMode`."""
        checked = self._group.checkedButton()
        if checked is None:
            return CompareMode.IMAGE
        return CompareMode(checked.property("compare_mode"))

    def set_mode(self, mode: CompareMode) -> None:
        """Select *mode* programmatically and emit :attr:`mode_changed`."""
        btn = self._buttons[mode]
        if not btn.isChecked():
            btn.setChecked(True)
            self.mode_changed.emit(mode.value)

    def _on_click(self, btn: QPushButton) -> None:
        mode_value: str = btn.property("compare_mode")
        self.mode_changed.emit(mode_value)


def _segment_position(index: int, count: int) -> str:
    """'first' / 'middle' / 'last' — lets the stylesheet round the outer ends."""
    if index == 0:
        return "first"
    return "last" if index == count - 1 else "middle"


# ---------------------------------------------------------------------------
# Pure numpy compositing helpers
# ---------------------------------------------------------------------------

def _to_uint8(img: np.ndarray) -> np.ndarray:
    """Normalize an image to uint8 [0, 255]."""
    if img.dtype == bool:
        return img.astype(np.uint8) * 255
    if img.dtype in (np.float32, np.float64):
        if img.max() > 0:
            return (np.clip(img, 0, 1) * 255).astype(np.uint8)
        return np.zeros(img.shape[:2], dtype=np.uint8)
    return img.astype(np.uint8)


def create_overlay(
    original: np.ndarray,
    binary: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """Blend grayscale *original* with coloured *binary* mask.

    Returns an ``(H, W, 3)`` uint8 RGB image where white binary regions
    are tinted green over the original grayscale background.

    Parameters
    ----------
    original:
        2-D grayscale image (any numeric dtype).
    binary:
        2-D binary image (bool or 0/1).
    alpha:
        Blend strength for the green tint (0 = none, 1 = full green).
    """
    orig_u8 = _to_uint8(original)
    rgb = np.stack([orig_u8, orig_u8, orig_u8], axis=-1)

    mask = binary.astype(bool)
    green_channel = rgb[mask, 1].astype(np.int16)
    rgb[mask, 1] = np.clip(
        green_channel + int(alpha * 200), 0, 255,
    ).astype(np.uint8)
    return rgb


def create_wipe(
    original: np.ndarray,
    binary: np.ndarray,
    split_fraction: float = 0.5,
) -> np.ndarray:
    """Create wipe composite: left = *original*, right = *binary*.

    Returns an ``(H, W, 3)`` uint8 RGB image with a red divider line.

    Parameters
    ----------
    original:
        2-D grayscale image.
    binary:
        2-D binary image.
    split_fraction:
        Horizontal split position in [0, 1].
    """
    h, w = original.shape[:2]
    split_x = int(w * split_fraction)

    orig_u8 = _to_uint8(original)
    rgb = np.stack([orig_u8, orig_u8, orig_u8], axis=-1)

    # Right side: show binary as white-on-black
    bin_u8 = _to_uint8(binary)
    right = np.stack([bin_u8[:, split_x:]] * 3, axis=-1)
    rgb[:, split_x:, :] = right

    # Draw a 2-pixel red divider line
    line_start = max(0, split_x - 1)
    line_end = min(w, split_x + 1)
    rgb[:, line_start:line_end, :] = np.array([255, 0, 0], dtype=np.uint8)

    return rgb
