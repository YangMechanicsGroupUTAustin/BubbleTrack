"""Dark bottom status bar: frame, format, scale, ROI; background work on the right."""

from __future__ import annotations

from collections.abc import Sequence

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QWidget

from bubbletrack.ui.frame_scrubber import format_frame_text, label_for

MEASURE_MODE = "Measure"  # the mode outside fitting and fixing (no longer shown)
SCALE_NOT_SET = "Scale: not set"


def scale_text(um2px: float | None) -> str:
    """'Scale: 3.200 µm/px' (as the Export dialog shows it), or 'Scale: not set'."""
    return SCALE_NOT_SET if um2px is None else f"Scale: {um2px:.3f} µm/px"


class StatusBar(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("StatusBar")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 0, 16, 0)
        layout.setSpacing(24)

        self._frame_labels: tuple[str, ...] = ()
        self._frame_label = QLabel("Frame --/--")
        layout.addWidget(self._frame_label)

        self._format_label = QLabel("Format: --")
        layout.addWidget(self._format_label)

        self._scale_label = QLabel(SCALE_NOT_SET)
        self._scale_label.setToolTip("Set the frame rate and scale of this recording in Export")
        layout.addWidget(self._scale_label)

        self._roi_label = QLabel("ROI: --")
        layout.addWidget(self._roi_label)

        layout.addStretch()

        # Quiet background work (e.g. "Updating…" while tracking after an edit).
        self._activity_label = QLabel("")
        self._activity_label.setObjectName("dimText")
        layout.addWidget(self._activity_label)
        self._mode = MEASURE_MODE

    def set_frame_labels(self, labels: Sequence[str]) -> None:
        """File-number tags per frame, shown after the frame index."""
        self._frame_labels = tuple(labels)

    def update_frame(self, current: int, total: int):
        """Show the 1-based *current* frame, e.g. ``Frame 12/91 · _023``."""
        self._frame_label.setText(
            format_frame_text(current, total, label_for(self._frame_labels, current)),
        )

    def frame_text(self) -> str:
        return self._frame_label.text()

    def update_format(self, fmt: str):
        self._format_label.setText(f"Format: {fmt}")

    def update_scale(self, um2px: float | None):
        """The recording's µm/px; None until it is set for this recording."""
        self._scale_label.setText(scale_text(um2px))

    def scale_text(self) -> str:
        return self._scale_label.text()

    def update_roi(self, gridx: tuple[int, int], gridy: tuple[int, int]):
        self._roi_label.setText(
            f"ROI: [{gridx[0]}-{gridx[1]}, {gridy[0]}-{gridy[1]}]"
        )

    def set_activity(self, text: str) -> None:
        """Background work in plain words ('Updating…'); '' when idle."""
        self._activity_label.setText(text)

    def activity_text(self) -> str:
        return self._activity_label.text()

    def update_mode(self, mode: str):
        """Kept for callers; the mode is no longer shown (it told users nothing)."""
        self._mode = mode

    def mode(self) -> str:
        return self._mode
