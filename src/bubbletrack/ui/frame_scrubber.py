"""Frame scrubber — horizontal slider with frame counter and file number."""

from __future__ import annotations

import os
import re
from collections.abc import Sequence

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QSlider, QSpinBox, QWidget

VIDEO_PREFIX = "video:"
_TRAILING_NUMBER = re.compile(r"([_\-. ]?)(\d+)$")
_MAX_STEM_CHARS = 14


def frame_file_label(name: str) -> str:
    """Short file-number tag for a frame: ``"_011"`` for ``img_011.tif``.

    The trailing digits of the file stem (with the separator before them)
    identify the camera frame; video placeholders ``"video:i"`` give
    ``"#i"`` (the frame number in the video).  Names without trailing
    digits fall back to a shortened stem.
    """
    if name.startswith(VIDEO_PREFIX):
        return f"#{name[len(VIDEO_PREFIX):]}"
    stem = os.path.splitext(os.path.basename(name))[0]
    match = _TRAILING_NUMBER.search(stem)
    if match:
        return match.group(1) + match.group(2)
    if len(stem) > _MAX_STEM_CHARS:
        return stem[:_MAX_STEM_CHARS - 1] + "…"
    return stem


def frame_file_labels(names: Sequence[str]) -> tuple[str, ...]:
    return tuple(frame_file_label(n) for n in names)


def format_frame_text(current: int, total: int, file_label: str = "") -> str:
    """``"Frame 1/91 · _011"`` (1-based *current*); no tag when *file_label* is empty."""
    text = f"Frame {current}/{total}"
    return f"{text} · {file_label}" if file_label else text


def label_for(labels: Sequence[str], current_1based: int) -> str:
    """The file label of a 1-based frame, or "" when unknown."""
    idx = current_1based - 1
    return labels[idx] if 0 <= idx < len(labels) else ""


class FrameScrubber(QWidget):
    """``[<] [>] ====o==== [>]  Frame 1/100 · _011``."""

    value_changed = pyqtSignal(int)  # 0-indexed frame number

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)

        # Prev button
        self._prev = _NavButton("\u2039")  # glyph present in every UI font
        self._prev.setToolTip("Previous frame (Left arrow)")
        self._prev.clicked.connect(self._step_back)
        layout.addWidget(self._prev)

        # Play/pause button
        self._play_btn = _NavButton("\u25B6")
        self._play_btn.clicked.connect(lambda _checked: self.toggle_play())
        layout.addWidget(self._play_btn)

        # Slider
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(0)
        self._slider.valueChanged.connect(self._on_slider)
        layout.addWidget(self._slider, 1)

        # Next button
        self._next = _NavButton("\u203A")
        self._next.setToolTip("Next frame (Right arrow)")
        self._next.clicked.connect(self._step_forward)
        layout.addWidget(self._next)

        # FPS control
        self._fps_spin = QSpinBox()
        self._fps_spin.setRange(1, 60)
        self._fps_spin.setValue(10)
        self._fps_spin.setSuffix(" fps")
        self._fps_spin.setFixedWidth(72)
        self._fps_spin.valueChanged.connect(self._on_fps_changed)
        layout.addWidget(self._fps_spin)

        # Frame label: "Frame 12/91 · _023"
        self._labels: tuple[str, ...] = ()
        self._label = QLabel("Frame --/--")
        self._label.setObjectName("frameLabel")
        self._label.setMinimumWidth(
            self._label.fontMetrics().horizontalAdvance("Frame 0000/0000 · _00000") + 8,
        )
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setToolTip("Frame index / total · file number (video: frame number)")
        layout.addWidget(self._label)

        # Playback timer
        self._play_timer = QTimer()
        self._play_timer.setInterval(100)  # 1000 / 10 fps
        self._play_timer.timeout.connect(self._play_tick)
        self._playing = False

    def set_range(self, total: int):
        self._slider.setMaximum(max(total - 1, 0))
        self._update_label()

    def set_frame_labels(self, labels: Sequence[str]) -> None:
        """File-number tags per frame (see :func:`frame_file_labels`)."""
        self._labels = tuple(labels)
        self._update_label()

    def label_text(self) -> str:
        return self._label.text()

    def set_value(self, idx: int):
        self._slider.setValue(idx)

    def value(self) -> int:
        return self._slider.value()

    def _on_slider(self, v: int):
        self._update_label()
        self.value_changed.emit(v)

    def _step_back(self):
        self._slider.setValue(max(self._slider.value() - 1, 0))

    def _step_forward(self):
        self._slider.setValue(min(self._slider.value() + 1, self._slider.maximum()))

    # -- Playback controls --

    def toggle_play(self) -> None:
        """Toggle play/pause — public method for shortcut binding."""
        if self._playing:
            self._stop_play()
        else:
            self._start_play()

    def _start_play(self) -> None:
        # If at the end, wrap to start
        if self._slider.value() >= self._slider.maximum():
            self._slider.setValue(0)
        self._playing = True
        self._play_btn.setText("\u2016")  # pause; glyph present in every UI font
        self._play_timer.setInterval(max(1, 1000 // self._fps_spin.value()))
        self._play_timer.start()

    def _stop_play(self) -> None:
        self._playing = False
        self._play_timer.stop()
        self._play_btn.setText("\u25B6")  # ▶

    def _play_tick(self) -> None:
        nxt = self._slider.value() + 1
        if nxt > self._slider.maximum():
            self._stop_play()
            return
        self._slider.setValue(nxt)

    def _on_fps_changed(self, fps: int) -> None:
        if self._playing:
            self._play_timer.setInterval(max(1, 1000 // fps))

    def _update_label(self):
        cur = self._slider.value() + 1  # 1-indexed display
        total = self._slider.maximum() + 1
        self._label.setText(format_frame_text(cur, total, label_for(self._labels, cur)))


class _NavButton(QWidget):
    """Tiny navigation arrow button."""

    def __new__(cls, text: str):
        btn = QPushButton(text)
        btn.setFixedSize(32, 28)
        btn.setObjectName("secondaryBtn")
        return btn
