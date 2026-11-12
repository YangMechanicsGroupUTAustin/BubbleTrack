"""Dialog for configuring batch processing of folders and videos."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from bubbletrack.model.batch_experiments import BatchSource, as_source, count_frames
from bubbletrack.model.constants import (
    DEFAULT_FPS,
    DEFAULT_RMAX_FIT_LENGTH,
    DEFAULT_UM2PX,
)

_MODE_NOTES = {
    "edge": "Detection: Edge engine. A bubble-free reference frame is chosen "
            "automatically for every source.",
}


@dataclass(frozen=True)
class BatchConfig:
    """User-confirmed batch processing configuration."""

    sources: list[BatchSource]
    export_physical: bool = False
    fps: float = DEFAULT_FPS
    um2px: float = DEFAULT_UM2PX
    rmax_fit_length: int = DEFAULT_RMAX_FIT_LENGTH

    @property
    def folders(self) -> list[Path]:
        """Paths of the selected sources (name kept for older callers)."""
        return [s.path for s in self.sources]


def source_label(source: BatchSource, n_frames: int | None) -> str:
    """e.g. 'shot_1.avi  —  video · 256 frames' or '... · unreadable'."""
    count = "unreadable" if n_frames is None else f"{n_frames} frames"
    return f"{source.name}  —  {source.kind} · {count}"


class BatchConfigDialog(QDialog):
    """Modal dialog for reviewing sources and setting batch export options.

    Layout
    ------
    - Checkable source list (type and frame count per source)
    - Always exports R_data.mat (pixel data)
    - Optional checkbox to also export RofT_data.mat (physical data)
      - FPS, um/px, and Rmax Fit Length inputs appear when checked
    - Start / Cancel buttons
    """

    def __init__(
        self,
        sources: list,
        *,
        detection_mode: str = "intensity",
        default_fps: float = DEFAULT_FPS,
        default_um2px: float = DEFAULT_UM2PX,
        default_fit_length: int = DEFAULT_RMAX_FIT_LENGTH,
        frame_counter: Callable[[BatchSource], int | None] = count_frames,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._sources = [as_source(s) for s in sources]
        self._counts = [frame_counter(s) for s in self._sources]
        self._mode = detection_mode
        self.setWindowTitle("Batch Processing")
        self.setMinimumWidth(480)
        self._build_ui(default_fps, default_um2px, default_fit_length)

    def _build_ui(
        self,
        default_fps: float,
        default_um2px: float,
        default_fit_length: int,
    ) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # -- Source list --
        n_videos = sum(s.kind == "video" for s in self._sources)
        n_folders = len(self._sources) - n_videos
        self._summary = QLabel(
            f"Found {len(self._sources)} source(s): {n_folders} image folder(s), "
            f"{n_videos} video(s). Untick any to skip."
        )
        self._summary.setWordWrap(True)
        layout.addWidget(self._summary)

        self._source_list = QListWidget()
        self._source_list.setMaximumHeight(200)
        for source, n in zip(self._sources, self._counts):
            item = QListWidgetItem(source_label(source, n))
            item.setToolTip(str(source.path))
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            readable = bool(n)
            item.setCheckState(Qt.CheckState.Checked if readable else Qt.CheckState.Unchecked)
            self._source_list.addItem(item)
        self._source_list.itemChanged.connect(self._update_start_enabled)
        layout.addWidget(self._source_list)

        mode_lbl = QLabel(_MODE_NOTES.get(
            self._mode, f"Detection: {self._mode} (the current parameters, for every source)."))
        mode_lbl.setObjectName("dimText")
        mode_lbl.setWordWrap(True)
        layout.addWidget(mode_lbl)

        # -- Export options --
        pixel_lbl = QLabel(
            "Pixel Data (R_data_<time>.mat) is exported into each folder, and "
            "next to each video as R_data_<video name>_<time>.mat."
        )
        pixel_lbl.setObjectName("dimText")
        pixel_lbl.setWordWrap(True)
        layout.addWidget(pixel_lbl)

        self._phys_check = QCheckBox("Also export Physical Data (RofT_data.mat)")
        self._phys_check.toggled.connect(self._on_phys_toggled)
        layout.addWidget(self._phys_check)
        layout.addWidget(self._build_phys_group(default_fps, default_um2px, default_fit_length))

        # -- Buttons --
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setObjectName("secondaryBtn")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)

        self._start_btn = QPushButton("Start Batch")
        self._start_btn.clicked.connect(self.accept)
        self._start_btn.setDefault(True)
        btn_row.addWidget(self._start_btn)

        layout.addLayout(btn_row)
        self._update_start_enabled()

    def _build_phys_group(self, default_fps: float, default_um2px: float,
                          default_fit_length: int) -> QWidget:
        self._phys_group = QWidget()
        phys_layout = QVBoxLayout(self._phys_group)
        phys_layout.setContentsMargins(20, 0, 0, 0)
        phys_layout.setSpacing(6)

        note = QLabel("These values apply to all sources (same camera setup):")
        note.setObjectName("dimText")
        note.setWordWrap(True)
        phys_layout.addWidget(note)

        self._fps_spin = QDoubleSpinBox()
        self._fps_spin.setRange(0.01, 100_000_000)
        self._fps_spin.setDecimals(0)
        self._fps_spin.setValue(default_fps)
        phys_layout.addLayout(self._row("FPS:", self._fps_spin, 120))

        self._um_spin = QDoubleSpinBox()
        self._um_spin.setRange(0.001, 10000)
        self._um_spin.setDecimals(3)
        self._um_spin.setValue(default_um2px)
        phys_layout.addLayout(self._row("µm/px:", self._um_spin, 120))

        self._fit_spin = QSpinBox()
        self._fit_spin.setRange(3, 101)
        self._fit_spin.setSingleStep(2)
        self._fit_spin.setValue(default_fit_length)
        phys_layout.addLayout(self._row("Rmax Fit Length:", self._fit_spin, 80))

        self._phys_group.hide()
        return self._phys_group

    @staticmethod
    def _row(label: str, widget: QWidget, width: int) -> QHBoxLayout:
        row = QHBoxLayout()
        row.addWidget(QLabel(label))
        widget.setMinimumWidth(width)
        row.addWidget(widget)
        row.addStretch()
        return row

    def _on_phys_toggled(self, checked: bool) -> None:
        self._phys_group.setVisible(checked)
        self.adjustSize()

    def _selected(self) -> list[BatchSource]:
        return [s for i, s in enumerate(self._sources)
                if self._source_list.item(i).checkState() == Qt.CheckState.Checked]

    def _update_start_enabled(self, *_args) -> None:
        self._start_btn.setEnabled(bool(self._selected()))

    def get_config(self) -> BatchConfig:
        """Return the user-confirmed configuration (ticked sources only)."""
        return BatchConfig(
            sources=self._selected(),
            export_physical=self._phys_check.isChecked(),
            fps=self._fps_spin.value(),
            um2px=self._um_spin.value(),
            rmax_fit_length=self._fit_spin.value(),
        )
