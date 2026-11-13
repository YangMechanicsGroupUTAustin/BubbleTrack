"""Image source panel — dual drop-zone selector for folders and videos.

Inspired by STAQ-DIC-GUI's ``_DropZone`` design
(``staq-dic-python/src/staq_dic/gui/panels/left_sidebar.py:75-145``):
each zone supports both a click gesture (opens a file dialog) and a drag-
and-drop gesture (accepts ``QMimeData`` URLs).  Two zones are arranged
side-by-side so the user can pick either an image folder or a video file
without first toggling a mode.

Loaded state is communicated via Qt's dynamic property mechanism
(``setProperty("loaded", ...)`` + re-polish) so the QSS attribute selector
``QWidget#dropZone[loaded="true"]`` can swap the border colour without
runtime stylesheet rewriting.
"""

from __future__ import annotations

import os
from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QMouseEvent, QResizeEvent
from PyQt6.QtWidgets import (
    QFileDialog, QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget,
)

from bubbletrack.model.frame_source import RECORDING_DIALOG_FILTER, is_recording_file
from bubbletrack.model.long_paths import os_path


def format_rate(fps: float) -> str:
    """Frame rate with an SI prefix: '29.97 fps', '50 kfps', '5 Mfps'."""
    for scale, unit in ((1e6, "Mfps"), (1e3, "kfps")):
        if fps >= scale:
            return f"{fps / scale:.4g} {unit}"
    return f"{fps:.4g} fps"


class _DropZone(QWidget):
    """A drop-and-click target for either a folder or a single file.

    Emits ``path_selected(str)`` when the user picks a path via either
    gesture.  Folder mode also accepts a dropped file by falling back to
    the file's parent directory (matches STAQ-DIC behaviour).
    """

    path_selected = pyqtSignal(str)

    def __init__(
        self,
        *,
        mode: str,
        icon: str,
        prompt: str,
        dialog_title: str,
        dialog_filter: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        if mode not in ("folder", "video"):
            raise ValueError(
                f"_DropZone mode must be 'folder' or 'video', got {mode!r}"
            )
        self._mode = mode
        self._dialog_title = dialog_title
        self._dialog_filter = dialog_filter
        self._original_prompt = prompt

        self.setObjectName("dropZone")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setAcceptDrops(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setProperty("loaded", False)
        self.setMinimumHeight(80)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 10, 8, 10)
        layout.setSpacing(2)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._icon_label = QLabel(icon)
        self._icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._icon_label.setStyleSheet(
            "font-size: 22px; background: transparent; border: none;"
        )
        layout.addWidget(self._icon_label)

        self._prompt_label = QLabel(prompt)
        self._prompt_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._prompt_label.setWordWrap(True)
        # A long name must not widen the zone (it is elided to fit instead).
        self._prompt_label.setSizePolicy(QSizePolicy.Policy.Ignored,
                                         QSizePolicy.Policy.Preferred)
        layout.addWidget(self._prompt_label)
        self._summary: list[str] = []

    # -- Public state API --

    def set_loaded(self, summary_lines: list[str], tooltip: str = "") -> None:
        """Show *summary_lines* (the first, the name, elided to fit) in the loaded style."""
        self._summary = list(summary_lines)
        self._prompt_label.setToolTip(tooltip)
        self._show_summary()
        self._set_loaded_property(True)

    def summary_text(self) -> str:
        """The lines as shown (the name possibly elided)."""
        return self._prompt_label.text()

    def reset(self) -> None:
        """Restore the original prompt and clear the loaded visual style."""
        self._summary = []
        self._prompt_label.setToolTip("")
        self._prompt_label.setText(self._original_prompt)
        self._set_loaded_property(False)

    def _show_summary(self) -> None:
        if not self._summary:
            return
        width = max(self._prompt_label.width(), self.width() - 16, 40)
        metrics = self._prompt_label.fontMetrics()
        first = metrics.elidedText(self._summary[0], Qt.TextElideMode.ElideMiddle, width)
        self._prompt_label.setText("\n".join([first, *self._summary[1:]]))

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802 (Qt override)
        super().resizeEvent(event)
        self._show_summary()

    def _set_loaded_property(self, value: bool) -> None:
        self.setProperty("loaded", value)
        # Re-polish so the QSS [loaded="true"] selector takes effect
        style = self.style()
        style.unpolish(self)
        style.polish(self)
        self.update()

    # -- Click gesture --

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        if self._mode == "folder":
            path = QFileDialog.getExistingDirectory(self, self._dialog_title)
        else:
            path, _ = QFileDialog.getOpenFileName(
                self, self._dialog_title, "", self._dialog_filter,
            )
        if path:
            self.path_selected.emit(path)

    # -- Drop gesture --

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802
        urls = event.mimeData().urls()
        if not urls:
            return
        path = urls[0].toLocalFile()
        if path:
            self.accept_path(path)

    def accept_path(self, path: str) -> None:
        """Emit *path* if this zone takes it (a dropped file selects its folder)."""
        if self._mode == "folder":
            if os.path.isdir(os_path(path)):
                self.path_selected.emit(path)
            elif os.path.isfile(os_path(path)):
                # Fallback: use the file's parent directory
                self.path_selected.emit(str(Path(path).parent))
        else:  # video or .dat recording
            if os.path.isfile(os_path(path)) and is_recording_file(path):
                self.path_selected.emit(path)


class ImageSource(QWidget):
    """Folder/video selector that emits the chosen path.

    Public signals (preserved for backward compatibility with the
    controller wiring):
    - ``folder_selected(str)`` — emitted when an image folder is chosen
    - ``video_selected(str)`` — emitted when a video file is chosen
    """

    folder_selected = pyqtSignal(str)
    video_selected = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 8)
        layout.setSpacing(6)

        title = QLabel("Image Source")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)

        zones_row = QHBoxLayout()
        zones_row.setContentsMargins(0, 0, 0, 0)
        zones_row.setSpacing(8)

        self._folder_zone = _DropZone(
            mode="folder",
            icon="\U0001F4C1",  # folder emoji
            prompt="Drop image\nfolder\nor click",
            dialog_title="Select Image Folder",
        )
        self._video_zone = _DropZone(
            mode="video",
            icon="\U0001F3AC",  # clapper-board emoji
            prompt="Drop video\nor .dat file\nor click",
            dialog_title="Select Video or .dat Recording",
            dialog_filter=RECORDING_DIALOG_FILTER,
        )
        zones_row.addWidget(self._folder_zone, 1)
        zones_row.addWidget(self._video_zone, 1)
        layout.addLayout(zones_row)

        # Re-emit child signals through the public API
        self._folder_zone.path_selected.connect(self.folder_selected)
        self._video_zone.path_selected.connect(self.video_selected)

    # -- Public API used by FileController --

    def set_folder_loaded(self, folder: str, count: int) -> None:
        """Show a loaded folder in the folder zone; reset the video zone.

        Only one source can be active at a time, so loading a folder
        always clears any previously displayed video summary.
        """
        name = os.path.basename(folder.rstrip("/\\")) or folder
        noun = "image" if count == 1 else "images"
        self._folder_zone.set_loaded([f"\u2713 {name}", f"{count} {noun}"], tooltip=folder)
        self._video_zone.reset()

    def set_video_loaded(
        self, video_path: str, frames: int, fps: float,
    ) -> None:
        """Show a loaded video in the video zone; reset the folder zone."""
        name = os.path.basename(video_path)
        self._video_zone.set_loaded(
            [f"\u2713 {name}", f"{frames} fr  |  {format_rate(fps)}"], tooltip=video_path,
        )
        self._folder_zone.reset()
