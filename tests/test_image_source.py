"""Tests for the dual drop-zone ImageSource widget.

Verifies:
- Both drop zones accept drops and exist as named children
- Folder zone emits ``folder_selected`` for directory drops
- Folder zone falls back to parent directory when a file is dropped
  (matches STAQ-DIC ``_DropZone`` behaviour)
- Video zone emits ``video_selected`` for valid video files
- Video zone silently rejects non-video files
- ``set_folder_loaded`` / ``set_video_loaded`` toggle the ``loaded`` Qt
  dynamic property and reset the opposite zone

These tests instantiate real Qt widgets and synthesize ``QDropEvent``
objects directly — no real OS-level drag is performed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from PyQt6.QtCore import QMimeData, QPointF, Qt, QUrl
from PyQt6.QtGui import QDropEvent
from PyQt6.QtWidgets import QApplication

from bubbletrack.ui.image_source import ImageSource


def _norm(path: str) -> Path:
    """Normalize a string path for cross-separator comparison.

    Qt's ``QUrl.fromLocalFile`` round-trips a Windows path with forward
    slashes, while Python's ``str(WindowsPath)`` uses backslashes.  Both
    refer to the same filesystem location, so compare them as ``Path``
    objects after resolving.
    """
    return Path(path)


@pytest.fixture(scope="session")
def qapp():
    """Session-scoped QApplication — Qt allows only one per process."""
    app = QApplication.instance() or QApplication(sys.argv)
    yield app


def _make_drop_event(local_path: str, _alive: list) -> QDropEvent:
    """Build a QDropEvent carrying a single local file/folder URL.

    Qt's ``QDropEvent`` does not take ownership of the ``QMimeData``
    pointer, so the caller must keep a Python reference alive for the
    duration of ``dropEvent``.  Pass an ``_alive`` list and the helper
    will append the mime object to it.
    """
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(local_path)])
    _alive.append(mime)
    return QDropEvent(
        QPointF(10, 10),
        Qt.DropAction.CopyAction,
        mime,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
        QDropEvent.Type.Drop,
    )


def test_image_source_exposes_two_drop_zones(qapp):
    src = ImageSource()
    assert hasattr(src, "_folder_zone"), "ImageSource missing _folder_zone"
    assert hasattr(src, "_video_zone"), "ImageSource missing _video_zone"
    assert src._folder_zone.acceptDrops() is True
    assert src._video_zone.acceptDrops() is True


def test_folder_zone_emits_signal_on_dir_drop(qapp, tmp_path):
    src = ImageSource()
    received: list[str] = []
    src.folder_selected.connect(received.append)
    alive: list = []

    src._folder_zone.dropEvent(_make_drop_event(str(tmp_path), alive))

    assert len(received) == 1
    assert _norm(received[0]) == _norm(str(tmp_path))


def test_folder_zone_falls_back_to_parent_when_file_dropped(qapp, tmp_path):
    file_path = tmp_path / "frame.tif"
    file_path.write_bytes(b"")
    src = ImageSource()
    received: list[str] = []
    src.folder_selected.connect(received.append)
    alive: list = []

    src._folder_zone.dropEvent(_make_drop_event(str(file_path), alive))

    assert len(received) == 1
    assert _norm(received[0]) == _norm(str(tmp_path))


def test_video_zone_emits_signal_on_video_drop(qapp, tmp_path):
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"")
    src = ImageSource()
    received: list[str] = []
    src.video_selected.connect(received.append)
    alive: list = []

    src._video_zone.dropEvent(_make_drop_event(str(video_path), alive))

    assert len(received) == 1
    assert _norm(received[0]) == _norm(str(video_path))


def test_video_zone_rejects_non_video_files(qapp, tmp_path):
    txt_path = tmp_path / "notes.txt"
    txt_path.write_bytes(b"")
    src = ImageSource()
    received: list[str] = []
    src.video_selected.connect(received.append)
    folder_received: list[str] = []
    src.folder_selected.connect(folder_received.append)
    alive: list = []

    src._video_zone.dropEvent(_make_drop_event(str(txt_path), alive))

    assert received == [], "video_selected should not fire for non-video files"
    assert folder_received == [], "folder_selected should not fire either"


def test_set_folder_loaded_changes_loaded_property(qapp):
    src = ImageSource()
    src.set_folder_loaded("/tmp/foo", 42)

    assert src._folder_zone.property("loaded") is True
    assert src._video_zone.property("loaded") is False


def test_set_video_loaded_resets_folder_zone(qapp):
    src = ImageSource()
    src.set_folder_loaded("/tmp/foo", 42)
    assert src._folder_zone.property("loaded") is True

    src.set_video_loaded("/tmp/clip.mp4", 120, 30.0)

    assert src._video_zone.property("loaded") is True
    assert src._folder_zone.property("loaded") is False, (
        "Loading a video must reset the folder zone — only one source "
        "can be active at a time."
    )
