"""Data under a path longer than Windows' MAX_PATH (260) still loads.

Without the registry's LongPathsEnabled switch, plain Win32 calls fail on
such paths; the ``\\\\?\\`` prefix lifts the limit.  Paths stay plain in the
app (state, sessions, labels); only the OS calls get the prefix.
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np
import pytest

from bubbletrack.model import long_paths
from bubbletrack.model.long_paths import MAX_PATH, os_path

windows_only = pytest.mark.skipif(sys.platform != "win32", reason="Windows path limit")
PREFIX = "\\\\?\\"


def _long_dir(root) -> str:
    """A directory whose path is about 300 characters, created with the prefix."""
    parts = [str(root)]
    while len("\\".join(parts)) < 300:
        parts.append("d" * 60)
    path = "\\".join(parts)
    os.makedirs(PREFIX + path, exist_ok=True)
    return path


def _write_tiff(path: str, value: int = 1000) -> None:
    img = np.full((40, 60), value, np.uint16)
    img[10:30, 20:40] = 200
    ok, buf = cv2.imencode(".tiff", img)
    assert ok
    with open(os_path(path), "wb") as fh:
        fh.write(buf.tobytes())


class TestOsPath:
    def test_short_paths_are_unchanged(self, tmp_path):
        path = str(tmp_path / "a.tiff")
        assert os_path(path) == path

    @windows_only
    def test_long_absolute_paths_get_the_prefix(self):
        path = "C:\\" + "\\".join(["x" * 50] * 6) + "\\f.tiff"
        assert len(path) > MAX_PATH
        assert os_path(path) == PREFIX + path

    @windows_only
    def test_long_unc_paths_use_the_unc_form(self):
        path = "\\\\server\\share\\" + "\\".join(["x" * 50] * 6)
        assert os_path(path) == PREFIX + "UNC\\server\\share\\" + "\\".join(["x" * 50] * 6)

    @windows_only
    def test_forward_slashes_and_dots_are_normalised_first(self):
        path = "C:/" + "/".join(["x" * 50] * 6) + "/../f.tiff"
        out = os_path(path)
        assert out.startswith(PREFIX) and "/" not in out and ".." not in out

    @windows_only
    def test_already_prefixed_paths_are_kept(self):
        path = PREFIX + "C:\\" + "x" * 300
        assert os_path(path) == path

    def test_other_platforms_are_unchanged(self, monkeypatch):
        monkeypatch.setattr(long_paths, "_WINDOWS", False)
        path = "/" + "/".join(["x" * 50] * 7)
        assert os_path(path) == path


@windows_only
class TestReadingUnderALongPath:
    def test_image_is_read(self, tmp_path):
        from bubbletrack.model.image_io import imread_unicode, load_and_normalize

        folder = _long_dir(tmp_path)
        path = folder + "\\frame_001.tiff"
        _write_tiff(path)
        raw = imread_unicode(path)
        assert raw is not None and raw.shape == (40, 60)
        cur_img, *_ = load_and_normalize(path, 0.5, (1, 40), (1, 60))
        assert cur_img.shape == (40, 60)

    def test_folder_is_scanned_and_paths_stay_plain(self, tmp_path):
        from bubbletrack.model.image_io import scan_folder

        folder = _long_dir(tmp_path)
        for i in range(3):
            _write_tiff(f"{folder}\\frame_{i + 1:03d}.tiff")
        images = scan_folder(folder)
        assert [os.path.basename(p) for p in images] == [
            "frame_001.tiff", "frame_002.tiff", "frame_003.tiff"]
        assert not any(p.startswith(PREFIX) for p in images)

    def test_engine_reads_the_raw_frame(self, tmp_path):
        from bubbletrack.model.edge_pipeline import read_raw

        path = _long_dir(tmp_path) + "\\frame_001.tiff"
        _write_tiff(path, 1234)
        assert read_raw(path)[0, 0] == 1234

    def test_video_is_opened(self, tmp_path):
        from bubbletrack.model.image_io import VideoFrameReader

        path = _long_dir(tmp_path) + "\\clip.avi"
        writer = cv2.VideoWriter(os_path(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48), False)
        if not writer.isOpened():
            pytest.skip("no MJPG writer in this OpenCV build")
        for k in range(4):
            writer.write(np.full((48, 64), 40 * k, np.uint8))
        writer.release()
        reader = VideoFrameReader(path)
        try:
            assert reader.total_frames == 4
            assert reader.read_frame(1).shape == (48, 64)
        finally:
            reader.close()

    def test_drop_zone_accepts_the_folder(self, qapp, tmp_path):
        from bubbletrack.ui.image_source import ImageSource

        folder = _long_dir(tmp_path)
        source = ImageSource()
        got = []
        source.folder_selected.connect(got.append)
        source._folder_zone.accept_path(folder)
        assert got == [folder]


class TestReadFailureMessage:
    def test_missing_file_says_not_found(self, tmp_path):
        from bubbletrack.model.image_io import ImageLoadError, load_and_normalize

        with pytest.raises(ImageLoadError, match="not found"):
            load_and_normalize(str(tmp_path / "gone.tiff"), 0.5, (1, 9), (1, 9))

    def test_damaged_file_says_so(self, tmp_path):
        from bubbletrack.model.image_io import ImageLoadError, load_and_normalize

        path = tmp_path / "bad.tiff"
        path.write_bytes(b"not an image at all")
        with pytest.raises(ImageLoadError, match="damaged"):
            load_and_normalize(str(path), 0.5, (1, 9), (1, 9))

    @windows_only
    def test_a_long_path_that_still_fails_says_path_too_long(self, tmp_path, monkeypatch):
        from bubbletrack.model import edge_pipeline, image_io

        path = str(tmp_path / ("x" * 120) / ("y" * 120) / "frame.tiff")
        monkeypatch.setattr(image_io, "imread_unicode", lambda *_a, **_k: None)
        monkeypatch.setattr(edge_pipeline, "imread_unicode", lambda *_a, **_k: None)
        with pytest.raises(image_io.ImageLoadError, match="path is too long"):
            image_io.load_and_normalize(path, 0.5, (1, 9), (1, 9))
        with pytest.raises(image_io.ImageLoadError, match="path is too long"):
            edge_pipeline.read_raw(path)
