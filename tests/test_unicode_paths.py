"""Image and video reads from folders with non-ASCII names.

``cv2.imread`` cannot open such paths on Windows (it returns None), and the
lab's folders are named like ``实验_25µs``.  These tests write files with
``cv2.imencode`` + ``ndarray.tofile`` (``cv2.imwrite`` has the same problem)
and check that every read path in the app copes.
"""

from __future__ import annotations

import os
import shutil

import cv2
import numpy as np
import pytest

from _golden_cases import SceneSpec, render_scene
from bubbletrack import cli
from bubbletrack.model.image_io import (
    ImageLoadError,
    VideoFrameReader,
    detect_bit_depth,
    imread_unicode,
    load_and_normalize,
    scan_folder,
)

UNICODE_NAME = "实验 25µs it's Å😀"


def _write_unicode(path, img: np.ndarray, ext: str = ".tiff") -> str:
    ok, buf = cv2.imencode(ext, img)
    assert ok
    buf.tofile(str(path))
    return str(path)


@pytest.fixture()
def unicode_dir(tmp_path):
    folder = tmp_path / UNICODE_NAME
    folder.mkdir()
    return folder


@pytest.fixture()
def frame16() -> np.ndarray:
    return render_scene(SceneSpec(radius=40.0, center=(60.0, 80.0), shape=(120, 160), seed=3))


class TestImreadUnicode:
    def test_reads_16bit_tiff_bit_exact(self, unicode_dir, frame16):
        path = _write_unicode(unicode_dir / "帧_001.tiff", frame16)
        img = imread_unicode(path, cv2.IMREAD_UNCHANGED)
        assert img is not None
        assert img.dtype == np.uint16
        np.testing.assert_array_equal(img, frame16)

    def test_reads_8bit_png(self, unicode_dir):
        src = np.arange(40 * 50, dtype=np.uint32).reshape(40, 50).astype(np.uint8)
        path = _write_unicode(unicode_dir / "µ.png", src, ".png")
        np.testing.assert_array_equal(imread_unicode(path, cv2.IMREAD_UNCHANGED), src)

    def test_default_flags_keep_bit_depth(self, unicode_dir, frame16):
        path = _write_unicode(unicode_dir / "a.tiff", frame16)
        assert imread_unicode(path).dtype == np.uint16

    def test_accepts_pathlib_path(self, unicode_dir, frame16):
        path = unicode_dir / "a.tiff"
        _write_unicode(path, frame16)
        assert imread_unicode(path) is not None

    def test_missing_file_returns_none(self, unicode_dir):
        assert imread_unicode(str(unicode_dir / "nope.tiff")) is None

    def test_empty_file_returns_none(self, unicode_dir):
        path = unicode_dir / "empty.tiff"
        path.write_bytes(b"")
        assert imread_unicode(str(path)) is None

    def test_corrupted_file_returns_none(self, unicode_dir):
        path = unicode_dir / "bad.tiff"
        path.write_bytes(os.urandom(256))
        assert imread_unicode(str(path)) is None

    def test_directory_returns_none(self, unicode_dir):
        assert imread_unicode(str(unicode_dir)) is None


class TestLoadersUseUnicodeReader:
    def test_scan_and_load_and_normalize(self, unicode_dir, frame16):
        for i in range(3):
            _write_unicode(unicode_dir / f"frame_{i}.tiff", frame16)
        paths = scan_folder(str(unicode_dir))
        assert len(paths) == 3
        full, binary, roi, roi_bin = load_and_normalize(paths[0], 0.5, (1, 120), (1, 160))
        assert full.shape == (120, 160)
        assert binary.any() and not binary.all()

    def test_load_missing_unicode_file_still_raises(self, unicode_dir):
        with pytest.raises(ImageLoadError, match="missing or corrupted"):
            load_and_normalize(str(unicode_dir / "nope.tiff"), 0.5, (1, 10), (1, 10))

    def test_detect_bit_depth(self, unicode_dir, frame16):
        path = _write_unicode(unicode_dir / "a.tiff", frame16)
        assert detect_bit_depth(path) == 65535

    def test_cli_fits_frames_in_unicode_folder(self, unicode_dir, frame16, tmp_path, capsys):
        for i in range(2):
            _write_unicode(unicode_dir / f"frame_{i}.tiff", frame16)
        out = tmp_path / "out.csv"
        cli.main(["--folder", str(unicode_dir), "--output", str(out)])
        assert "2/2 frames fitted" in capsys.readouterr().out


def test_video_capture_opens_unicode_path(unicode_dir, tmp_path):
    """OpenCV's FFmpeg backend opens non-ASCII video paths (unlike imread)."""
    ascii_path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(ascii_path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    if not writer.isOpened():
        pytest.skip("no MJPG video writer in this OpenCV build")
    for i in range(4):
        writer.write(np.full((48, 64, 3), 40 * i, np.uint8))
    writer.release()
    target = unicode_dir / "视频.avi"
    shutil.copyfile(ascii_path, target)

    reader = VideoFrameReader(str(target))
    try:
        assert reader.total_frames == 4
        assert reader.read_frame(0).shape == (48, 64)
    finally:
        reader.close()
