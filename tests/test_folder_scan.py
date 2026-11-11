"""Tests for folder scanning (model/folder_scan.py) and its scan_folder wrapper."""

from __future__ import annotations

import logging
import os

import cv2
import numpy as np

from bubbletrack.model.folder_scan import ScanResult, scan_folder_report
from bubbletrack.model.image_io import scan_folder

_IMG = np.zeros((8, 10), np.uint8)


def _touch_image(path) -> str:
    ok, buf = cv2.imencode(os.path.splitext(str(path))[1], _IMG)
    assert ok
    buf.tofile(str(path))
    return str(path)


def _names(paths) -> list[str]:
    return [os.path.basename(p) for p in paths]


class TestHiddenFiles:
    def test_report_counts_skipped_hidden_files(self, tmp_path):
        _touch_image(tmp_path / "frame_001.tif")
        (tmp_path / "._frame_001.tif").write_bytes(b"\x00\x05\x16\x07")
        (tmp_path / ".DS_Store").write_bytes(b"junk")  # not an image name
        report = scan_folder_report(str(tmp_path))
        assert _names(report.images) == ["frame_001.tif"]
        assert report.hidden_skipped == 1
        assert any("hidden" in w for w in report.warnings)


class TestExtensionGroups:
    def test_tif_and_tiff_are_one_group_natural_sorted(self, tmp_path):
        for name in ("a_10.tiff", "a_2.TIFF", "a_1.tif", "a_3.Tif"):
            _touch_image(tmp_path / name)
        assert _names(scan_folder(str(tmp_path))) == [
            "a_1.tif", "a_2.TIFF", "a_3.Tif", "a_10.tiff",
        ]

    def test_jpg_and_jpeg_are_one_group(self, tmp_path):
        for name in ("f1.jpg", "f2.JPEG", "f3.jpeg", "f11.JPG"):
            _touch_image(tmp_path / name)
        assert _names(scan_folder(str(tmp_path))) == ["f1.jpg", "f2.JPEG", "f3.jpeg", "f11.JPG"]

    def test_priority_is_kept_and_other_groups_are_reported(self, tmp_path, caplog):
        for i in range(3):
            _touch_image(tmp_path / f"frame_{i}.tif")
        for i in range(2):
            _touch_image(tmp_path / f"preview_{i}.png")
        _touch_image(tmp_path / "thumb.bmp")

        with caplog.at_level(logging.WARNING, logger="bubbletrack.model.folder_scan"):
            report = scan_folder_report(str(tmp_path))

        assert _names(report.images) == ["frame_0.tif", "frame_1.tif", "frame_2.tif"]
        assert report.group == "tiff"
        assert report.ignored_counts == {".png": 2, ".bmp": 1}
        assert any(".png" in w for w in report.warnings)
        assert any("ignored" in r.message.lower() for r in caplog.records)

    def test_no_warning_when_one_group(self, tmp_path):
        _touch_image(tmp_path / "a.png")
        report = scan_folder_report(str(tmp_path))
        assert report.group == "png"
        assert report.ignored_counts == {}
        assert report.warnings == ()

    def test_directory_with_image_suffix_is_not_a_frame(self, tmp_path):
        (tmp_path / "subdir.tif").mkdir()
        _touch_image(tmp_path / "a.tif")
        assert _names(scan_folder(str(tmp_path))) == ["a.tif"]

    def test_non_image_files_are_not_reported(self, tmp_path):
        _touch_image(tmp_path / "a.tif")
        (tmp_path / "notes.txt").write_text("x")
        (tmp_path / "result.mat").write_bytes(b"x")
        assert scan_folder_report(str(tmp_path)).ignored_counts == {}

    def test_scan_folder_keeps_return_type(self, tmp_path):
        _touch_image(tmp_path / "a.tif")
        paths = scan_folder(str(tmp_path))
        assert isinstance(paths, list)
        assert all(isinstance(p, str) and os.path.isabs(p) for p in paths)
        assert paths == list(scan_folder_report(str(tmp_path)).images)

    def test_missing_folder_gives_empty_report(self, tmp_path):
        report = scan_folder_report(str(tmp_path / "missing"))
        assert report == ScanResult(images=())
        assert report.group is None
