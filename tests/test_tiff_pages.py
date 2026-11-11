"""Multi-page TIFF detection: tiff_page_count and the scan report flag."""

from __future__ import annotations

import os
import struct

import cv2
import numpy as np
import pytest

from bubbletrack.model import folder_scan
from bubbletrack.model.folder_scan import scan_folder_report, tiff_page_count
from bubbletrack.model.image_io import scan_folder

_IMG = np.zeros((8, 10), np.uint8)


def _write_single(path) -> str:
    ok, buf = cv2.imencode(os.path.splitext(str(path))[1], _IMG)
    assert ok
    buf.tofile(str(path))
    return str(path)


def _write_multipage(path, pages: int) -> str:
    ok, buf = cv2.imencodemulti(".tiff", [_IMG + i for i in range(pages)])
    assert ok
    buf.tofile(str(path))
    return str(path)


class TestTiffPageCount:
    def test_counts_pages_written_by_opencv(self, tmp_path):
        path = _write_multipage(tmp_path / "stack.tif", 3)
        assert tiff_page_count(path) == 3
        assert tiff_page_count(path) == cv2.imcount(path)

    def test_single_page(self, tmp_path):
        assert tiff_page_count(_write_single(tmp_path / "one.tiff")) == 1

    def test_unicode_path(self, tmp_path):
        folder = tmp_path / "实验_µ'"
        folder.mkdir()
        assert tiff_page_count(_write_multipage(folder / "堆栈.tif", 4)) == 4

    @pytest.mark.parametrize("kwargs", [{"byteorder": ">"}, {"bigtiff": True}])
    def test_big_endian_and_bigtiff(self, tmp_path, kwargs):
        tifffile = pytest.importorskip("tifffile")
        path = tmp_path / "stack.tif"
        stack = np.zeros((5, 8, 10), np.uint16)
        tifffile.imwrite(str(path), stack, photometric="minisblack", **kwargs)
        assert tiff_page_count(str(path)) == 5

    def test_not_a_tiff_or_unreadable_is_zero(self, tmp_path):
        assert tiff_page_count(_write_single(tmp_path / "a.png")) == 0
        assert tiff_page_count(str(tmp_path / "missing.tif")) == 0
        empty = tmp_path / "empty.tif"
        empty.write_bytes(b"")
        assert tiff_page_count(str(empty)) == 0
        assert tiff_page_count(str(tmp_path)) == 0  # a directory

    def test_ifd_chain_loop_terminates(self, tmp_path):
        path = tmp_path / "loop.tif"
        data = bytearray(open(_write_single(path), "rb").read())
        first_ifd = struct.unpack("<I", data[4:8])[0]
        n_entries = struct.unpack("<H", data[first_ifd:first_ifd + 2])[0]
        next_ptr = first_ifd + 2 + 12 * n_entries
        data[next_ptr:next_ptr + 4] = struct.pack("<I", first_ifd)  # points to itself
        path.write_bytes(bytes(data))
        assert tiff_page_count(str(path)) == 1

    def test_truncated_chain_counts_readable_pages(self, tmp_path):
        path = tmp_path / "cut.tif"
        data = bytearray(open(_write_single(path), "rb").read())
        first_ifd = struct.unpack("<I", data[4:8])[0]
        n_entries = struct.unpack("<H", data[first_ifd:first_ifd + 2])[0]
        next_ptr = first_ifd + 2 + 12 * n_entries
        data[next_ptr:next_ptr + 4] = struct.pack("<I", len(data) + 1000)  # past EOF
        path.write_bytes(bytes(data))
        assert tiff_page_count(str(path)) == 1


def _basenames(mapping: dict[str, int]) -> dict[str, int]:
    return {os.path.basename(k): v for k, v in mapping.items()}


class TestMultipageFlag:
    def test_report_flags_multipage_files(self, tmp_path):
        _write_multipage(tmp_path / "stack.tif", 5)
        _write_single(tmp_path / "single.tif")
        report = scan_folder_report(str(tmp_path))
        assert _basenames(report.multipage) == {"stack.tif": 5}
        assert any("stack.tif" in w and "5 pages" in w for w in report.warnings)

    def test_page_check_can_be_skipped(self, tmp_path):
        _write_multipage(tmp_path / "stack.tif", 5)
        assert scan_folder_report(str(tmp_path), check_pages=False).multipage == {}

    def test_png_folder_is_not_page_checked(self, tmp_path):
        _write_single(tmp_path / "a.png")
        assert scan_folder_report(str(tmp_path)).multipage == {}

    def test_plain_scan_folder_reads_no_headers(self, tmp_path, monkeypatch):
        """Existing callers must not pay for page counting."""
        _write_multipage(tmp_path / "stack.tif", 5)

        def fail(path):
            raise AssertionError("scan_folder must not count pages")

        monkeypatch.setattr(folder_scan, "tiff_page_count", fail)
        assert len(scan_folder(str(tmp_path))) == 1
