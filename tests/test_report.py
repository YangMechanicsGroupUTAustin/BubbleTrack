"""Tests for PDF report generation (model/report.py)."""

from __future__ import annotations

import os

import numpy as np
import pytest

from bubbletrack.model.report import build_report_parameters, generate_report
from bubbletrack.model.state import AppState, update_state


def _make_state_with_results(n: int = 20) -> AppState:
    """Create an AppState with synthetic radius data."""
    state = AppState(
        folder_path="/tmp/images",
        img_thr=0.45,
        removing_factor=30,
        gridx=(10, 200),
        gridy=(20, 300),
        fps=1_000_000.0,
        um2px=3.2,
        images=tuple(f"img_{i:04d}.tif" for i in range(n)),
    )
    state = state.with_results_initialized(n)
    # Fill with synthetic radii (some invalid)
    radii = np.linspace(50, 100, n)
    radii[0] = -1.0  # invalid
    radii[5] = -1.0  # invalid
    np.copyto(state.radius, radii)
    return state


class TestGenerateReport:
    """Verify PDF report generation."""

    def test_creates_pdf_file(self, tmp_path):
        state = _make_state_with_results()
        out = str(tmp_path / "report.pdf")
        generate_report(out, state)
        assert os.path.exists(out)

    def test_pdf_is_valid(self, tmp_path):
        """A valid PDF starts with the %PDF magic bytes."""
        state = _make_state_with_results()
        out = str(tmp_path / "report.pdf")
        generate_report(out, state)
        with open(out, "rb") as f:
            header = f.read(5)
        assert header == b"%PDF-"

    def test_pdf_has_nonzero_size(self, tmp_path):
        state = _make_state_with_results()
        out = str(tmp_path / "report.pdf")
        generate_report(out, state)
        assert os.path.getsize(out) > 100

    def test_no_radius_data(self, tmp_path):
        """Report should still generate when no results are available."""
        state = AppState(folder_path="/tmp/images")
        out = str(tmp_path / "empty_report.pdf")
        generate_report(out, state)
        assert os.path.exists(out)
        with open(out, "rb") as f:
            assert f.read(5) == b"%PDF-"

    def test_all_invalid_radii(self, tmp_path):
        """Report should handle the case where all radii are -1."""
        state = _make_state_with_results(10)
        state.radius[:] = -1.0
        out = str(tmp_path / "no_valid.pdf")
        generate_report(out, state)
        assert os.path.exists(out)

    def test_with_chart_image(self, tmp_path):
        """Report should embed a chart image when provided."""
        # Create a minimal valid PNG (1x1 white pixel)
        import struct
        import zlib

        def _make_minimal_png(path: str) -> None:
            sig = b"\x89PNG\r\n\x1a\n"
            # IHDR chunk
            ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
            ihdr_crc = zlib.crc32(b"IHDR" + ihdr_data) & 0xFFFFFFFF
            ihdr = struct.pack(">I", 13) + b"IHDR" + ihdr_data + struct.pack(">I", ihdr_crc)
            # IDAT chunk (1x1 RGB white pixel)
            raw = b"\x00\xff\xff\xff"
            compressed = zlib.compress(raw)
            idat_crc = zlib.crc32(b"IDAT" + compressed) & 0xFFFFFFFF
            idat = struct.pack(">I", len(compressed)) + b"IDAT" + compressed + struct.pack(">I", idat_crc)
            # IEND chunk
            iend_crc = zlib.crc32(b"IEND") & 0xFFFFFFFF
            iend = struct.pack(">I", 0) + b"IEND" + struct.pack(">I", iend_crc)
            with open(path, "wb") as f:
                f.write(sig + ihdr + idat + iend)

        chart_path = str(tmp_path / "chart.png")
        _make_minimal_png(chart_path)

        state = _make_state_with_results()
        out = str(tmp_path / "report_with_chart.pdf")
        generate_report(out, state, chart_image_path=chart_path)
        assert os.path.exists(out)
        # PDF with image should be larger than one without
        out_no_chart = str(tmp_path / "report_no_chart.pdf")
        generate_report(out_no_chart, state)
        assert os.path.getsize(out) > os.path.getsize(out_no_chart)

    def test_nonexistent_chart_image_ignored(self, tmp_path):
        """A missing chart_image_path should be silently ignored."""
        state = _make_state_with_results()
        out = str(tmp_path / "report.pdf")
        generate_report(out, state, chart_image_path="/nonexistent/chart.png")
        assert os.path.exists(out)

    def test_panel_fps_and_scale_override_state(self):
        """Values given explicitly (from the panel) win over state defaults."""
        state = _make_state_with_results()
        params = build_report_parameters(state, fps=500_000.0, um2px=1.5)
        assert params["FPS"] == "500000 Hz"
        assert params["Scale"] == "1.50 um/px"

    def test_state_fps_and_scale_used_by_default(self):
        state = _make_state_with_results()
        params = build_report_parameters(state)
        assert params["FPS"] == "1000000 Hz"
        assert params["Scale"] == "3.20 um/px"

    def test_generate_report_accepts_panel_values(self, tmp_path):
        state = _make_state_with_results()
        out = str(tmp_path / "panel.pdf")
        generate_report(out, state, fps=500_000.0, um2px=1.5)
        with open(out, "rb") as f:
            assert f.read(5) == b"%PDF-"

    def test_video_source_in_params(self, tmp_path):
        """When folder_path is empty, video_path should be used as source."""
        state = update_state(
            _make_state_with_results(),
            folder_path="",
            video_path="/tmp/video.mp4",
        )
        out = str(tmp_path / "video_report.pdf")
        generate_report(out, state)
        assert os.path.exists(out)
