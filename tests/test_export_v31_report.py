"""v3.1 PDF report: detection mode, radius definition and fit-quality summary."""

from __future__ import annotations

import fpdf
import numpy as np
import pytest

from bubbletrack.model.provenance import FitSource
from bubbletrack.model.report import (
    build_report_parameters,
    build_report_quality,
    generate_report,
)
from bubbletrack.model.state import AppState


def _state(n: int = 10, **fields) -> AppState:
    state = AppState(
        folder_path="/data/run1",
        images=tuple(f"img_{i:03d}.tif" for i in range(n)),
        **fields,
    ).with_results_initialized(n)
    state.radius[:] = np.linspace(20.0, 40.0, n)
    state.radius[[0, 1]] = -1.0  # two failed frames
    state.fit_source[:] = FitSource.AUTO
    state.fit_source[[0, 1]] = FitSource.NONE
    state.fit_source[5] = FitSource.MANUAL
    return state


class TestParameters:
    def test_detection_mode_and_radius_definition(self):
        params = build_report_parameters(_state(detection_mode="edge"))
        assert params["Detection Mode"] == "Auto (reference frame)"  # user-facing name of "edge"
        assert params["Radius Definition"] == (
            "area-equivalent sqrt(A/pi) of the 50% iso-contour")

    def test_area_mode(self):
        params = build_report_parameters(_state(detection_mode="area"))
        assert params["Radius Definition"] == "area-equivalent (dark regions)"


class TestQuality:
    def test_confidence_summary_over_scored_fits(self):
        state = _state()
        # Frames 0-1 failed (ignored); 8 valid fits, 6 of them scored.
        state.fit_confidence[:] = [0.9, 0.9, 0.2, 0.4, 0.6, 0.8, 0.7, 0.9, np.nan, np.nan]
        quality = build_report_quality(state)
        assert quality["Confidence (median)"] == "0.65 (6 of 8 fits scored)"
        assert quality["Confidence >= 0.5"] == "66.7% of scored fits (4 of 6)"

    def test_unscored(self):
        quality = build_report_quality(_state())
        assert quality["Confidence (median)"] == "N/A (not scored)"
        assert quality["Confidence >= 0.5"] == "N/A"

    def test_flagged_locked_and_sources(self):
        state = _state()
        state.fit_flags[3] = frozenset({"fragmented"})
        state.fit_flags[4] = frozenset({"low_contrast", "clipped"})
        state.fit_locked[[5, 6, 7]] = True
        quality = build_report_quality(state)
        assert quality["Flagged Frames"] == "2"
        assert quality["Locked Frames"] == "3"
        assert quality["Fit Sources"] == "AUTO 7, MANUAL 1, NONE 2"

    def test_state_without_results(self):
        quality = build_report_quality(AppState())
        assert quality["Flagged Frames"] == "0"
        assert quality["Locked Frames"] == "0"
        assert quality["Fit Sources"] == "N/A"
        assert quality["Confidence (median)"] == "N/A (not scored)"


class _PlainPDF(fpdf.FPDF):
    """Uncompressed, so the test can find the report text in the bytes."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.set_compression(False)


@pytest.fixture
def plain_pdf(monkeypatch):
    monkeypatch.setattr(fpdf, "FPDF", _PlainPDF)


def test_report_prints_the_new_sections(tmp_path, plain_pdf):
    state = _state(detection_mode="edge")
    state.fit_confidence[2:] = 0.8
    state.fit_locked[4] = True
    out = tmp_path / "report.pdf"
    generate_report(str(out), state)
    # PDF string literals escape parentheses.
    text = out.read_bytes().replace(rb"\(", b"(").replace(rb"\)", b")")
    for needle in (b"Detection Mode", b"Radius Definition", b"50% iso-contour",
                   b"Fit Quality", b"Confidence (median)", b"Locked Frames",
                   b"AUTO 7, MANUAL 1, NONE 2"):
        assert needle in text, needle
