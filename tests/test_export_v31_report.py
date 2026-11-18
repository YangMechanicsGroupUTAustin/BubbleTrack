"""v3.1 PDF report: detection mode, radius definition and fit-quality summary."""

from __future__ import annotations

import fpdf
import numpy as np
import pytest

from bubbletrack.model.provenance import AREA_RADIUS_FLAG, FitSource
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
    if fields.get("bubble_non_round") and fields.get("detection_mode") == "edge":
        for i in (2, 3, 4, 6, 7, 8, 9):  # as that mode stores its results
            state.fit_flags[i] = frozenset({AREA_RADIUS_FLAG})
    return state


class TestParameters:
    def test_detection_mode_and_radius_definition(self):
        params = build_report_parameters(_state(detection_mode="edge"))
        assert params["Detection Mode"] == "Auto (reference frame)"  # user-facing name of "edge"
        assert params["Radius Definition"] == (
            "least-squares circle (Taubin, one outlier pass) through the bubble outline "
            "(50 % steep edge)")
        params = build_report_parameters(_state(detection_mode="edge", bubble_non_round=True))
        assert params["Radius Definition"] == (
            "area-equivalent sqrt(A/pi) of the 50% iso-contour")

    def test_the_definition_follows_the_results_not_the_setting(self):
        circle = ("least-squares circle (Taubin, one outlier pass) through the bubble outline "
                  "(50 % steep edge)")
        state = _state(detection_mode="edge")
        ticked = AppState(**{**state.__dict__, "bubble_non_round": True})  # nothing refitted
        assert build_report_parameters(ticked)["Radius Definition"] == circle
        ticked.fit_flags[9] = frozenset({AREA_RADIUS_FLAG})  # one frame refitted by area
        text = build_report_parameters(ticked)["Radius Definition"]
        assert text.startswith("mixed: 1 frame area-equivalent") and "6 frames least-squares" in text

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
    state = _state(detection_mode="edge", bubble_non_round=True)
    state.fit_confidence[2:] = 0.8
    state.fit_locked[4] = True
    out = tmp_path / "report.pdf"
    generate_report(str(out), state)
    # PDF string literals escape parentheses.
    text = out.read_bytes().replace(rb"\(", b"(").replace(rb"\)", b")")
    for needle in (b"Detection Mode", b"Radius Definition", b"50% iso-contour",
                   b"Confidence", b"Confidence (median)", b"Locked Frames",
                   b"AUTO 7, MANUAL 1, NONE 2"):
        assert needle in text, needle
