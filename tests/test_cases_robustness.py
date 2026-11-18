"""Hard-case library and export metadata edge cases (2026-11-18 bug hunt)."""

from __future__ import annotations

import pytest

from bubbletrack import cases_cli
from bubbletrack.model.cases.report import compare_with_baseline
from bubbletrack.model.cases.schema import CaseFormatError, _number
from bubbletrack.model.export_meta import build_export_metadata
from bubbletrack.model.state import AppState


def test_a_case_that_no_longer_opens_is_a_regression():
    """It is reported under its file name, not its case_id: its correct frames used
    to drop out of the comparison and eval-cases exited 0."""
    baseline = {"cases": [{"case_id": "lic02-12", "bundle": "lic02-12.zip",
                           "results": [{"frame": 3, "outcome": "correct"},
                                       {"frame": 4, "outcome": "wrong"}]}]}
    now = [{"case_id": "lic02-12.zip", "bundle": "lic02-12.zip", "error": "not a readable zip",
            "results": []}]
    diff = compare_with_baseline(now, baseline)
    assert [(r["frame"], r["before"], r["now"]) for r in diff["regressions"]] == [(3, "correct", "error")]
    assert diff["missing_cases"] == [] and diff["new_cases"] == []


def test_a_huge_integer_is_a_format_error():
    with pytest.raises(CaseFormatError):
        _number(10 ** 400, "radius")
    assert _number(12, "radius") == 12.0


def test_tolerances_must_be_finite(tmp_path):
    with pytest.raises(SystemExit) as exc:
        cases_cli.main([str(tmp_path), "--radius-tol-px", "inf"])
    assert exc.value.code == 2


def test_classic_exports_record_no_reference_frame():
    state = AppState(detection_mode="intensity", reference_frame=4)
    meta = build_export_metadata(state, fps=1.0, um2px=1.0, rmax_fit_length=11)
    assert meta["reference_frame"] is None and meta["reference_frame_origin"] is None
    auto = build_export_metadata(AppState(detection_mode="edge", reference_frame=4),
                                 fps=1.0, um2px=1.0, rmax_fit_length=11)
    assert auto["reference_frame"] == 5 and auto["reference_frame_origin"] == "set"
