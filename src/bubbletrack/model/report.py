"""Generate PDF summary report of analysis results."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np

from bubbletrack.model.export_meta import radius_definition
from bubbletrack.model.provenance import FitSource

CONFIDENCE_THRESHOLD = 0.5
"""Fits scored at or above this count as confident in the report summary."""

# Fitted sources first, then frames without a fit.
_SOURCE_ORDER = tuple(s for s in FitSource if s != FitSource.NONE) + (FitSource.NONE,)


def build_report_parameters(
    state,
    *,
    fps: float | None = None,
    um2px: float | None = None,
) -> dict[str, str]:
    """Return the "Parameters" section of the report as label -> text.

    Parameters
    ----------
    state : AppState
        Current application state.
    fps, um2px : float or None
        Post-processing values as shown in the panel.  When given they are
        printed instead of the (possibly stale) values stored in *state*.
    """
    fps_value = state.fps if fps is None else fps
    um2px_value = state.um2px if um2px is None else um2px
    return {
        "Source": state.folder_path or state.video_path or "(none)",
        "Detection Mode": state.detection_mode,
        "Radius Definition": radius_definition(state.detection_mode, state.edge_definition),
        "Threshold": f"{state.img_thr:.2f}",
        "Removing Factor": str(state.removing_factor),
        "ROI": f"rows {state.gridx}, cols {state.gridy}",
        "FPS": f"{fps_value:.0f} Hz",
        "Scale": f"{um2px_value:.2f} um/px",
    }


def _valid_fits(state) -> np.ndarray:
    """Boolean mask of frames with a fitted radius (all False without results)."""
    if state.radius is None:
        return np.zeros(0, dtype=bool)
    radius = np.asarray(state.radius, dtype=np.float64)
    return radius > 0  # NaN compares False


def _confidence_summary(state, valid: np.ndarray) -> dict[str, str]:
    scores = np.array([])
    if state.fit_confidence is not None and len(state.fit_confidence) == len(valid):
        scores = np.asarray(state.fit_confidence, dtype=np.float64)[valid]
        scores = scores[np.isfinite(scores)]
    if len(scores) == 0:
        return {"Confidence (median)": "N/A (not scored)",
                f"Confidence >= {CONFIDENCE_THRESHOLD}": "N/A"}
    confident = int(np.count_nonzero(scores >= CONFIDENCE_THRESHOLD))
    return {
        "Confidence (median)": (
            f"{np.median(scores):.2f} ({len(scores)} of {int(valid.sum())} fits scored)"),
        f"Confidence >= {CONFIDENCE_THRESHOLD}": (
            f"{100.0 * confident / len(scores):.1f}% of scored fits "
            f"({confident} of {len(scores)})"),
    }


def _source_counts(state) -> str:
    if state.fit_source is None or len(state.fit_source) == 0:
        return "N/A"
    codes = np.asarray(state.fit_source)
    parts = [f"{source.name} {int(np.count_nonzero(codes == source))}"
             for source in _SOURCE_ORDER if np.any(codes == source)]
    return ", ".join(parts) or "N/A"


def build_report_quality(state) -> dict[str, str]:
    """Return the "Fit Quality" section of the report as label -> text.

    Confidence statistics use the fitted frames that were scored (the
    Edge engine scores them; other modes leave them unscored).  Flagged
    and locked counts cover every frame.
    """
    valid = _valid_fits(state)
    flagged = sum(1 for flags in (state.fit_flags or ()) if flags)
    locked = 0 if state.fit_locked is None else int(np.count_nonzero(state.fit_locked))
    return {
        **_confidence_summary(state, valid),
        "Flagged Frames": str(flagged),
        "Locked Frames": str(locked),
        "Fit Sources": _source_counts(state),
    }


def _print_section(pdf, title: str, rows: dict[str, str]) -> None:
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    for key, value in rows.items():
        pdf.cell(50, 6, key + ":", new_x="RIGHT")
        pdf.cell(0, 6, str(value), new_x="LMARGIN", new_y="NEXT")


def generate_report(
    filepath: str,
    state,
    chart_image_path: str | None = None,
    *,
    fps: float | None = None,
    um2px: float | None = None,
) -> None:
    """Generate a PDF report with parameters, statistics, and R-t chart.

    Parameters
    ----------
    filepath : str
        Output PDF path.
    state : AppState
        Current application state with results.
    chart_image_path : str or None
        Optional path to a chart screenshot to embed.
    fps, um2px : float or None
        Panel values to print; default to the values stored in *state*.

    Raises
    ------
    ImportError
        If fpdf2 is not installed.
    """
    from fpdf import FPDF  # lazy import to avoid hard dependency

    pdf = FPDF()
    pdf.add_page()

    # Header
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "BubbleTrack Analysis Report", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    pdf.ln(10)

    _print_section(pdf, "Parameters", build_report_parameters(state, fps=fps, um2px=um2px))

    # Statistics
    if state.radius is not None:
        valid = np.asarray(state.radius)
        valid = valid[valid > 0]
    else:
        valid = np.array([])

    stats = {
        "Total Frames": str(state.total_frames),
        "Valid Fits": str(len(valid)),
        "Min Radius": f"{valid.min():.2f} px" if len(valid) else "N/A",
        "Max Radius": f"{valid.max():.2f} px" if len(valid) else "N/A",
        "Mean Radius": f"{valid.mean():.2f} px" if len(valid) else "N/A",
    }
    pdf.ln(5)
    _print_section(pdf, "Results Summary", stats)
    pdf.ln(5)
    _print_section(pdf, "Fit Quality", build_report_quality(state))

    # Chart image
    if chart_image_path and Path(chart_image_path).exists():
        pdf.ln(5)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "R-t Chart", new_x="LMARGIN", new_y="NEXT")
        pdf.image(chart_image_path, w=170)

    pdf.output(filepath)
