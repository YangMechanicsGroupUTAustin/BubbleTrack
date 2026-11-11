"""Generate PDF summary report of analysis results."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np


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
        "Threshold": f"{state.img_thr:.2f}",
        "Removing Factor": str(state.removing_factor),
        "ROI": f"rows {state.gridx}, cols {state.gridy}",
        "FPS": f"{fps_value:.0f} Hz",
        "Scale": f"{um2px_value:.2f} um/px",
    }


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

    # Parameters section
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Parameters", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)

    params = build_report_parameters(state, fps=fps, um2px=um2px)
    for key, value in params.items():
        pdf.cell(50, 6, key + ":", new_x="RIGHT")
        pdf.cell(0, 6, str(value), new_x="LMARGIN", new_y="NEXT")

    # Statistics
    if state.radius is not None:
        valid = np.asarray(state.radius)
        valid = valid[valid > 0]
    else:
        valid = np.array([])

    pdf.ln(5)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Results Summary", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)

    stats = {
        "Total Frames": str(state.total_frames),
        "Valid Fits": str(len(valid)),
        "Min Radius": f"{valid.min():.2f} px" if len(valid) else "N/A",
        "Max Radius": f"{valid.max():.2f} px" if len(valid) else "N/A",
        "Mean Radius": f"{valid.mean():.2f} px" if len(valid) else "N/A",
    }
    for key, value in stats.items():
        pdf.cell(50, 6, key + ":", new_x="RIGHT")
        pdf.cell(0, 6, str(value), new_x="LMARGIN", new_y="NEXT")

    # Chart image
    if chart_image_path and Path(chart_image_path).exists():
        pdf.ln(5)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "R-t Chart", new_x="LMARGIN", new_y="NEXT")
        pdf.image(chart_image_path, w=170)

    pdf.output(filepath)
