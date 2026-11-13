"""Export per-frame results as a table: CSV or Excel (.xlsx).

Both writers share one header and one row builder, so their columns are
identical.  New columns are only ever appended, never reordered, so
existing column positions stay stable for downstream scripts.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Iterator, Mapping, Sequence

import numpy as np

from bubbletrack.model.constants import DEFAULT_RMAX_FIT_LENGTH
from bubbletrack.model.export_fit import FitColumns
from bubbletrack.model.long_paths import os_path
from bubbletrack.model.export_meta import (
    META_SHEET_TITLE,
    excel_value,
    plain_metadata,
    write_json_sidecar,
)
from bubbletrack.model.rmax import locate_rmax, valid_frame_mask

# A data cell: its value plus the format spec used to render it as CSV
# text.  Excel receives the raw value so that cells stay numeric (text
# cells, e.g. Source and Flags, are passed through unchanged).
_Cell = tuple["int | float | str", str]

_FMT_INT = "d"
_FMT_TIME = ".8f"
_FMT_LENGTH = ".4f"


def _excel_number(value: int | float | str) -> int | float | str | None:
    """Return *value* for an Excel cell; NaN/inf (unsupported) become empty."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _build_header(
    *, has_fps: bool, has_scale: bool, has_region_count: bool = False,
) -> list[str]:
    """Build the column header list for CSV/Excel export."""
    header: list[str] = ["Frame"]
    if has_fps:
        header.append("Time_s")
    header.extend(["Radius_px", "Center_Row_px", "Center_Col_px"])
    if has_scale:
        header.extend(["Radius_um", "Center_Row_um", "Center_Col_um"])
    if has_region_count:
        header.append("Region_Count")
    if has_fps:
        header.append("Time_from_Rmax_s")
    return header


def _build_cells(
    frame_idx: int,
    radius_val: float,
    center_row: float,
    center_col: float,
    *,
    fps: float | None,
    um2px: float | None,
    region_count: int | None = None,
    rmax_frame: float | None = None,
) -> list[_Cell]:
    """Build the cells of a single data row for CSV/Excel export.

    *frame_idx* is 0-based; *rmax_frame* is the 1-based Rmax position.
    """
    cells: list[_Cell] = [(frame_idx + 1, _FMT_INT)]
    if fps is not None:
        cells.append((frame_idx / fps, _FMT_TIME))
    cells.extend([
        (radius_val, _FMT_LENGTH),
        (center_row, _FMT_LENGTH),
        (center_col, _FMT_LENGTH),
    ])
    if um2px is not None:
        cells.extend([
            (radius_val * um2px, _FMT_LENGTH),
            (center_row * um2px, _FMT_LENGTH),
            (center_col * um2px, _FMT_LENGTH),
        ])
    if region_count is not None:
        cells.append((region_count, _FMT_INT))
    if fps is not None and rmax_frame is not None:
        # Same expression as the RofT .mat ``t``: (frame - RmaxFrame) * dt.
        cells.append(((frame_idx + 1 - rmax_frame) * (1.0 / fps), _FMT_TIME))
    return cells


def _iter_data_rows(
    radius: np.ndarray,
    circle_fit_par: np.ndarray,
    *,
    fps: float | None,
    um2px: float | None,
    region_count: np.ndarray | None,
    rmax_frame: float | None,
    fit: FitColumns,
) -> Iterator[list[_Cell]]:
    """Yield the cells of every processed frame, skipping ``radius <= 0``."""
    for i in range(len(radius)):
        if radius[i] <= 0:
            continue  # skip unprocessed frames
        yield _build_cells(
            i, float(radius[i]),
            float(circle_fit_par[i, 0]),
            float(circle_fit_par[i, 1]),
            fps=fps, um2px=um2px,
            region_count=(
                None if region_count is None else int(region_count[i])
            ),
            rmax_frame=rmax_frame,
        ) + fit.cells(i)


def _build_table(
    radius: np.ndarray,
    circle_fit_par: np.ndarray,
    *,
    fps: float | None,
    um2px: float | None,
    region_count: np.ndarray | None,
    rmax_fit_length: int,
    fit: FitColumns,
) -> tuple[list[str], Iterator[list[_Cell]], str]:
    """Return ``(header, rows, warning)`` shared by the CSV and Excel writers.

    With *fps*, Rmax is located by :func:`locate_rmax` (the same estimate
    as the RofT ``.mat`` export) to fill ``Time_from_Rmax_s``; *warning*
    carries its caveats, or is empty.  *fit* adds the trailing per-frame
    fit-result columns.
    """
    rmax_frame: float | None = None
    warning = ""
    if fps is not None:
        R = np.asarray(radius, dtype=np.float64).ravel()
        if valid_frame_mask(R).any():
            estimate = locate_rmax(R, rmax_fit_length)
            rmax_frame = estimate.frame
            warning = " ".join(estimate.warnings)
        else:
            rmax_frame = float("nan")  # no valid frame: time is undefined

    header = _build_header(
        has_fps=fps is not None,
        has_scale=um2px is not None,
        has_region_count=region_count is not None,
    ) + fit.header()
    rows = _iter_data_rows(
        radius, circle_fit_par,
        fps=fps, um2px=um2px, region_count=region_count,
        rmax_frame=rmax_frame, fit=fit,
    )
    return header, rows, warning


def export_csv(
    filepath: str,
    radius: np.ndarray,
    circle_fit_par: np.ndarray,
    *,
    fps: float | None = None,
    um2px: float | None = None,
    region_count: np.ndarray | None = None,
    rmax_fit_length: int = DEFAULT_RMAX_FIT_LENGTH,
    metadata: dict | None = None,
    fit_source: np.ndarray | None = None,
    fit_confidence: np.ndarray | None = None,
    fit_locked: np.ndarray | None = None,
    fit_flags: Sequence[frozenset[str]] | None = None,
    fit_metrics: Mapping[str, np.ndarray] | None = None,
) -> str:
    """Export results to CSV format.

    Parameters
    ----------
    filepath : str
        Output file path.
    radius : (N,) array
        Fitted radii in pixels (-1 or <=0 for unprocessed frames).
    circle_fit_par : (N, 2) array
        Circle centres ``[row, col]``.
    fps : float or None
        Frames per second.  If provided, the Time_s column and a trailing
        Time_from_Rmax_s column are included.
    um2px : float or None
        Micrometres per pixel.  If provided, physical-unit columns are included.
    region_count : (N,) int array or None
        Selected region count per frame; adds a Region_Count column.
    rmax_fit_length : int
        Rmax fit window length used for Time_from_Rmax_s.
    metadata : dict or None
        Provenance; when given it is written as JSON to a sidecar file
        ``<name>.meta.json`` next to the CSV.  The CSV itself is unchanged.
    fit_source, fit_confidence, fit_locked, fit_flags, fit_metrics
        Per-frame fit results as stored in ``AppState`` (see
        :func:`~bubbletrack.model.export_fit.fit_result_arrays`).  Each one
        given appends its columns after all the others: ``Source`` (name),
        ``Confidence``, ``Locked`` (0/1), ``Flags`` (``a|b``) and one column
        per metric (``R_Circle_px``, ``Area_px2``, ...).  ValueError if
        one does not have an entry per frame.

    Returns
    -------
    str
        Warnings about the Rmax estimate (empty when none, or without fps).

    Notes
    -----
    Frame and time conventions: ``Frame`` is the 1-based frame number.
    ``Time_s = (Frame - 1) / fps``, so the first image is at t = 0.
    ``Time_from_Rmax_s = (Frame - RmaxFrame) / fps``, where RmaxFrame is
    the 1-based (possibly fractional) Rmax position from
    :func:`locate_rmax`; this equals ``t`` of the RofT ``.mat`` export.
    """
    fit = FitColumns.build(
        len(radius), fit_source=fit_source, fit_confidence=fit_confidence,
        fit_locked=fit_locked, fit_flags=fit_flags, fit_metrics=fit_metrics,
    )
    header, rows, warning = _build_table(
        radius, circle_fit_par,
        fps=fps, um2px=um2px, region_count=region_count,
        rmax_fit_length=rmax_fit_length, fit=fit,
    )

    with open(os_path(filepath), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for cells in rows:
            writer.writerow([format(value, spec) for value, spec in cells])
    if metadata is not None:
        write_json_sidecar(filepath, metadata)
    return warning


def export_excel(
    filepath: str,
    radius: np.ndarray,
    circle_fit_par: np.ndarray,
    *,
    fps: float | None = None,
    um2px: float | None = None,
    region_count: np.ndarray | None = None,
    rmax_fit_length: int = DEFAULT_RMAX_FIT_LENGTH,
    metadata: dict | None = None,
    fit_source: np.ndarray | None = None,
    fit_confidence: np.ndarray | None = None,
    fit_locked: np.ndarray | None = None,
    fit_flags: Sequence[frozenset[str]] | None = None,
    fit_metrics: Mapping[str, np.ndarray] | None = None,
) -> str:
    """Export results to Excel (.xlsx) format with numeric cells.

    Parameters
    ----------
    filepath : str
        Output file path.
    radius : (N,) array
        Fitted radii in pixels (-1 or <=0 for unprocessed frames).
    circle_fit_par : (N, 2) array
        Circle centres ``[row, col]``.
    fps : float or None
        Frames per second.  If provided, the Time_s column and a trailing
        Time_from_Rmax_s column are included.
    um2px : float or None
        Micrometres per pixel.  If provided, physical-unit columns are included.
    region_count : (N,) int array or None
        Selected region count per frame; adds a Region_Count column.
    rmax_fit_length : int
        Rmax fit window length used for Time_from_Rmax_s.
    metadata : dict or None
        Provenance; when given it is written to a second sheet
        ``Metadata`` as Key/Value rows (lists are JSON-encoded).
    fit_source, fit_confidence, fit_locked, fit_flags, fit_metrics
        Per-frame fit results; columns as in :func:`export_csv`.  Numbers
        stay numeric and NaN becomes an empty cell.

    Returns
    -------
    str
        Warnings about the Rmax estimate (empty when none, or without fps).

    Notes
    -----
    Columns and frame/time conventions are identical to :func:`export_csv`.
    """
    # Lazy import to avoid hard dependency
    from openpyxl import Workbook  # type: ignore[import-untyped]

    fit = FitColumns.build(
        len(radius), fit_source=fit_source, fit_confidence=fit_confidence,
        fit_locked=fit_locked, fit_flags=fit_flags, fit_metrics=fit_metrics,
    )
    header, rows, warning = _build_table(
        radius, circle_fit_par,
        fps=fps, um2px=um2px, region_count=region_count,
        rmax_fit_length=rmax_fit_length, fit=fit,
    )

    wb = Workbook()
    ws = wb.active
    ws.title = "Bubble Radius Data"
    ws.append(header)
    for cells in rows:
        # Raw numbers (not CSV text) so Excel treats the cells as numeric.
        ws.append([_excel_number(value) for value, _spec in cells])

    if metadata is not None:
        meta_ws = wb.create_sheet(META_SHEET_TITLE)
        meta_ws.append(["Key", "Value"])
        for key, value in plain_metadata(metadata).items():
            meta_ws.append([key, excel_value(value)])

    wb.save(os_path(filepath))
    return warning
