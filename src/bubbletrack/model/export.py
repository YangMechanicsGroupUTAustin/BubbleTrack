"""Export radius data to .mat, CSV and Excel files."""

from __future__ import annotations

import csv
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io
from scipy.io import savemat

from bubbletrack.model.constants import (
    DEFAULT_RMAX_FIT_LENGTH,
    RMAX_OUTLIER_NEIGHBOURS,
    RMAX_OUTLIER_TOLERANCE,
)


def safe_loadmat(filepath: str, expected_keys: frozenset[str]) -> dict:
    """Load a ``.mat`` file with structure validation.

    Parameters
    ----------
    filepath : str
        Path to the ``.mat`` file.
    expected_keys : frozenset[str]
        Set of keys that must be present in the loaded data.

    Returns
    -------
    dict
        The loaded data dictionary.

    Raises
    ------
    ValueError
        If the file is invalid or missing expected keys.
    """
    try:
        data = scipy.io.loadmat(filepath)
    except Exception as e:
        raise ValueError(f"Invalid MAT file '{filepath}': {e}") from e

    missing = expected_keys - set(data.keys())
    if missing:
        raise ValueError(f"MAT file missing keys: {missing}")
    return data


def export_r_data(
    save_path: str,
    radius: np.ndarray,
    centers: np.ndarray,
    edge_pts: list,
    region_count: np.ndarray | None = None,
) -> None:
    """Save raw fitting results to a ``.mat`` file.

    Parameters
    ----------
    save_path : str
        Full file path for the output ``.mat`` file.
    radius : (N,) array
        Fitted radii in pixels (-1 for unprocessed frames).
    centers : (N, 2) array
        Circle centres ``[row, col]``.
    edge_pts : list of (M, 2) arrays or None
        Edge coordinates per frame.
    """
    # Replace None entries with empty arrays for savemat compatibility
    cleaned_edges = [
        (e if e is not None else np.empty((0, 2))) for e in edge_pts
    ]

    data = {
        "Radius": radius.reshape(1, -1),
        "CircleFitPar": centers,
        "CircleXY": np.array(cleaned_edges, dtype=object),
    }
    if region_count is not None:
        data["RegionCount"] = region_count.reshape(1, -1)
    savemat(save_path, data)


def rmax_fit_half_window(rmax_fit_length: int) -> int:
    """Return the number of frames on each side of the peak in the Rmax fit.

    The rule ``(L - 1) // 2`` is monotonic in L: an odd L gives a window of
    exactly L frames, an even L gives L - 1 frames (never more than L).
    L < 1 gives 0, i.e. no fit (the discrete maximum is used).
    """
    return max(0, (int(rmax_fit_length) - 1) // 2)


@dataclass(frozen=True)
class RmaxEstimate:
    """Location and value of the maximum bubble radius.

    Frame positions are **1-based** (frame 1 is the first image), matching
    the ``Frame`` column of the CSV/XLSX export and the MATLAB convention of
    the RofT ``.mat`` export.

    Attributes
    ----------
    frame : float
        1-based frame position of Rmax.  Fractional when the parabola
        vertex is accepted, otherwise equal to ``peak_frame``.
    radius : float
        Rmax in pixels (parabola vertex value or discrete maximum).
    peak_frame : int
        1-based frame of the discrete maximum among valid frames.
    window : tuple[int, int]
        Inclusive 1-based fit window, clamped to ``[1, N]``.
    fitted : bool
        True when the parabola vertex was accepted.
    warnings : tuple[str, ...]
        Human-readable caveats about the estimate; empty when none apply.
    """

    frame: float
    radius: float
    peak_frame: int
    window: tuple[int, int]
    fitted: bool
    warnings: tuple[str, ...] = ()


def _valid_frame_mask(radius: np.ndarray) -> np.ndarray:
    """Return True for frames with a usable radius (> 0 and not NaN)."""
    return (radius > 0) & ~np.isnan(radius)


def _boundary_warning(
    peak_frame: int, valid_indices: np.ndarray, half: int,
) -> str | None:
    """Warn when the peak is at, or within half a window of, a data edge.

    The edges are the first and last *valid* frames.  Closer than *half*
    frames, the fit window runs past the data and is truncated; a peak on
    the edge itself is always flagged because the true maximum may lie
    outside the recording.
    """
    margin = max(half, 1)
    first_frame = int(valid_indices[0]) + 1
    last_frame = int(valid_indices[-1]) + 1
    sides = [
        side for side, distance in (
            ("start", peak_frame - first_frame),
            ("end", last_frame - peak_frame),
        )
        if distance < margin
    ]
    if not sides:
        return None
    return (
        f"Rmax (frame {peak_frame}) is at or within half a fit window of the "
        f"{' and '.join(sides)} boundary of the valid frames; the fit window "
        f"was truncated, so Rmax and t=0 may be inaccurate."
    )


def _outlier_warning(
    radius: np.ndarray, valid_indices: np.ndarray, peak_pos: int,
) -> str | None:
    """Warn when the peak frame towers over its nearest valid neighbours.

    The reference is the median of up to ``RMAX_OUTLIER_NEIGHBOURS`` valid
    frames on each side of the peak (failed frames are skipped).  *peak_pos*
    indexes into *valid_indices*.
    """
    n = RMAX_OUTLIER_NEIGHBOURS
    neighbours = np.concatenate([
        valid_indices[max(0, peak_pos - n):peak_pos],
        valid_indices[peak_pos + 1:peak_pos + 1 + n],
    ])
    if len(neighbours) == 0:
        return None
    peak_value = float(radius[valid_indices[peak_pos]])
    median = float(np.median(radius[neighbours]))
    if peak_value <= median * (1.0 + RMAX_OUTLIER_TOLERANCE):
        return None
    return (
        f"Rmax frame {int(valid_indices[peak_pos]) + 1} ({peak_value:.2f} px) "
        f"exceeds the median of its {len(neighbours)} nearest valid frames "
        f"({median:.2f} px) by {peak_value / median - 1.0:.0%} (limit "
        f"{RMAX_OUTLIER_TOLERANCE:.0%}); it may be a spurious detection that "
        f"misplaces t=0."
    )


def _fit_parabola_vertex(
    frames: np.ndarray,
    values: np.ndarray,
    window: tuple[int, int],
) -> tuple[float, float] | None:
    """Fit a parabola and return its vertex ``(frame, value)``, or None.

    The vertex is rejected (None) when fewer than 3 points are given, the
    parabola is not concave, or the vertex falls outside *window*.
    """
    if len(frames) < 3:
        return None
    try:
        p = np.polyfit(frames, values, 2)
    except (np.linalg.LinAlgError, ValueError):
        return None
    if p[0] >= 0:
        return None  # not concave
    vertex_frame = -p[1] / (2.0 * p[0])
    vertex_value = (4 * p[0] * p[2] - p[1] ** 2) / (4 * p[0])
    # Only trust the vertex inside the window it was fitted on.
    if vertex_frame < window[0] or vertex_frame > window[1]:
        return None
    return float(vertex_frame), float(vertex_value)


def locate_rmax(radius: np.ndarray, rmax_fit_length: int) -> RmaxEstimate:
    """Locate the maximum radius with a local quadratic fit.

    The discrete maximum among valid frames (radius > 0, not NaN) is
    refined by fitting a parabola to the valid frames of a window of
    ``rmax_fit_half_window(L)`` frames on each side, clamped to the data.
    The vertex is accepted only if the parabola is concave and the vertex
    lies inside that window; otherwise the discrete maximum is used.

    Parameters
    ----------
    radius : (N,) array
        Radii in pixels; invalid frames (<= 0 or NaN) are ignored.
    rmax_fit_length : int
        Fit window length L (see :func:`rmax_fit_half_window`).

    Returns
    -------
    RmaxEstimate
        Positions are 1-based frame numbers.

    Raises
    ------
    ValueError
        If *radius* contains no valid frame.
    """
    R = np.asarray(radius, dtype=np.float64).ravel()
    valid_mask = _valid_frame_mask(R)
    valid_indices = np.flatnonzero(valid_mask)  # 0-based
    if len(valid_indices) == 0:
        raise ValueError("No valid frames to locate Rmax")

    peak_pos = int(np.argmax(R[valid_indices]))  # position among valid frames
    peak_idx = int(valid_indices[peak_pos])
    peak_frame = peak_idx + 1  # 1-based
    half = rmax_fit_half_window(rmax_fit_length)
    window = (max(1, peak_frame - half), min(len(R), peak_frame + half))

    window_frames = np.arange(window[0], window[1] + 1)
    fit_frames = window_frames[valid_mask[window_frames - 1]]
    vertex = _fit_parabola_vertex(fit_frames, R[fit_frames - 1], window)

    warnings = tuple(
        w for w in (
            _boundary_warning(peak_frame, valid_indices, half),
            _outlier_warning(R, valid_indices, peak_pos),
        )
        if w is not None
    )
    if vertex is None:
        return RmaxEstimate(
            frame=float(peak_frame), radius=float(R[peak_idx]),
            peak_frame=peak_frame, window=window, fitted=False,
            warnings=warnings,
        )
    return RmaxEstimate(
        frame=vertex[0], radius=vertex[1],
        peak_frame=peak_frame, window=window, fitted=True,
        warnings=warnings,
    )


def export_rof_t_data(
    save_path: str,
    radius: np.ndarray,
    um2px: float,
    fps: float,
    rmax_fit_length: int,
) -> tuple[bool, str]:
    """Convert pixel radii to physical units and save to a ``.mat`` file.

    Parameters
    ----------
    save_path : str
        Full file path for the output ``.mat`` file.
    radius : (N,) array
        Fitted radii in pixels.  Frames with radius <= 0 or NaN (e.g. the
        -1 "unprocessed/failed" sentinel) are invalid: they are exported as
        NaN in ``R`` while keeping their sample in ``t``.
    um2px : float
        Micrometres per pixel.
    fps : float
        Frames per second.
    rmax_fit_length : int
        Window length L for quadratic Rmax fitting.  The window spans
        ``(L - 1) // 2`` frames on each side of the discrete peak, so an
        odd L uses L points and an even L uses L - 1 points.

    Returns
    -------
    (success, message) : (bool, str)
        On failure, *message* explains why nothing was written.  On
        success, *message* is empty or holds warnings about the Rmax
        estimate (e.g. a peak near the data boundary); the file is
        written either way.

    Notes
    -----
    ``t`` is measured from Rmax: frame k (1-based) maps to
    ``(k - RmaxFrame) / fps``, and the Rmax sample is inserted at t = 0.
    ``RmaxTimeLoc`` is the 1-based frame position of Rmax divided by fps.
    The scalars ``fps``, ``um2px`` and ``RmaxFitLength`` record the
    conversion parameters.
    """
    deltax = um2px
    deltat = 1.0 / fps

    R = np.asarray(radius, dtype=np.float64).ravel()
    t = np.arange(1, len(R) + 1, dtype=np.float64)  # 1-based frame numbers

    valid_mask = _valid_frame_mask(R)
    n_valid = int(valid_mask.sum())
    if n_valid < 3:
        return False, f"Not enough valid frames (need >= 3, got {n_valid})"

    estimate = locate_rmax(R, rmax_fit_length)
    rmax_loc = estimate.frame
    rmax_all = estimate.radius

    # Build output arrays — insert Rmax at the peak and shift time.
    # Failed frames are exported as NaN (not a scaled -1 sentinel) so that
    # MATLAB plots and statistics skip them; their time samples are kept.
    mask_before = t < rmax_loc
    mask_after = t > rmax_loc
    R_valid = np.where(valid_mask, R, np.nan)

    R_out = np.concatenate([
        R_valid[mask_before], [rmax_all], R_valid[mask_after],
    ]) * deltax
    t_out = np.concatenate([
        t[mask_before] - rmax_loc,
        [0.0],
        t[mask_after] - rmax_loc,
    ]) * deltat
    rmax_all_phys = rmax_all * deltax
    rmax_time_loc_phys = rmax_loc * deltat

    savemat(save_path, {
        "t": t_out.reshape(-1, 1),
        "R": R_out.reshape(-1, 1),
        "RmaxAll": float(rmax_all_phys),
        "RmaxTimeLoc": float(rmax_time_loc_phys),
        # Conversion parameters, stored as MATLAB doubles.
        "fps": float(fps),
        "um2px": float(um2px),
        "RmaxFitLength": float(rmax_fit_length),
    })

    return True, " ".join(estimate.warnings)


# ---------------------------------------------------------------------------
# CSV / Excel export
# ---------------------------------------------------------------------------

# A data cell: its numeric value plus the format spec used to render it as
# CSV text.  Excel receives the raw value so that cells stay numeric.
_Cell = tuple["int | float", str]

_FMT_INT = "d"
_FMT_TIME = ".8f"
_FMT_LENGTH = ".4f"


def _build_header(
    *, has_fps: bool, has_scale: bool, has_region_count: bool = False,
) -> list[str]:
    """Build the column header list for CSV/Excel export.

    New columns are only ever appended, so existing column positions stay
    stable for downstream scripts.
    """
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
        )


def _build_table(
    radius: np.ndarray,
    circle_fit_par: np.ndarray,
    *,
    fps: float | None,
    um2px: float | None,
    region_count: np.ndarray | None,
    rmax_fit_length: int,
) -> tuple[list[str], Iterator[list[_Cell]], str]:
    """Return ``(header, rows, warning)`` shared by the CSV and Excel writers.

    With *fps*, Rmax is located by :func:`locate_rmax` (the same estimate
    as the RofT ``.mat`` export) to fill ``Time_from_Rmax_s``; *warning*
    carries its caveats, or is empty.
    """
    rmax_frame: float | None = None
    warning = ""
    if fps is not None:
        R = np.asarray(radius, dtype=np.float64).ravel()
        if _valid_frame_mask(R).any():
            estimate = locate_rmax(R, rmax_fit_length)
            rmax_frame = estimate.frame
            warning = " ".join(estimate.warnings)
        else:
            rmax_frame = float("nan")  # no valid frame: time is undefined

    header = _build_header(
        has_fps=fps is not None,
        has_scale=um2px is not None,
        has_region_count=region_count is not None,
    )
    rows = _iter_data_rows(
        radius, circle_fit_par,
        fps=fps, um2px=um2px, region_count=region_count,
        rmax_frame=rmax_frame,
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
    header, rows, warning = _build_table(
        radius, circle_fit_par,
        fps=fps, um2px=um2px, region_count=region_count,
        rmax_fit_length=rmax_fit_length,
    )

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for cells in rows:
            writer.writerow([format(value, spec) for value, spec in cells])
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

    header, rows, warning = _build_table(
        radius, circle_fit_par,
        fps=fps, um2px=um2px, region_count=region_count,
        rmax_fit_length=rmax_fit_length,
    )

    wb = Workbook()
    ws = wb.active
    ws.title = "Bubble Radius Data"
    ws.append(header)
    for cells in rows:
        # Raw numbers (not CSV text) so Excel treats the cells as numeric.
        ws.append([value for value, _spec in cells])

    wb.save(filepath)
    return warning
