"""Locate the maximum bubble radius (Rmax) with a local quadratic fit.

Shared by the RofT ``.mat`` export and the ``Time_from_Rmax_s`` column of
the CSV/Excel export, so both agree on where t = 0 is.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bubbletrack.model.constants import (
    RMAX_OUTLIER_NEIGHBOURS,
    RMAX_OUTLIER_TOLERANCE,
)


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


def valid_frame_mask(radius: np.ndarray) -> np.ndarray:
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
        f"data boundary ({' and '.join(sides)} of the valid frames); the fit "
        f"window was truncated, so Rmax and t=0 may be inaccurate."
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
    valid_mask = valid_frame_mask(R)
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
