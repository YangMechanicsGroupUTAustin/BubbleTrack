"""Export radius data to .mat, CSV and Excel files.

This module is the public entry point for every export.  The pieces live
in focused modules and are re-exported here, so callers keep importing
from ``bubbletrack.model.export``:

- ``export_meta``  -- provenance metadata (struct / sheet / JSON sidecar),
- ``export_fit``   -- per-frame fit results (source, confidence, flags, ...),
- ``rmax``         -- the Rmax estimate shared by RofT and the tables,
- ``export_table`` -- the CSV and Excel writers.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import scipy.io
from scipy.io import savemat

from bubbletrack.model.export_fit import FitColumns, fit_result_arrays
from bubbletrack.model.export_meta import (
    META_MAT_VARIABLE,
    META_SHEET_TITLE,
    META_SIDECAR_SUFFIX,
    build_export_metadata,
    mat_metadata,
    radius_definition,
)
from bubbletrack.model.export_table import export_csv, export_excel
from bubbletrack.model.long_paths import os_path
from bubbletrack.model.rmax import (
    RmaxEstimate,
    locate_rmax,
    rmax_fit_half_window,
    valid_frame_mask,
)

__all__ = [
    "META_MAT_VARIABLE",
    "META_SHEET_TITLE",
    "META_SIDECAR_SUFFIX",
    "RmaxEstimate",
    "build_export_metadata",
    "export_csv",
    "export_excel",
    "export_r_data",
    "export_rof_t_data",
    "fit_result_arrays",
    "locate_rmax",
    "radius_definition",
    "rmax_fit_half_window",
    "safe_loadmat",
]


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
        data = scipy.io.loadmat(os_path(filepath))
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
    *,
    metadata: dict | None = None,
    fit_source: np.ndarray | None = None,
    fit_confidence: np.ndarray | None = None,
    fit_locked: np.ndarray | None = None,
    fit_flags: Sequence[frozenset[str]] | None = None,
    fit_metrics: Mapping[str, np.ndarray] | None = None,
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
    region_count : (N,) int array or None
        Selected region count per frame, saved as ``RegionCount``.
    metadata : dict or None
        Provenance (detection settings, version, ...).  When given it is
        saved as the struct variable ``BubbleTrackMeta``.
    fit_source, fit_confidence, fit_locked, fit_flags, fit_metrics
        Per-frame fit results as stored in ``AppState`` (see
        :func:`fit_result_arrays`); each one given adds its variables:
        ``FitSource`` (int8 codes) with ``FitSourceNames`` (cellstr, so
        ``FitSourceNames{FitSource(k) + 1}`` names frame k),
        ``FitConfidence``, ``FitLocked`` (logical), ``FitFlags`` (cellstr
        of ``a|b`` strings) and the struct ``FitMetrics``.

    Raises
    ------
    ValueError
        If a per-frame fit result does not have one entry per frame.
    """
    fit = FitColumns.build(
        len(radius), fit_source=fit_source, fit_confidence=fit_confidence,
        fit_locked=fit_locked, fit_flags=fit_flags, fit_metrics=fit_metrics,
    )
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
    data.update(fit.mat_variables())
    if metadata is not None:
        data[META_MAT_VARIABLE] = mat_metadata(metadata)
    savemat(os_path(save_path), data)


def export_rof_t_data(
    save_path: str,
    radius: np.ndarray,
    um2px: float,
    fps: float,
    rmax_fit_length: int,
    *,
    metadata: dict | None = None,
    confidence: np.ndarray | None = None,
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
    metadata : dict or None
        Provenance; when given it is saved as the struct ``BubbleTrackMeta``.
    confidence : (N,) array or None
        Per-frame fit confidence; when given it is saved as ``Confidence``,
        aligned with ``t`` and ``R``.  Invalid frames get NaN, and so does
        the inserted Rmax sample unless it is a measured frame.

    Returns
    -------
    (success, message) : (bool, str)
        On failure, *message* explains why nothing was written.  On
        success, *message* is empty or holds warnings about the Rmax
        estimate (e.g. a peak near the data boundary); the file is
        written either way.

    Raises
    ------
    ValueError
        If *confidence* does not have one entry per frame.

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
    if confidence is not None:
        confidence = np.asarray(confidence, dtype=np.float64).ravel()
        if len(confidence) != len(R):
            raise ValueError(
                f"confidence has {len(confidence)} entries, expected {len(R)}")

    valid_mask = valid_frame_mask(R)
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

    data = {
        "t": t_out.reshape(-1, 1),
        "R": R_out.reshape(-1, 1),
        "RmaxAll": float(rmax_all_phys),
        "RmaxTimeLoc": float(rmax_time_loc_phys),
        # Conversion parameters, stored as MATLAB doubles.
        "fps": float(fps),
        "um2px": float(um2px),
        "RmaxFitLength": float(rmax_fit_length),
    }
    if confidence is not None:
        data["Confidence"] = _rof_t_confidence(
            confidence, valid_mask, t, rmax_loc,
        ).reshape(-1, 1)
    if metadata is not None:
        data[META_MAT_VARIABLE] = mat_metadata(metadata)
    savemat(os_path(save_path), data)

    return True, " ".join(estimate.warnings)


def _rof_t_confidence(
    confidence: np.ndarray, valid_mask: np.ndarray, t: np.ndarray, rmax_loc: float,
) -> np.ndarray:
    """Align per-frame confidence with the RofT ``t``/``R`` samples.

    The sample at t = 0 is the Rmax estimate: it takes the confidence of
    the frame it replaces (a discrete peak), or NaN for a fitted vertex
    between frames, which no single frame measured.
    """
    conf = np.where(valid_mask, confidence, np.nan)
    at_peak = conf[t == rmax_loc]
    return np.concatenate([
        conf[t < rmax_loc],
        [at_peak[0] if len(at_peak) else np.nan],
        conf[t > rmax_loc],
    ])

