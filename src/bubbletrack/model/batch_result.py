"""Lightweight data containers for batch processing results."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FolderResult:
    """Result summary for a single experiment folder.

    Attributes
    ----------
    folder_path : Full filesystem path.
    folder_name : Display name (basename).
    success : Whether the folder was processed without error.
    message : Human-readable status, e.g. "120/150 fitted".
    n_frames : Total image count in the folder.
    n_fitted : Number of frames with radius > 0.
    radius : 1-D array (N,), -1 = unprocessed/failed.
    quality_scores : 1-D array (N,), 0.0 = invalid.
    """

    folder_path: str
    folder_name: str
    success: bool
    message: str
    n_frames: int
    n_fitted: int
    radius: np.ndarray
    quality_scores: np.ndarray
    # Optional so older callers keep working; needed to restore a folder fully.
    circle_fit_par: np.ndarray | None = None  # (N, 2) [row_c, col_c]
    circle_xy: list | None = None  # per-frame (M, 2) edge points or None
    region_count: np.ndarray | None = None  # (N,)


@dataclass(frozen=True)
class BatchResultStore:
    """Immutable collection of all folder results from one batch run.

    Attributes
    ----------
    results : Tuple of FolderResult (one per folder).
    timestamp : Batch run start time, e.g. "20260324_153000".
    gridx, gridy : ROI bounds used during the batch.
    sensitivity : Threshold sensitivity used during the batch.
    """

    results: tuple[FolderResult, ...]
    timestamp: str
    gridx: tuple[int, int]
    gridy: tuple[int, int]
    sensitivity: float
