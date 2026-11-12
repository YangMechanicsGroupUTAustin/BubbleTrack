"""Lightweight data containers for batch processing results."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bubbletrack.model.provenance import FitSource


@dataclass(frozen=True)
class FolderResult:
    """Result summary for one batch source (an image folder or a video).

    Attributes
    ----------
    folder_path : Full filesystem path of the folder or video file.
    folder_name : Display name (basename).
    success : Whether the source was processed and exported without error.
    message : Human-readable status, e.g. "120/150 fitted", or the error.
    n_frames : Total frame count of the source (0 if it could not be read).
    n_fitted : Number of frames with radius > 0.
    radius : 1-D array (N,), -1 = unprocessed/failed (empty if never fitted).
    quality_scores : 1-D array (N,); Edge runs store the fit confidence here.
    source_kind : "folder" or "video".
    confidence : (N,) Edge fit confidence (NaN = not scored), None otherwise.
    flags : per-frame frozensets of engine flags (Edge), None otherwise.
    metrics : {FIT_METRIC_KEYS: (N,)} shape metrics (Edge), None otherwise.
    outputs : Files written for this source.
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
    source_kind: str = "folder"
    confidence: np.ndarray | None = None
    flags: list | None = None
    metrics: dict | None = None
    outputs: tuple[str, ...] = ()

    @classmethod
    def failure(cls, path: str, name: str, message: str, *,
                source_kind: str = "folder", n_frames: int = 0) -> FolderResult:
        """A source that produced no results (unreadable, empty, crashed)."""
        return cls(folder_path=path, folder_name=name, success=False,
                   message=message, n_frames=n_frames, n_fitted=0,
                   radius=np.empty(0), quality_scores=np.empty(0),
                   source_kind=source_kind)

    @property
    def has_results(self) -> bool:
        """True when there are per-frame results to load into the main view."""
        return self.radius.size > 0


def mean_score(result: FolderResult) -> float:
    """Mean confidence (Edge) or quality score over fitted frames; NaN if none.

    Area-mode frames are not quality-scored (NaN) and are ignored.
    """
    fitted = result.radius > 0
    if not np.any(fitted):
        return float("nan")
    scores = result.confidence if result.confidence is not None else result.quality_scores
    values = np.asarray(scores, dtype=float)[fitted]
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else float("nan")


def score_label(result: FolderResult) -> str:
    """'Avg conf' for Edge results (engine confidence), else 'Avg Q'."""
    return "Avg conf" if result.confidence is not None else "Avg Q"


def format_score(result: FolderResult) -> str:
    """e.g. 'Avg conf 0.87', 'Avg Q 72%' or 'Avg Q n/a'."""
    value = mean_score(result)
    label = score_label(result)
    if not np.isfinite(value):
        return f"{label} n/a"
    if result.confidence is not None:
        return f"{label} {value:.2f}"
    return f"{label} {value:.0%}"


def restore_into_state(state, result: FolderResult) -> int:
    """Copy *result*'s per-frame arrays into a freshly loaded *state*.

    Writes into the state's result arrays in place (as every fitting path
    does) and returns the number of frames copied.
    """
    n_copy = min(len(state.radius), len(result.radius))
    state.radius[:n_copy] = result.radius[:n_copy]
    if result.circle_fit_par is not None:
        state.circle_fit_par[:n_copy] = result.circle_fit_par[:n_copy]
    if result.circle_xy is not None:
        for i in range(n_copy):
            state.circle_xy[i] = result.circle_xy[i]
    if result.region_count is not None and state.region_count is not None:
        state.region_count[:n_copy] = result.region_count[:n_copy]
    if result.confidence is not None and state.fit_confidence is not None:
        state.fit_confidence[:n_copy] = result.confidence[:n_copy]
    if result.flags is not None and state.fit_flags is not None:
        for i in range(n_copy):
            state.fit_flags[i] = frozenset(result.flags[i])
    if result.metrics is not None and state.fit_metrics is not None:
        for key, values in state.fit_metrics.items():
            if key in result.metrics:
                values[:n_copy] = result.metrics[key][:n_copy]
    if state.fit_source is not None:
        fitted = state.radius[:n_copy] > 0
        state.fit_source[:n_copy] = np.where(fitted, FitSource.AUTO, FitSource.NONE)
    return n_copy


@dataclass(frozen=True)
class BatchResultStore:
    """Immutable collection of all source results from one batch run.

    Attributes
    ----------
    results : Tuple of FolderResult (one per source, failures included).
    timestamp : Batch run start time, e.g. "20260324_153000".
    gridx, gridy : ROI bounds used during the batch.
    sensitivity : Threshold sensitivity used during the batch.
    detection_mode : Detection mode used during the batch.
    error : Why the run itself stopped early ("" when it did not).
    """

    results: tuple[FolderResult, ...]
    timestamp: str
    gridx: tuple[int, int]
    gridy: tuple[int, int]
    sensitivity: float
    detection_mode: str = "intensity"
    error: str = ""

    @property
    def failures(self) -> tuple[FolderResult, ...]:
        return tuple(r for r in self.results if not r.success)

    def error_summary(self) -> list[str]:
        """One line per problem: the run error, then each failed source."""
        lines = [f"Batch stopped: {self.error}"] if self.error else []
        lines.extend(f"{r.folder_name}: {r.message}" for r in self.failures)
        return lines
