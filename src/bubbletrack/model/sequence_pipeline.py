"""Bridge between the app's per-frame results and the sequence solver.

Fit All leaves one single-frame result per frame.  The solver tracks the
bubble through them and re-measures inconsistent frames with a circle
prior.  This module turns the result arrays into solver inputs and runs
those re-measurements with the Edge engine, keeping the full fits (outline,
metrics, flags): the solver itself only returns circles.

No Qt and no state mutation here, so a solve can run on a worker thread
from a snapshot taken on the GUI thread.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterable

import numpy as np

from bubbletrack.model.edge_pipeline import EdgeFrameFit, EdgeJob, fit_edge
from bubbletrack.model.provenance import user_confirmed_mask
from bubbletrack.model.sequence import FrameSolution, Observation

Circle = tuple[float, float, float]  # (rc, cc, R), 1-based full-image
_NAN = float("nan")


def observations(state) -> list[Observation]:
    """One solver observation per frame from the stored results."""
    n = len(state.radius)
    conf = state.fit_confidence if state.fit_confidence is not None else np.full(n, np.nan)
    flags = state.fit_flags if state.fit_flags is not None else [frozenset()] * n
    out = []
    for i in range(n):
        radius = float(state.radius[i])
        rc, cc = state.circle_fit_par[i]
        out.append(Observation(radius if radius > 0 else _NAN, rc, cc, conf[i], flags[i]))
    return out


def anchors(state) -> dict[int, Circle]:
    """Manual (user-confirmed) and locked frames: the solver keeps them and
    tracks from them.

    A locked frame without a result is an accepted "no bubble here".
    """
    n = len(state.radius)
    source = state.fit_source if state.fit_source is not None else np.zeros(n, np.int8)
    locked = state.fit_locked if state.fit_locked is not None else np.zeros(n, bool)
    out: dict[int, Circle] = {}
    for i in np.flatnonzero(locked | user_confirmed_mask(source)):
        radius = float(state.radius[i])
        if radius > 0:
            rc, cc = state.circle_fit_par[i]
            out[int(i)] = (float(rc), float(cc), radius)
        elif locked[i]:
            out[int(i)] = (_NAN, _NAN, _NAN)
    return out


def retrack_frames(state, queued: Iterable[int]) -> frozenset[int]:
    """Frames a re-track after an edit may change.

    Only frames that do not look good: those without a result and those the
    Review queue lists (*queued*).  Every other frame, and every manual,
    drawn-circle or locked frame (a locked empty frame is an accepted "no
    bubble here"), stays exactly as it is.
    """
    n = len(state.radius)
    radius = np.asarray(state.radius, dtype=float)
    unfitted = set(np.flatnonzero(~(np.isfinite(radius) & (radius > 0))).tolist())
    candidates = unfitted | {int(i) for i in queued if 0 <= i < n}
    locked = state.fit_locked if state.fit_locked is not None else np.zeros(n, bool)
    kept = np.asarray(locked, bool) | user_confirmed_mask(state.fit_source, n)
    return frozenset(i for i in candidates if not kept[i])


class TrackingStopped(BaseException):
    """Raised inside a solve when the user pressed Stop.

    A BaseException (like ``asyncio.CancelledError``): the solver survives a
    failing re-detection (``except Exception``), but a stop must end the solve.
    """


class Redetector:
    """``redetect(idx, prior)`` for the solver, measuring with the Edge engine.

    Reads frames through *read_raw* (give it a private reader: readers are
    not thread-safe) and remembers every fit, so the circle the solver picks
    can be stored with its outline and metrics.
    """

    def __init__(self, read_raw: Callable[[int], np.ndarray], gridx: tuple[int, int],
                 gridy: tuple[int, int], job: EdgeJob,
                 stop_requested: Callable[[], bool] = lambda: False) -> None:
        self._read_raw = read_raw
        self._gridx, self._gridy = gridx, gridy
        self._job = job
        self._stop_requested = stop_requested
        self._fits: dict[int, list[EdgeFrameFit]] = {}

    def __call__(self, idx: int, prior: Circle | None) -> Observation:
        if self._stop_requested():
            raise TrackingStopped
        # The prior replaces the seed click: on a moving bubble the click
        # is far from the bubble in most frames.
        params = dataclasses.replace(self._job.params, prior=prior, seed=None)
        fit = fit_edge(self._read_raw(idx), self._gridx, self._gridy,
                       dataclasses.replace(self._job, params=params))
        self._fits.setdefault(idx, []).append(fit)
        if fit.radius <= 0:
            return Observation(_NAN, _NAN, _NAN, fit.confidence, fit.flags)
        return Observation(fit.radius, fit.rc, fit.cc, fit.confidence, fit.flags)

    def fit_for(self, idx: int, solution: FrameSolution) -> EdgeFrameFit | None:
        """The fit behind the solver's circle for frame *idx*, if it made one."""
        target = (solution.radius, solution.rc, solution.cc)
        for fit in reversed(self._fits.get(idx, ())):
            if fit.radius > 0 and np.allclose((fit.radius, fit.rc, fit.cc), target):
                return fit
        return None
