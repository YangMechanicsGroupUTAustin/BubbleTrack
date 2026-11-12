"""App states with results for the hard-case tests, built without Qt.

``fit_all`` stores results the way Fit All does (``radius``, centre, source,
and for Auto mode confidence and flags), so cases exported from these states
look like cases exported from the app.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from bubbletrack.model import edge_pipeline
from bubbletrack.model.frame_pipeline import FrameParams, fit_image_file
from bubbletrack.model.image_io import imread_unicode, scan_folder
from bubbletrack.model.provenance import FitSource
from bubbletrack.model.removing_factor import compute_removing_factor
from bubbletrack.model.state import AppState


def folder_state(folder: Path, *, roi=None, mode: str = "intensity", **params) -> AppState:
    images = scan_folder(str(folder))
    h, w = imread_unicode(images[0]).shape[:2]
    gridx, gridy = roi if roi is not None else ((1, h), (1, w))
    state = AppState(images=tuple(images), folder_path=str(folder), gridx=gridx, gridy=gridy,
                     detection_mode=mode, **params)
    return state.with_results_initialized(len(images))


def max_radius(state: AppState) -> float:
    return float(max(imread_unicode(state.images[0]).shape[:2]))


def _legacy_params(state: AppState) -> FrameParams:
    return FrameParams(
        sensitivity=state.img_thr, gridx=state.gridx, gridy=state.gridy,
        removing_factor=compute_removing_factor(state.removing_factor, state.gridx, state.gridy),
        edges=tuple(state.bubble_cross_edges), mode=state.detection_mode,
        gaussian_sigma=state.gaussian_sigma, clahe_clip=state.clahe_clip,
        max_radius=max_radius(state))


def edge_job(state: AppState, reference: int | None) -> edge_pipeline.EdgeJob:
    ref = edge_pipeline.read_raw(state.images[reference]) if reference is not None else None
    return edge_pipeline.EdgeJob(params=edge_pipeline.params_from_state(state), reference=ref,
                                 max_radius=max_radius(state))


def fit_all(state: AppState, *, reference: int | None = None) -> AppState:
    """Fill every frame's result in place (as Fit All without tracking) and return *state*."""
    if state.detection_mode == "edge":
        job = edge_job(state, reference)
        for i, path in enumerate(state.images):
            fit = edge_pipeline.fit_edge(edge_pipeline.read_raw(path), state.gridx, state.gridy, job)
            _store(state, i, fit.radius, fit.rc, fit.cc)
            state.fit_confidence[i] = fit.confidence
            state.fit_flags[i] = frozenset(fit.flags)
        return state
    params = _legacy_params(state)
    for i, path in enumerate(state.images):
        fit = fit_image_file(path, params)
        _store(state, i, fit.radius, fit.center_row, fit.center_col)
    return state


def _store(state: AppState, i: int, radius: float, rc: float, cc: float) -> None:
    ok = radius > 0
    state.radius[i] = radius if ok else -1.0
    state.circle_fit_par[i] = [rc, cc] if ok else [np.nan, np.nan]
    state.fit_source[i] = FitSource.AUTO if ok else FitSource.NONE


def set_manual(state: AppState, i: int, radius: float, rc: float, cc: float) -> None:
    state.radius[i] = radius
    state.circle_fit_par[i] = [rc, cc]
    state.fit_source[i] = FitSource.MANUAL
