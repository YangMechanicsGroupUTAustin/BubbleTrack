"""Controller-side helpers for the Edge engine, shared by every fitting path.

Single-frame fit, chart refit, neighbour refit and Fit All all resolve the
reference frame, build engine parameters and store results the same way.
"""

from __future__ import annotations

import numpy as np

from bubbletrack.controller.display_mixin import current_source
from bubbletrack.model import edge_pipeline
from bubbletrack.model.edge_pipeline import EdgeFrameFit, EdgeJob
from bubbletrack.model.provenance import FitSource
from bubbletrack.model.state import AppState

# Auto-selected reference per source; choosing reads several frames.
_reference_cache: dict[tuple, int | None] = {}


def _source_key(state: AppState) -> tuple:
    return (state.folder_path, state.video_path, len(state.images), state.reference_frame)


def reference_index(state: AppState) -> int | None:
    """The explicit reference frame, or the cached auto-selected one."""
    key = _source_key(state)
    if key not in _reference_cache:
        _reference_cache[key] = edge_pipeline.resolve_reference_index(
            lambda i: edge_pipeline.read_raw(current_source(state, i)),
            len(state.images),
            state.reference_frame,
        )
    return _reference_cache[key]


def reference_text(state: AppState) -> str:
    """Label for the UI, e.g. 'Frame 2 (auto)' or 'None found'."""
    if not state.images:
        return "Auto"
    idx = reference_index(state)
    if idx is None:
        return "None found (surface fallback)"
    origin = "set" if state.reference_frame is not None else "auto"
    return f"Frame {idx + 1} ({origin})"


def seed_text(state: AppState) -> str:
    if state.edge_seed is None:
        return "None"
    row, col = state.edge_seed
    return f"row {row:.0f}, col {col:.0f}"


def build_job(state: AppState, max_radius: float, *,
              prior: tuple[float, float, float] | None = None) -> EdgeJob:
    idx = reference_index(state)
    reference = (edge_pipeline.read_raw(current_source(state, idx))
                 if idx is not None else None)
    return EdgeJob(params=edge_pipeline.params_from_state(state, prior=prior),
                   reference=reference, max_radius=max_radius)


def fit_frame(state: AppState, idx: int, max_radius: float, *,
              prior: tuple[float, float, float] | None = None) -> EdgeFrameFit:
    raw = edge_pipeline.read_raw(current_source(state, idx))
    return edge_pipeline.fit_edge(raw, state.gridx, state.gridy,
                                  build_job(state, max_radius, prior=prior))


def store_fit(state: AppState, idx: int, fit: EdgeFrameFit, source: FitSource) -> None:
    """Write a fit into the result arrays (rejected fits mark the frame unfitted)."""
    accepted = fit.radius > 0
    state.radius[idx] = fit.radius if accepted else -1.0
    state.circle_fit_par[idx] = [fit.rc, fit.cc] if accepted else [np.nan, np.nan]
    state.circle_xy[idx] = fit.edge_xy if accepted else None
    if state.region_count is not None:
        state.region_count[idx] = 0
    if state.fit_confidence is not None:
        state.fit_confidence[idx] = fit.confidence
    if state.fit_source is not None:
        state.fit_source[idx] = source if accepted else FitSource.NONE


def status_text(idx: int, fit: EdgeFrameFit) -> tuple[str, str]:
    """(message, colour) for the header after fitting frame *idx*."""
    if fit.radius > 0:
        flags = f", {', '.join(sorted(fit.flags))}" if fit.flags else ""
        return (f"Frame {idx + 1}: R = {fit.radius:.1f} px "
                f"(confidence {fit.confidence:.2f}{flags})", "#10b981")
    return (f"Frame {idx + 1}: {fit.rejected_reason} — not accepted", "#f59e0b")
