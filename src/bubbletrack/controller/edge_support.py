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
from bubbletrack.model.suggestion import Suggestion, set_suggestion, shown_suggestion

# Auto-selected reference per source; choosing reads several frames.
_reference_cache: dict[tuple, int | None] = {}
_reference_raw_cache: dict[tuple, np.ndarray] = {}  # one entry: the current reference frame
_unusable_cache: dict[tuple, np.ndarray] = {}  # one entry: the current reference and ROI
GUESS_HINT = "Auto's guess is shown dashed (see Review)"


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


def reference_raw(state: AppState) -> np.ndarray | None:
    """The reference frame's pixels (read once per source and reference)."""
    idx = reference_index(state)
    if idx is None:
        return None
    key = (*_source_key(state), idx)
    if key not in _reference_raw_cache:
        _reference_raw_cache.clear()
        _reference_raw_cache[key] = edge_pipeline.read_raw(current_source(state, idx))
    return _reference_raw_cache[key]


def unusable_pixels(state: AppState) -> np.ndarray | None:
    """The ROI's pixels Auto cannot use with the reference frame (bool, read-only; None
    without a reference).  They depend on the reference and the ROI only, so they are
    computed once per source, reference and ROI: scrubbing frames reuses them."""
    idx = reference_index(state)
    if idx is None:
        return None
    key = (*_source_key(state), idx, tuple(state.gridx), tuple(state.gridy))
    if key not in _unusable_cache:
        mask = edge_pipeline.unusable_pixels(reference_raw(state), state.gridx, state.gridy)
        mask.flags.writeable = False
        _unusable_cache.clear()
        _unusable_cache[key] = mask
    return _unusable_cache[key]


def build_job(state: AppState, max_radius: float, *,
              prior: tuple[float, float, float] | None = None) -> EdgeJob:
    return EdgeJob(params=edge_pipeline.params_from_state(state, prior=prior),
                   reference=reference_raw(state), max_radius=max_radius)


def view_frame(state: AppState, idx: int) -> np.ndarray:
    """Frame *idx*'s ROI as the engine sees it (frame / reference)."""
    raw = edge_pipeline.read_raw(current_source(state, idx))
    return edge_pipeline.view_edge(raw, state.gridx, state.gridy,
                                   build_job(state, max_radius=float("inf")))


def fit_frame(state: AppState, idx: int, max_radius: float, *,
              prior: tuple[float, float, float] | None = None) -> EdgeFrameFit:
    raw = edge_pipeline.read_raw(current_source(state, idx))
    return edge_pipeline.fit_edge(raw, state.gridx, state.gridy,
                                  build_job(state, max_radius, prior=prior))


def store_fit(state: AppState, idx: int, fit: EdgeFrameFit, source: FitSource) -> None:
    """Write a fit into the result arrays (rejected fits mark the frame unfitted
    and keep the engine's guess, if any, as the frame's suggestion)."""
    accepted = fit.radius > 0
    state.radius[idx] = fit.radius if accepted else -1.0
    state.circle_fit_par[idx] = [fit.rc, fit.cc] if accepted else [np.nan, np.nan]
    state.circle_xy[idx] = fit.edge_xy if accepted else None
    if state.region_count is not None:
        state.region_count[idx] = 0
    if state.fit_confidence is not None:
        state.fit_confidence[idx] = fit.confidence
    store_meta(state, idx, {"flags": fit.flags, "metrics": fit.metrics,
                            "suggestion": None if accepted else fit.suggestion})
    if state.fit_source is not None:
        state.fit_source[idx] = source if accepted else FitSource.NONE


def store_meta(state: AppState, idx: int, meta: dict) -> None:
    """Write per-frame flags and shape metrics (unreported metrics become NaN)
    and Auto's guess for a rejected frame (cleared unless *meta* has one)."""
    if state.fit_flags is not None:
        state.fit_flags[idx] = frozenset(meta.get("flags", ()))
    if state.fit_metrics is not None:
        reported = meta.get("metrics", {})
        for key, values in state.fit_metrics.items():
            values[idx] = reported.get(key, np.nan)
    set_suggestion(state, idx, meta.get("suggestion"))


def clear_suggestion(state: AppState, idx: int) -> None:
    """Drop frame *idx*'s guess (a new result or fit attempt replaces it)."""
    set_suggestion(state, idx, None)


def fit_from_suggestion(suggestion: Suggestion) -> EdgeFrameFit:
    """The fit a suggestion stands for, as the engine reported it."""
    return EdgeFrameFit(suggestion.radius, suggestion.rc, suggestion.cc,
                        np.array(suggestion.edge_xy, dtype=np.float64), None,
                        suggestion.confidence, suggestion.flags, "",
                        metrics=suggestion.metrics_dict)


def accept_suggestion(state: AppState, idx: int) -> bool:
    """Accept frame *idx*'s shown guess: store it as an Auto result and lock it
    (locked = accepted by the user; Fit All and tracking keep it).  False if
    the frame shows no guess."""
    suggestion = shown_suggestion(state, idx)
    if suggestion is None:
        return False
    store_fit(state, idx, fit_from_suggestion(suggestion), FitSource.AUTO)
    if state.fit_locked is not None:
        state.fit_locked[idx] = True
    return True


def status_text(idx: int, fit: EdgeFrameFit) -> tuple[str, str]:
    """(message, colour) for the header after fitting frame *idx*."""
    if fit.radius > 0:
        flags = f", {', '.join(sorted(fit.flags))}" if fit.flags else ""
        return (f"Frame {idx + 1}: R = {fit.radius:.1f} px "
                f"(confidence {fit.confidence:.2f}{flags})", "#10b981")
    text = f"Frame {idx + 1}: {fit.rejected_reason} — not accepted"
    if fit.suggestion is not None:
        text += f"; {GUESS_HINT}"
    return text, "#f59e0b"

