"""Adapter between the app and the Edge detection engine (``model.edge``).

The engine works on raw full-bit-depth frames and a bubble-free reference
frame instead of the 8-bit adaptive-threshold image, so this module owns:

* reading raw frames (paths, or frames already decoded from a video),
* resolving the reference frame (explicit index, or auto-selected),
* turning app state into ``EdgeParams``,
* applying the confidence gate and the plausible-radius check.

Everything here is picklable and Qt-free so worker processes can use it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import cv2
import numpy as np

from bubbletrack.model.image_io import ImageLoadError, imread_unicode


def _engine():
    """The engine entry point (indirection so tests can inject a fake)."""
    from bubbletrack.model.edge import detect_edge
    return detect_edge


def _choose_reference(read_raw: Callable[[int], np.ndarray], n_frames: int) -> int | None:
    from bubbletrack.model.edge import choose_reference_frame
    return choose_reference_frame(read_raw, n_frames)


def _params_class():
    from bubbletrack.model.edge import EdgeParams
    return EdgeParams


def make_params(*, edge_definition: str = "steep", polarity: str = "dark",
                seed: tuple[float, float] | None = None,
                prior: tuple[float, float, float] | None = None):
    return _params_class()(edge_definition=edge_definition, polarity=polarity,
                           seed=seed, prior=prior)


def params_from_state(state, *, prior: tuple[float, float, float] | None = None):
    """Engine settings for *state*; a *prior* circle replaces the seed click.

    With both, the seed decides the candidate and the prior is ignored; on
    the labelled LIC sets that cost 3 points of accuracy for neighbour and
    drag refits (85 -> 82 %, 34 -> 31 %), and on a moving bubble the click
    is far from the bubble in most frames anyway.
    """
    seed = None if prior is not None else state.edge_seed
    return make_params(edge_definition=state.edge_definition, polarity=state.edge_polarity,
                       seed=seed, prior=prior)


def read_raw(source) -> np.ndarray:
    """Raw frame at its native bit depth: a path is read, an array is returned as is."""
    if isinstance(source, np.ndarray):
        return source
    raw = imread_unicode(source, cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise ImageLoadError(f"Cannot read image (file may be missing or corrupted): {source}")
    return raw


def resolve_reference_index(read_frame: Callable[[int], np.ndarray], n_frames: int,
                            explicit: int | None) -> int | None:
    """The user's reference frame if set and valid, else the auto-selected one."""
    if explicit is not None and 0 <= explicit < n_frames:
        return explicit
    return _choose_reference(read_frame, n_frames)


@dataclass(frozen=True)
class EdgeJob:
    """Everything a worker needs besides the frame itself (picklable)."""

    params: object  # EdgeParams
    reference: np.ndarray | None
    max_radius: float


@dataclass(frozen=True)
class EdgeFrameFit:
    radius: float  # -1.0 when rejected (no result, low confidence, implausible)
    rc: float
    cc: float
    edge_xy: np.ndarray
    mask: np.ndarray | None
    confidence: float
    flags: frozenset
    rejected_reason: str  # "" when accepted
    metrics: dict = field(default_factory=dict)  # subset of state.FIT_METRIC_KEYS


def _metrics(res) -> dict:
    """Shape metrics the engine reports (missing ones are simply absent)."""
    from bubbletrack.model.state import FIT_METRIC_KEYS

    out = {}
    for key in FIT_METRIC_KEYS:
        value = getattr(res, key, None)
        if value is not None:
            out[key] = float(value)
    return out


def view_edge(raw: np.ndarray, gridx: tuple[int, int], gridy: tuple[int, int],
              job: EdgeJob) -> np.ndarray:
    """The ROI as the engine sees it (frame / reference); see ``engine_view``."""
    from bubbletrack.model.edge.engine import engine_view

    return engine_view(raw, gridx, gridy, reference=job.reference, params=job.params)


def fit_edge(raw: np.ndarray, gridx: tuple[int, int], gridy: tuple[int, int],
             job: EdgeJob) -> EdgeFrameFit:
    """Run the engine on one raw frame and apply the acceptance rules."""
    res = _engine()(raw, gridx, gridy, reference=job.reference, params=job.params)
    confidence = float(res.confidence)
    if not np.isfinite(res.radius):
        reason = "no bubble found"
    elif not res.accepted:
        reason = f"low confidence ({confidence:.2f})"
    elif not (0 < res.radius <= job.max_radius):
        reason = "implausible radius"
    else:
        reason = ""
    if reason:
        return EdgeFrameFit(-1.0, float("nan"), float("nan"), np.empty((0, 2)),
                            res.mask, confidence, frozenset(res.flags), reason,
                            metrics=_metrics(res))
    return EdgeFrameFit(float(res.radius), float(res.rc), float(res.cc),
                        np.asarray(res.edge_xy, dtype=np.float64).reshape(-1, 2),
                        res.mask, confidence, frozenset(res.flags), "",
                        metrics=_metrics(res))
