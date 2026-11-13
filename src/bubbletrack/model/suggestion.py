"""Auto's guess on a frame it rejected: kept for display, a result only once accepted.

The Edge engine rejects a frame whose confidence is below its gate and the
frame gets no result.  On the labelled sets many of those outlines are
right, so the candidate is kept as a *suggestion* (``AppState.fit_suggestion``)
next to the result arrays:

* it is display-only: never charted, exported, tracked or labelled, and the
  frame still counts as "no result";
* it shows only while its frame has no result and is not locked
  (:func:`shown_suggestion`); every path that stores an Auto fit or erases a
  frame replaces or clears it;
* Accept in Review turns it into the frame's result (``edge_support``).

Qt-free; sessions store suggestions with :func:`to_json` / :func:`from_json`.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True, eq=False)
class Suggestion:
    """A rejected fit's circle and outline (1-based full-image ``[row, col]``)."""

    radius: float
    rc: float
    cc: float
    edge_xy: np.ndarray  # (M, 2) outline points; never modified
    confidence: float
    flags: frozenset = frozenset()
    metrics: tuple[tuple[str, float], ...] = ()  # sorted (key, value) pairs

    @property
    def metrics_dict(self) -> dict[str, float]:
        return dict(self.metrics)


def make_suggestion(radius: float, rc: float, cc: float, edge_xy, confidence: float,
                    flags=frozenset(), metrics: Mapping[str, float] | None = None,
                    ) -> Suggestion | None:
    """A suggestion for a candidate circle, or None without one (radius <= 0, NaN)."""
    values = (float(radius), float(rc), float(cc))
    if not all(math.isfinite(v) for v in values) or values[0] <= 0:
        return None
    xy = np.array(edge_xy if edge_xy is not None else np.empty((0, 2)), dtype=np.float64)
    return Suggestion(*values, edge_xy=xy.reshape(-1, 2), confidence=float(confidence),
                      flags=frozenset(flags),
                      metrics=tuple(sorted((str(k), float(v)) for k, v in (metrics or {}).items())))


def _same_float(a: float, b: float) -> bool:
    return a == b or (math.isnan(a) and math.isnan(b))


def same_suggestion(a: Suggestion | None, b: Suggestion | None) -> bool:
    """Equal suggestions (NaN equals NaN); None equals only None."""
    if a is b:
        return True
    if a is None or b is None:
        return False
    return (all(_same_float(x, y) for x, y in ((a.radius, b.radius), (a.rc, b.rc), (a.cc, b.cc),
                                                (a.confidence, b.confidence)))
            and a.flags == b.flags
            and len(a.metrics) == len(b.metrics)
            and all(ka == kb and _same_float(va, vb)
                    for (ka, va), (kb, vb) in zip(a.metrics, b.metrics))
            and np.array_equal(a.edge_xy, b.edge_xy, equal_nan=True))


# -- per-frame access -------------------------------------------------------- #

def stored_suggestion(state, idx: int) -> Suggestion | None:
    """Frame *idx*'s stored suggestion, whether or not it is shown."""
    values = getattr(state, "fit_suggestion", None)
    if values is None or not 0 <= idx < len(values):
        return None
    return values[idx]


def set_suggestion(state, idx: int, suggestion: Suggestion | None) -> None:
    """Replace frame *idx*'s suggestion in place (no-op without the array)."""
    values = getattr(state, "fit_suggestion", None)
    if values is not None and 0 <= idx < len(values):
        values[idx] = suggestion


def shown_suggestion(state, idx: int) -> Suggestion | None:
    """The suggestion that applies to frame *idx*: none once the frame has a
    result, or is locked (an accepted "no bubble here")."""
    suggestion = stored_suggestion(state, idx)
    if suggestion is None or state.radius is None or idx >= len(state.radius):
        return None
    if state.radius[idx] > 0:
        return None
    locked = state.fit_locked
    if locked is not None and idx < len(locked) and locked[idx]:
        return None
    return suggestion


def suggested_mask(state, n: int) -> np.ndarray:
    """Boolean mask of the *n* frames whose suggestion is shown (as
    :func:`shown_suggestion`, vectorised: the review queue asks often)."""
    values = getattr(state, "fit_suggestion", None)
    if values is None or len(values) != n or state.radius is None or len(state.radius) != n:
        return np.zeros(n, dtype=bool)
    mask = np.fromiter((s is not None for s in values), dtype=bool, count=n)
    if mask.any():
        mask &= ~(np.asarray(state.radius, dtype=float) > 0)
        locked = state.fit_locked
        if locked is not None and len(locked) == n:
            mask &= ~np.asarray(locked, dtype=bool)
    return mask


# -- sessions ---------------------------------------------------------------- #

def _finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None


def to_json(suggestions) -> list | None:
    """Per-frame suggestions as strict-JSON data (None entries stay null)."""
    if suggestions is None:
        return None
    out: list = []
    for s in suggestions:
        if s is None:
            out.append(None)
            continue
        xy = np.asarray(s.edge_xy, dtype=np.float64).reshape(-1, 2)
        out.append({
            "radius": s.radius,
            "center": [s.rc, s.cc],
            "edge_xy": [[_finite_or_none(float(v)) for v in row] for row in xy],
            "confidence": _finite_or_none(s.confidence),
            "flags": sorted(s.flags),
            "metrics": {k: _finite_or_none(v) for k, v in s.metrics},
        })
    return out


def _entry(data) -> Suggestion | None:
    if data is None:
        return None
    rc, cc = data["center"]
    xy = np.array([[np.nan if v is None else v for v in row] for row in data.get("edge_xy") or []],
                  dtype=np.float64).reshape(-1, 2)
    conf = data.get("confidence")
    metrics = {k: (np.nan if v is None else float(v))
               for k, v in (data.get("metrics") or {}).items()}
    return make_suggestion(data["radius"], rc, cc, xy, np.nan if conf is None else conf,
                           frozenset(str(f) for f in data.get("flags") or ()), metrics)


def from_json(data) -> list[Suggestion | None] | None:
    """Inverse of :func:`to_json`; a malformed entry becomes None (logged)."""
    if not isinstance(data, list):
        return None
    out: list[Suggestion | None] = []
    for i, entry in enumerate(data):
        try:
            out.append(_entry(entry))
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("Ignoring the saved Auto guess of frame %d: %s", i + 1, exc)
            out.append(None)
    return out
