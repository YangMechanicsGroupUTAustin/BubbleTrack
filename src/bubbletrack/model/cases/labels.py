"""What the app shows on a frame, and whether the user verified it (its label).

The labels are the frames the user fixed:

* a manual fit, or a result accepted / locked in review or on the chart:
  that circle is the right answer;
* a frame accepted (locked) without a result, or whose result the user
  deleted: the right answer is "no bubble here";
* anything else is unlabelled: nobody checked it.
"""

from __future__ import annotations

import math

import numpy as np

from bubbletrack.model.cases.schema import (
    BY_ACCEPTED, BY_DELETED, BY_MANUAL, LABEL_BUBBLE, LABEL_NO_BUBBLE, AppResult, Label,
)
from bubbletrack.model.provenance import FitSource

_MANUAL = FitSource.MANUAL.name.lower()


def _item(values, idx: int, default):
    return default if values is None or idx >= len(values) else values[idx]


def _source_name(code: int, fitted: bool) -> str:
    try:
        return FitSource(int(code)).name.lower()
    except ValueError:
        return FitSource.AUTO.name.lower() if fitted else FitSource.NONE.name.lower()


def _finite(value) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def app_result(state, idx: int) -> AppResult:
    """Frame *idx*'s result as the app shows it (no result: ``radius`` None)."""
    radius = _finite(state.radius[idx])
    fitted = radius is not None and radius > 0
    center = None
    if fitted and state.circle_fit_par is not None:
        rc, cc = (_finite(v) for v in np.asarray(state.circle_fit_par[idx], dtype=float))
        center = None if rc is None or cc is None else (rc, cc)
    flags = _item(state.fit_flags, idx, frozenset()) or frozenset()
    default_source = FitSource.AUTO if fitted else FitSource.NONE
    return AppResult(
        radius=radius if fitted else None,
        center=center,
        confidence=_finite(_item(state.fit_confidence, idx, np.nan)),
        flags=tuple(sorted(flags)),
        source=_source_name(_item(state.fit_source, idx, default_source), fitted),
        locked=bool(_item(state.fit_locked, idx, False)),
    )


def frame_label(result: AppResult, *, deleted: bool = False) -> Label:
    """The verified answer behind *result*; *deleted*: the user deleted its result."""
    if result.radius is not None:
        if result.source == _MANUAL:
            return Label(LABEL_BUBBLE, result.radius, result.center, BY_MANUAL)
        if result.locked:
            return Label(LABEL_BUBBLE, result.radius, result.center, BY_ACCEPTED)
        return Label()
    if result.locked:
        return Label(LABEL_NO_BUBBLE, by=BY_ACCEPTED)
    if deleted:
        return Label(LABEL_NO_BUBBLE, by=BY_DELETED)
    return Label()
