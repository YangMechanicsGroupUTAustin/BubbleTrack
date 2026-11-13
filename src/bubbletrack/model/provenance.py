"""Per-frame provenance: where each fitted radius came from.

Stored as an int8 array (``AppState.fit_source``) so it is cheap to copy,
pickle, save in sessions and export.  Manual fixes are protected from being
silently overwritten by Fit All.
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np


class FitSource(IntEnum):
    NONE = 0  # not fitted, failed, deleted or cleared
    AUTO = 1  # Fit All / batch
    MANUAL = 2  # user-clicked edge points (or a drawn circle stored as is)
    REFIT = 3  # single-frame refit from the chart menu or Fit Current Frame
    NEIGHBOR = 4  # refit with a circle prior from neighbouring frames
    TRACKED = 5  # re-measured by the sequence solver near its tracked circle
    GUIDED = 6  # drawn circle, wall measured by Auto (the Manual tab's "Drag a circle")
    ML = 7  # ML assist's own outline (Auto found no wall): computed, always reviewed


AREA_RADIUS_FLAG = "area_radius"
"""Per-frame flag: this radius is the area-equivalent radius of the measured outline
("Bubble may be non-round"); without it an Auto radius is the least-squares circle through
the bubble outline.
Set by the Edge engine and ML assist; Review, the legend and exports read it."""


GUIDED_COUNTS_AS_CONFIRMED = True
"""Owner decision (pending; default True): a circle fix is the user's decision.

True: GUIDED frames are treated like MANUAL ones everywhere: diamond on the
R(t) chart, kept by Fit All (the keep/overwrite prompt counts them), anchors
of the sequence solver, never re-measured by tracking, not listed for review.
False: they are computed results like a neighbour refit (the source name
"GUIDED" is still recorded and exported).  Flip this one line to change it.
"""


def confirmed_sources() -> frozenset[FitSource]:
    """Sources the user decided, per ``GUIDED_COUNTS_AS_CONFIRMED``."""
    if GUIDED_COUNTS_AS_CONFIRMED:
        return frozenset({FitSource.MANUAL, FitSource.GUIDED})
    return frozenset({FitSource.MANUAL})


USER_CONFIRMED_SOURCES: frozenset[FitSource] = confirmed_sources()
"""Results the user decided: protected like manual fits (see above)."""


def is_user_confirmed(source) -> bool:
    """True for a source code the user decided (MANUAL, and GUIDED by default)."""
    try:
        return FitSource(int(source)) in USER_CONFIRMED_SOURCES
    except ValueError:
        return False


def user_confirmed_mask(sources, n: int | None = None) -> np.ndarray:
    """Boolean mask of user-decided frames; all False when *sources* is None."""
    if sources is None:
        return np.zeros(0 if n is None else n, dtype=bool)
    codes = np.asarray(sources, dtype=int)
    return np.isin(codes, [int(s) for s in USER_CONFIRMED_SOURCES])
