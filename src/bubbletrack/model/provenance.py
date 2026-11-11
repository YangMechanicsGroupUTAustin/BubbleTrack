"""Per-frame provenance: where each fitted radius came from.

Stored as an int8 array (``AppState.fit_source``) so it is cheap to copy,
pickle, save in sessions and export.  Manual fixes are protected from being
silently overwritten by Fit All.
"""

from __future__ import annotations

from enum import IntEnum


class FitSource(IntEnum):
    NONE = 0  # not fitted, failed, deleted or cleared
    AUTO = 1  # Fit All / batch
    MANUAL = 2  # user-clicked edge points
    REFIT = 3  # single-frame refit from the chart menu or Fit Current Frame
    NEIGHBOR = 4  # refit with a circle prior from neighbouring frames
