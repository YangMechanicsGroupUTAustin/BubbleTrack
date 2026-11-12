"""Value types of the sequence solver: per-frame observations in, per-frame solutions out.

Coordinates are 1-based full-image ``(row, col)`` as everywhere in the app.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

# --------------------------------------------------------------------------- #
#  Vocabulary
# --------------------------------------------------------------------------- #
SOURCE_RAW = "raw"            # the single-frame result, confirmed by the track
SOURCE_TRACKED = "tracked"    # re-measured with the track's circle prior (repaired)
SOURCE_ANCHOR = "anchor"      # manual / accepted frame, copied unchanged
SOURCE_LOST = "lost"          # nothing consistent with the track; values are the unverified raw ones
SOURCE_NONE = "none"          # no bubble: before nucleation, after a cloud, or nothing to track
SOURCES = (SOURCE_RAW, SOURCE_TRACKED, SOURCE_ANCHOR, SOURCE_LOST, SOURCE_NONE)

FLAG_DISAGREE = "disagree"            # the solution differs from the single-frame result (or the two passes differ)
FLAG_REACQUIRED = "reacquired"        # first accepted frame after a loss
FLAG_LOST = "lost"
FLAG_COLLAPSE = "collapse"            # a collapse (radius minimum) frame
FLAG_LOW_CONFIDENCE = "low_confidence"
FLAG_JUMP = "jump"                    # accepted outside the tight gate (sudden change)
FLAG_CLOUD = "cloud"                  # post-collapse fragment cloud: tracking stopped

RAW_REVIEW_FLAGS = ("fragmented", "non_circular", "clipped_completed")
"""Single-frame (engine) flags that put a frame on the review list."""


def _as_float(value, name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number, got {value!r}") from exc


def _finite(*values: float) -> bool:
    return all(math.isfinite(v) for v in values)


@dataclass(frozen=True)
class Observation:
    """One frame's single-frame result (from Fit All or from a re-detection)."""

    radius: float
    """<= 0 or NaN when nothing was found."""
    rc: float
    cc: float
    confidence: float
    """In [0, 1]; NaN when the detector has no confidence (legacy modes): then it never gates."""
    flags: frozenset = frozenset()

    def __post_init__(self) -> None:
        for name in ("radius", "rc", "cc", "confidence"):
            object.__setattr__(self, name, _as_float(getattr(self, name), name))
        object.__setattr__(self, "flags", frozenset(self.flags))

    @property
    def ok(self) -> bool:
        """A finite radius > 0 with a finite centre."""
        return _finite(self.radius, self.rc, self.cc) and self.radius > 0

    @property
    def circle(self) -> tuple[float, float, float]:
        return (self.rc, self.cc, self.radius)


NO_OBSERVATION = Observation(float("nan"), float("nan"), float("nan"), float("nan"))

Detector = Callable[[int, "tuple[float, float, float] | None"], Observation]
"""``redetect(idx, prior)``: re-run frame ``idx`` with a circle prior ``(rc, cc, R)``;
``prior=None`` asks for an unconstrained (global) detection."""


@dataclass(frozen=True)
class FrameSolution:
    """The solver's verdict for one frame."""

    radius: float
    rc: float
    cc: float
    confidence: float
    source: str
    """One of :data:`SOURCES`."""
    flags: frozenset
    """Solver flags (``FLAG_*``) plus the engine flags of the observation used."""
    cycle: int
    """0-based cycle index, -1 outside any cycle."""

    def __post_init__(self) -> None:
        if self.source not in SOURCES:
            raise ValueError(f"source must be one of {SOURCES}, got {self.source!r}")
        object.__setattr__(self, "flags", frozenset(self.flags))

    @property
    def ok(self) -> bool:
        return _finite(self.radius, self.rc, self.cc) and self.radius > 0


@dataclass(frozen=True)
class SequenceResult:
    """Solved sequence."""

    frames: tuple[FrameSolution, ...]
    collapse_frames: tuple[int, ...]
    cycles: tuple[tuple[int, int], ...]
    """Inclusive ``(start, end)`` frame range per cycle; each collapse ends a cycle."""
    review: tuple[tuple[int, tuple[str, ...]], ...]
    """``(frame, reasons)``, most important first; reasons are ordered by importance too."""
    seed_frame: int
    """First frame the tracking started from (an anchor, else the automatic seed; -1 when
    there was nothing to track)."""
    redetect_calls: int = field(default=0, compare=False)
    """Number of ``redetect`` calls the solve made (diagnostic)."""
