"""Score a replay against a hard case's labels (pure).

Only an ACCEPTED result counts: a rejected frame shows no result in the app.
For every labelled frame the outcome is

* ``correct``  -- bubble label: an accepted result whose radius is within
  ``max(1 px, 3 % R)`` and whose centre is within ``max(2 px, 10 % R)`` of the
  label (R = label radius).  "No bubble" label: no result;
* ``wrong``    -- an accepted result that does not match, including any
  result on a "no bubble" frame (the app shows a wrong number);
* ``rejected`` -- bubble label, but no result (the app shows a gap).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from bubbletrack.model.cases.replay import Replayed
from bubbletrack.model.cases.schema import LABEL_NO_BUBBLE, HardCase, Label

CORRECT, WRONG, REJECTED = "correct", "wrong", "rejected"
OUTCOMES = (CORRECT, WRONG, REJECTED)
_AUTOMATIC_SOURCES = frozenset({"auto", "tracked"})  # results Fit All produced


@dataclass(frozen=True)
class Tolerance:
    radius_px: float = 1.0
    radius_frac: float = 0.03
    center_px: float = 2.0
    center_frac: float = 0.10

    def radius_tol(self, radius: float) -> float:
        return max(self.radius_px, self.radius_frac * radius)

    def center_tol(self, radius: float) -> float:
        return max(self.center_px, self.center_frac * radius)

    def describe(self) -> str:
        return (f"radius within max({self.radius_px:g} px, {self.radius_frac:.0%}) and centre "
                f"within max({self.center_px:g} px, {self.center_frac:.0%} R) of the label")


def center_distance(a: tuple[float, float] | None, b: tuple[float, float] | None) -> float | None:
    if a is None or b is None:
        return None
    return math.hypot(a[0] - b[0], a[1] - b[1])


def matches(radius: float, center, ref_radius: float, ref_center, tol: Tolerance) -> bool:
    """Same circle within *tol* (sized by *ref_radius*); no reference centre: radius only."""
    if abs(radius - ref_radius) > tol.radius_tol(ref_radius):
        return False
    if ref_center is None:
        return True  # no centre to compare with
    distance = center_distance(center, ref_center)
    return distance is not None and distance <= tol.center_tol(ref_radius)


def judge(label: Label, replayed: Replayed, tol: Tolerance) -> str:
    """The outcome of *replayed* on a frame labelled *label* (must be labelled)."""
    if label.status == LABEL_NO_BUBBLE:
        return WRONG if replayed.accepted else CORRECT
    if not replayed.accepted:
        return REJECTED
    ok = matches(replayed.radius, replayed.center, label.radius, label.center, tol)
    return CORRECT if ok else WRONG


@dataclass(frozen=True)
class FrameScore:
    index: int
    outcome: str
    label: Label
    replayed: Replayed

    @property
    def radius_error(self) -> float | None:
        if self.label.radius is None or self.replayed.radius is None:
            return None
        return self.replayed.radius - self.label.radius

    @property
    def center_error(self) -> float | None:
        return center_distance(self.replayed.center, self.label.center)


@dataclass(frozen=True)
class CaseScore:
    """One case's outcome; ``error`` is set when it could not be replayed."""

    bundle: str  # bundle file / folder name
    case: HardCase | None
    frames: tuple[FrameScore, ...] = ()
    unlabelled: int = 0
    auto_results: int = 0  # the app's automatic results at export ...
    reproduced: int = 0  # ... that the replay gives again (within tolerance)
    error: str = ""

    @property
    def case_id(self) -> str:
        return self.case.case_id if self.case is not None and self.case.case_id else self.bundle

    def count(self, outcome: str) -> int:
        return sum(1 for f in self.frames if f.outcome == outcome)

    @property
    def labelled(self) -> int:
        return len(self.frames)


def _reproduced(case: HardCase, replayed: Mapping[int, Replayed], tol: Tolerance) -> tuple[int, int]:
    total = same = 0
    for frame in case.measured_frames():
        result, now = frame.result, replayed.get(frame.index)
        if result.radius is None or result.source not in _AUTOMATIC_SOURCES or now is None:
            continue
        total += 1
        same += bool(now.accepted and matches(now.radius, now.center, result.radius,
                                              result.center, tol))
    return total, same


def score_case(bundle: str, case: HardCase, replayed: Mapping[int, Replayed],
               tol: Tolerance = Tolerance()) -> CaseScore:
    """Compare *replayed* with every labelled selected / context frame of *case*."""
    frames, unlabelled = [], 0
    for frame in case.measured_frames():
        now = replayed.get(frame.index)
        if not frame.label.is_labelled or now is None:
            unlabelled += 1
            continue
        frames.append(FrameScore(frame.index, judge(frame.label, now, tol), frame.label, now))
    total, same = _reproduced(case, replayed, tol)
    return CaseScore(bundle, case, tuple(frames), unlabelled, total, same)
