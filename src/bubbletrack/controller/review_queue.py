"""Review queue rules (pure): which frames are likely wrong, and why.

``build_review_items`` takes the results and returns one ``ReviewItem`` per
suspicious frame, most likely wrong first.  ``review_segments`` groups them
into the rows the Review dock shows.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from bubbletrack.model.provenance import FitSource, user_confirmed_mask
from bubbletrack.model.sequence import CONTEXT_FLAGS
from bubbletrack.model.state import AppState
from bubbletrack.model.suggestion import suggested_mask

LOW_CONFIDENCE = 0.5  # engine confidence below this is worth a look
JUMP_FRACTION = 0.15  # |dR| to both valid neighbours, relative to each neighbour

# Higher = more likely wrong.  Engine and sequence flags share these keys.
REASON_WEIGHTS: dict[str, int] = {
    "lost": 100,
    "rejected": 90,
    "disagree": 85,
    "jump": 80,
    "low_confidence": 60,  # rises to 90 as the confidence falls to 0
    "ml_outline": 58,  # ML assist's own outline: always checked (see review_segments)
    "ml_disagrees": 56,  # plain Auto kept, but ML assist saw the bubble elsewhere
    "multiple_edges": 60,
    "non_circular": 55,
    "fragmented": 50,
    "clipped_completed": 45,
    "reacquired": 40,
    "collapse": 35,
    "cloud": 30,
    "shoulder": 25,
    "ml_unfamiliar": 20,  # unlike the frames ML assist was taught on (context only)
}
UNKNOWN_REASON_WEIGHT = 50  # reasons that only arrive through ``extra``
REVIEW_FLAGS = frozenset(REASON_WEIGHTS) - {"rejected"}
# Routine engine flags: they describe a listed frame and add to its rank, but
# never list a frame on their own (see the sequence solver's CONTEXT_FLAGS).
CONTEXT_REASONS = CONTEXT_FLAGS | {"ml_unfamiliar"}

GUESS = "guess"
"""A "rejected" frame that shows Auto's guess (``model.suggestion``).  It only
describes the frame: it comes with "rejected", ranks right after it, so the
row stays "No result", and Accept stores the guess instead of "no bubble"."""
_GUESS_WEIGHT = REASON_WEIGHTS["rejected"] - 1


@dataclass(frozen=True, eq=False)
class ReviewItem:
    frame: int  # 0-based
    reasons: tuple[str, ...]  # most severe first
    severity: int  # higher = more likely wrong
    confidence: float = float("nan")  # engine confidence, NaN if not scored

    def _key(self) -> tuple:
        # NaN != NaN would make every unscored item differ from itself.
        conf = None if math.isnan(self.confidence) else self.confidence
        return self.frame, self.reasons, self.severity, conf

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ReviewItem):
            return NotImplemented
        return self._key() == other._key()

    def __hash__(self) -> int:
        return hash(self._key())


def build_review_items(state: AppState, extra: Iterable[ReviewItem] | None = None,
                       ) -> list[ReviewItem]:
    """Frames worth a second look, most likely wrong first, then by frame.

    *extra* (e.g. the sequence solver's list) is merged per frame.  Locked
    frames (accepted in review) are never listed.
    """
    if state.radius is None:
        return []
    radius = np.asarray(state.radius, dtype=float)
    n = len(radius)
    conf = _per_frame(state.fit_confidence, n, np.nan, float)
    locked = _per_frame(state.fit_locked, n, False, bool)
    own = _frame_reasons(state, radius, conf)
    added = _merge_extra(extra, n)
    items = [_make_item(f, own.get(f, ()), added.get(f), float(conf[f]))
             for f in sorted(set(own) | set(added)) if not locked[f]]
    items = [item for item in items if not CONTEXT_REASONS.issuperset(item.reasons)]
    return sorted(items, key=lambda item: (-item.severity, item.frame))


def jump_frames(radius: np.ndarray, threshold: float = JUMP_FRACTION) -> np.ndarray:
    """Frames that spike above or dip below BOTH nearest valid neighbours.

    A steady fast change (collapse, rebound) is not a jump: each frame there
    lies between its neighbours.
    """
    radius = np.asarray(radius, dtype=float)
    valid = np.flatnonzero(np.isfinite(radius) & (radius > 0))
    if len(valid) < 3:
        return np.empty(0, dtype=int)
    r = radius[valid]
    mid, prev, nxt = r[1:-1], r[:-2], r[2:]
    up = (mid > prev * (1 + threshold)) & (mid > nxt * (1 + threshold))
    down = (mid < prev * (1 - threshold)) & (mid < nxt * (1 - threshold))
    return valid[1:-1][up | down]


def next_after_action(old: list[int], frame: int, new: list[int]) -> int | None:
    """Frame to show after acting on *frame*: the next old item still listed."""
    remaining = set(new)
    if frame in old:
        pos = old.index(frame)
        candidates = old[pos + 1:] + old[:pos]
    else:
        candidates = old
    for f in candidates:
        if f in remaining and f != frame:
            return f
    return next((f for f in new if f != frame), new[0] if new else None)


def _per_frame(values, n: int, fill, dtype) -> np.ndarray:
    if values is None or len(values) != n:
        return np.full(n, fill, dtype=dtype)
    return np.asarray(values, dtype=dtype)


def _frame_reasons(state: AppState, radius: np.ndarray, conf: np.ndarray,
                   ) -> dict[int, list[str]]:
    n = len(radius)
    source = _per_frame(state.fit_source, n, FitSource.NONE, int)
    manual = user_confirmed_mask(source)
    # A manual fix (or a drawn circle) supersedes the engine's verdict on that frame.
    engine = np.isfinite(conf) & ~manual
    fitted = np.isfinite(radius) & (radius > 0)
    found: dict[int, list[str]] = defaultdict(list)

    def add(i, reason: str) -> None:
        if reason not in found[int(i)]:
            found[int(i)].append(reason)

    for i in np.flatnonzero(engine & ~fitted):
        add(i, "rejected")
    for i in np.flatnonzero(engine & ~fitted & suggested_mask(state, n)):
        add(i, GUESS)
    for i in np.flatnonzero(engine & (conf < LOW_CONFIDENCE)):
        add(i, "low_confidence")
    for i, flags in enumerate((state.fit_flags or ())[:n]):
        if flags and not manual[i]:
            for reason in sorted(flags & REVIEW_FLAGS):
                add(i, reason)
    unscored = ~np.isfinite(conf)  # legacy modes produce no flags of their own
    for i in jump_frames(radius):
        if unscored[i]:
            add(i, "jump")
    return found


def _merge_extra(extra: Iterable[ReviewItem] | None, n: int) -> dict[int, ReviewItem]:
    merged: dict[int, ReviewItem] = {}
    for item in extra or ():
        if not 0 <= item.frame < n:
            continue
        prev = merged.get(item.frame)
        if prev is None:
            merged[item.frame] = item
        else:
            reasons = prev.reasons + tuple(r for r in item.reasons if r not in prev.reasons)
            merged[item.frame] = ReviewItem(item.frame, reasons,
                                            max(prev.severity, item.severity))
    return merged


def _weight(reason: str, confidence: float) -> int:
    if reason == "low_confidence" and np.isfinite(confidence):
        shortfall = (LOW_CONFIDENCE - min(max(confidence, 0.0), LOW_CONFIDENCE)) / LOW_CONFIDENCE
        return REASON_WEIGHTS[reason] + round(30 * shortfall)
    if reason == GUESS:
        return _GUESS_WEIGHT
    return REASON_WEIGHTS.get(reason, UNKNOWN_REASON_WEIGHT)


def _make_item(frame: int, own: Iterable[str], extra: ReviewItem | None,
               confidence: float) -> ReviewItem:
    reasons = list(own)
    if extra is not None:
        reasons += [r for r in extra.reasons if r not in reasons]
    ordered = tuple(sorted(reasons, key=lambda r: -_weight(r, confidence)))
    worst = max((_weight(r, confidence) for r in ordered), default=0)
    severity = worst + max(len(ordered) - 1, 0)  # more reasons rank higher
    if extra is not None:
        severity = max(severity, extra.severity)
    return ReviewItem(frame, ordered, severity, confidence)
