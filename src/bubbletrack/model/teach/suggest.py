"""Which frames to ask the user to verify first (``tools/studies/teach/RESULTS.md``).

Frames spread over bubble sizes teach best: the biggest bubble alone (or only big
ones) gives a model that misses small bubbles.  Sizes come from Auto's accepted
results; when too few are accepted (the recordings that need ML assist most),
frames evenly spaced in time stand in for sizes (the bubble grows and collapses).

The teaching panel asks for two confirmed frames of *different situations*
(:func:`differ`): bubbles of clearly different size (the smaller at most
``SIZE_RATIO`` of the larger), or frames at least ``TIME_SHARE`` of the
recording apart (another phase: growth / collapse, another pulse, drift).  A
hand-painted frame has no measured size and is told apart by time only.
:func:`pick_candidate` offers the next frame to judge: one with an outline,
different from every frame already kept, away from the frames the user
rejected, a measured result before Auto's guesses, the more confident first.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass
from itertools import combinations

import numpy as np

MIN_ACCEPTED = 5
SMALLEST_PERCENTILE = 0.2
SIZE_RATIO = 0.7
"""Two bubbles differ in size when the smaller is at most 70 % of the larger."""
TIME_SHARE = 0.25
"""Two frames at least a quarter of the recording apart are another situation."""
BIG_PERCENTILE = 0.9  # "a big bubble": robust to a few huge wrong guesses
CONFIDENCE_STEP = 0.1  # confidences this close count as equal (size decides)


@dataclass(frozen=True)
class FrameSize:
    """A frame (0-based) and its bubble radius in px (NaN: unknown, e.g. painted)."""

    frame: int
    radius: float = math.nan

    @property
    def sized(self) -> bool:
        return math.isfinite(self.radius) and self.radius > 0


def size_differs(a: FrameSize, b: FrameSize) -> bool:
    """Both sizes known and the smaller at most ``SIZE_RATIO`` of the larger."""
    if not (a.sized and b.sized):
        return False
    return min(a.radius, b.radius) <= SIZE_RATIO * max(a.radius, b.radius)


def differ(a: FrameSize, b: FrameSize, n_frames: int) -> bool:
    """True when *a* and *b* are different situations (see the module doc)."""
    if a.frame == b.frame:
        return False
    far = abs(a.frame - b.frame) >= max(1.0, TIME_SHARE * n_frames)
    return far or size_differs(a, b)


def find_pair(frames: Sequence[FrameSize], n_frames: int,
              prefer: Collection[int] = ()) -> tuple[FrameSize, FrameSize] | None:
    """Two frames of *frames* that differ, or None; pairs holding a frame of
    *prefer* (e.g. the ones on show) come first, then frame order."""
    order = sorted(frames, key=lambda f: (f.frame not in prefer, f.frame))
    for a, b in combinations(order, 2):
        if differ(a, b, n_frames):
            return a, b
    return None


def pick_candidate(radius: Sequence[float], confidence: Sequence[float], trusted: Sequence[bool], *,
                   exclude: Iterable[int] = (), keep_apart: Sequence[FrameSize] = (),
                   avoid: Sequence[FrameSize] = ()) -> int | None:
    """The next frame to judge (0-based), or None when no frame qualifies.

    Parameters
    ----------
    radius, confidence, trusted : sequences of the same length
        Per frame: the outline the user would judge (NaN / <= 0: none), its
        confidence, and whether it is a measured result (True) or Auto's guess.
    exclude : frames never offered (confirmed, rejected, on show, the reference).
    keep_apart : frames the candidate must differ from (``differ``).
    avoid : frames the user rejected: frames like them come last.
    """
    r = np.asarray(radius, dtype=float)
    c = np.nan_to_num(np.asarray(confidence, dtype=float), nan=0.0)
    t = np.asarray(trusted, dtype=bool)
    if not (r.ndim == 1 and r.shape == c.shape == t.shape):
        raise ValueError(f"radius, confidence and trusted must be 1-D of one length, got "
                         f"{r.shape}, {c.shape} and {t.shape}")
    n, skip = len(r), {int(i) for i in exclude}
    outline = [i for i in np.flatnonzero(np.isfinite(r) & (r > 0)) if int(i) not in skip]
    keep, avoid = list(keep_apart), list(avoid)
    eligible = [int(i) for i in outline if all(differ(FrameSize(int(i), r[i]), k, n) for k in keep)]
    if not eligible:
        return None
    sizes = r[outline]
    big, small = np.quantile(sizes, BIG_PERCENTILE), np.quantile(sizes, SMALLEST_PERCENTILE)
    kept = [k.radius for k in keep if k.sized]
    target = small if kept and max(kept) >= np.median(sizes) else big

    def rank(i: int) -> tuple:
        me = FrameSize(i, r[i])
        near_rejected = any(not differ(me, a, n) for a in avoid)
        same_size = any(not size_differs(me, k) for k in keep)
        band = math.floor(c[i] / CONFIDENCE_STEP + 1e-9)
        return (near_rejected, not t[i], same_size, -band, abs(math.log(r[i] / target)), i)

    return min(eligible, key=rank)


def _spaced(candidates: list[int], n: int) -> list[int]:
    m = len(candidates)
    pos = sorted({min(m - 1, int(round((k + 1) * m / (n + 1)))) for k in range(n)})
    return [candidates[p] for p in pos]


def suggest_frames(radius: Sequence[float], confidence: Sequence[float], *, n: int = 2,
                   exclude: Iterable[int] = (), gate: float = 0.3) -> tuple[int, ...]:
    """Up to *n* frame indices (0-based, ascending) to verify for ML assist.

    Parameters
    ----------
    radius, confidence : sequences of the same length
        Auto's per-frame results; non-finite or non-positive radii are "not measured".
    n : int
        How many frames (the study: 2, at most 3, are enough).
    exclude : iterable of int
        Frames already verified (or otherwise unwanted).
    gate : float
        Only results with ``confidence >= gate`` give sizes.

    Returns
    -------
    tuple of int
        The biggest accepted bubble, then accepted sizes spread down to the 20th
        percentile; without at least ``MIN_ACCEPTED`` accepted frames, frames evenly
        spaced in time.
    """
    r = np.asarray(radius, dtype=float)
    c = np.asarray(confidence, dtype=float)
    if r.ndim != 1 or r.shape != c.shape:
        raise ValueError(f"radius and confidence must be 1-D of the same length, got {r.shape} and {c.shape}")
    if int(n) < 1:
        raise ValueError(f"n must be >= 1, got {n!r}")
    skip = {int(i) for i in exclude}
    free = np.array([i not in skip for i in range(len(r))], bool)
    usable = free & np.isfinite(r) & (r > 0) & np.isfinite(c) & (c >= gate)
    idx = np.flatnonzero(usable)
    if len(idx) < max(MIN_ACCEPTED, int(n)):
        return tuple(_spaced([int(i) for i in np.flatnonzero(free)], int(n)))
    by_size = idx[np.argsort(r[idx], kind="stable")]
    pos = np.round(np.linspace(len(by_size) - 1, SMALLEST_PERCENTILE * (len(by_size) - 1), int(n))).astype(int)
    pick: list[int] = []
    for p in pos:
        while int(by_size[p]) in pick and p > 0:     # tiny sets: step to the next smaller size
            p -= 1
        if int(by_size[p]) not in pick:
            pick.append(int(by_size[p]))
    return tuple(sorted(pick))
