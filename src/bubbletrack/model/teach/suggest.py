"""Which frames to ask the user to verify first (``tools/studies/teach/RESULTS.md``).

Frames spread over bubble sizes teach best: the biggest bubble alone (or only big
ones) gives a model that misses small bubbles.  Sizes come from Auto's accepted
results; when too few are accepted (the recordings that need ML assist most),
frames evenly spaced in time stand in for sizes (the bubble grows and collapses).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

MIN_ACCEPTED = 5
SMALLEST_PERCENTILE = 0.2


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
