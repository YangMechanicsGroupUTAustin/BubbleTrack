"""Collapse and cycle detection on a radius time series.

A collapse is a radius minimum that is deep relative to the LOWER of its two
neighbouring peaks (topographic prominence), so a small late cycle counts as
much as the first one.  No smoothing is applied: a real collapse is often a
single-frame minimum, which a median filter would erase (the early-collapse
misses of the dark-area prototype).  Isolated one-frame dips (a wrong single
detection mid-cycle) are rejected by a valley-width test instead.

Invalid samples (NaN or <= 0) inside the bubble's lifetime are interpolated;
the frames before the first and after the last valid sample are outside any
cycle.  When the bubble disappears before the recording ends, its last deep
minimum is a collapse too.
"""

from __future__ import annotations

import numpy as np

from bubbletrack.model.sequence.config import DEFAULT_SEQUENCE_CONFIG, SequenceConfig


def _valid_span(r: np.ndarray) -> tuple[np.ndarray, int, int]:
    valid = np.isfinite(r) & (r > 0)
    idx = np.nonzero(valid)[0]
    if idx.size == 0:
        return valid, -1, -1
    return valid, int(idx[0]), int(idx[-1])


def _interpolated(r: np.ndarray, valid: np.ndarray, first: int, last: int) -> np.ndarray:
    span = np.arange(first, last + 1)
    return np.interp(span, np.nonzero(valid)[0], r[valid])


def _local_minima(s: np.ndarray) -> list[int]:
    """Minima of ``s``; a flat bottom is represented by its middle sample."""
    out, i, n = [], 1, len(s)
    while i < n - 1:
        if s[i] < s[i - 1]:
            j = i
            while j + 1 < n and s[j + 1] == s[i]:
                j += 1
            if j + 1 < n and s[j + 1] > s[i]:
                out.append((i + j) // 2)
            i = j + 1
        else:
            i += 1
    return out


def _side_peak(s: np.ndarray, i: int, step: int) -> float:
    """Highest sample between ``i`` and the first lower sample (or the end) on one side."""
    peak, j = s[i], i + step
    while 0 <= j < len(s) and s[j] >= s[i]:
        peak = max(peak, s[j])
        j += step
    return float(peak)


def _valley_width(s: np.ndarray, i: int, level: float) -> int:
    lo, hi = i, i
    while lo - 1 >= 0 and s[lo - 1] <= level:
        lo -= 1
    while hi + 1 < len(s) and s[hi + 1] <= level:
        hi += 1
    return hi - lo + 1


def _deep_enough(minimum: float, peak: float, cfg: SequenceConfig) -> bool:
    depth = peak - minimum
    return (peak > 0 and depth >= cfg.collapse_min_prominence_px
            and depth / peak >= cfg.collapse_rel_prominence)


def _interior_collapses(s: np.ndarray, cfg: SequenceConfig) -> list[tuple[int, float]]:
    """(index into s, relative depth) of every qualifying interior minimum."""
    found = []
    for i in _local_minima(s):
        peak = min(_side_peak(s, i, -1), _side_peak(s, i, +1))
        if not _deep_enough(s[i], peak, cfg):
            continue
        if _valley_width(s, i, s[i] + cfg.collapse_width_level * (peak - s[i])) < cfg.collapse_min_width:
            continue
        found.append((i, (peak - s[i]) / peak))
    return found


def _enforce_gap(found: list[tuple[int, float]], gap: int) -> list[int]:
    kept: list[int] = []
    for i, _ in sorted(found, key=lambda x: (-x[1], x[0])):
        if all(abs(i - k) >= gap for k in kept):
            kept.append(i)
    return sorted(kept)


def _final_collapse(s: np.ndarray, last_kept: int | None, cfg: SequenceConfig) -> int | None:
    """The bubble vanished after the end of ``s``: its trailing minimum, if deep enough."""
    n = len(s)
    m = n - 1 - int(np.argmin(s[::-1][:3]))  # the smallest of the last three samples (latest on ties)
    start = 0 if last_kept is None else last_kept + 1
    if m <= start or (last_kept is not None and m - last_kept < cfg.collapse_min_gap):
        return None
    return m if _deep_enough(s[m], float(np.max(s[start:m + 1])), cfg) else None


def detect_cycles(radius: np.ndarray, *, config: SequenceConfig = DEFAULT_SEQUENCE_CONFIG
                  ) -> tuple[tuple[int, ...], tuple[tuple[int, int], ...]]:
    """Collapse frames and inclusive ``(start, end)`` cycles of a radius series.

    Parameters
    ----------
    radius : array-like
        Radius per frame; NaN or <= 0 marks frames without a (trusted) bubble.
    config : SequenceConfig
        ``collapse_*`` fields set the depth, width and spacing tests.

    Returns
    -------
    (collapse_frames, cycles)
        Collapse frame indices (ascending), and one ``(start, end)`` range per
        cycle: the first starts at the first valid frame, each collapse ends a
        cycle, the last ends at the last valid frame.
    """
    r = np.asarray(radius, dtype=float).ravel()
    valid, first, last = _valid_span(r)
    if first < 0:
        return (), ()
    if last - first < 2:
        return (), ((first, last),)
    s = _interpolated(r, valid, first, last)
    kept = _enforce_gap(_interior_collapses(s, config), config.collapse_min_gap)
    if last < len(r) - 1:
        final = _final_collapse(s, kept[-1] if kept else None, config)
        if final is not None:
            kept.append(final)
    collapses = tuple(first + k for k in kept)
    cycles, start = [], first
    for c in collapses:
        cycles.append((start, c))
        start = c + 1
    if start <= last:
        cycles.append((start, last))
    return collapses, tuple(cycles)
