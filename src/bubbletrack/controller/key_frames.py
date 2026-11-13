"""Which frames the key-frame strip shows, and what it says about them (pure).

The strip gives an overview of the recording (first fit, Rmax, collapses,
last fit) plus a few frames *to check*: the first frame of the top Review
rows to check (red or amber), so the strip and the Review dock always agree
on what needs a look.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks

from bubbletrack.controller.review_queue import ReviewItem
from bubbletrack.ui.review_text import reason_text

COLLAPSE_PROMINENCE = 0.15  # a collapse dips by at least this fraction of Rmax
MAX_COLLAPSES = 4
MAX_TO_CHECK = 4
MAX_KEY_FRAMES = 12
TO_CHECK = "to check"
# A kind's words in the tooltip (the caption under the image keeps the short kind).
KIND_WORDS: dict[str, str] = {
    "first": "first fit",
    "Rmax": "largest radius (Rmax)",
    "collapse": "collapse",
    "last": "last fit",
    TO_CHECK: TO_CHECK,
}


@dataclass(frozen=True)
class KeyFrame:
    frame: int               # 0-based frame index
    kinds: tuple[str, ...]   # why it was chosen: "first", "Rmax", "collapse", ...


def find_collapses(radius: np.ndarray, max_count: int = MAX_COLLAPSES) -> list[int]:
    """Frames of the deepest local minima of R (bubble collapses), ascending.

    Only minima that dip by ``COLLAPSE_PROMINENCE * Rmax`` count, so fit
    noise is not mistaken for a collapse.  Invalid frames are skipped.
    """
    r = np.asarray(radius, dtype=float)
    idx = np.flatnonzero(np.isfinite(r) & (r > 0))
    if len(idx) < 3:
        return []
    values = r[idx]
    peaks, props = find_peaks(-values, prominence=COLLAPSE_PROMINENCE * values.max())
    deepest = peaks[np.argsort(props["prominences"])[::-1][:max_count]]
    return sorted(int(idx[p]) for p in deepest)


def to_check_frames(rows: Iterable) -> list[int]:
    """The first frame of each Review row to check (``ReviewSegment.to_check``,
    red or amber), in queue order; grey rows are for information only."""
    return [int(row.frames[0]) for row in rows if row.frames and row.to_check]


def choose_key_frames(
    radius: np.ndarray,
    to_check: Sequence[int] = (),
    *,
    max_check: int = MAX_TO_CHECK,
    max_total: int = MAX_KEY_FRAMES,
) -> list[KeyFrame]:
    """First fitted frame, Rmax, collapses and last fitted frame, plus the
    first *max_check* frames of *to_check* (Review's, most likely wrong
    first); ordered by frame.  Nothing else is ever "to check"."""
    r = np.asarray(radius, dtype=float)
    valid = np.isfinite(r) & (r > 0)
    if not valid.any():
        return []
    idx = np.flatnonzero(valid)
    kinds: dict[int, list[str]] = {}

    def add(frame: int, kind: str) -> None:
        kinds.setdefault(int(frame), []).append(kind)

    add(idx[0], "first")
    add(idx[np.argmax(r[idx])], "Rmax")
    for frame in find_collapses(r):
        add(frame, "collapse")
    add(idx[-1], "last")
    n = len(r)
    for frame in list(dict.fromkeys(f for f in to_check if 0 <= f < n))[:max(0, max_check)]:
        add(frame, TO_CHECK)
    return [KeyFrame(f, tuple(kinds[f])) for f in sorted(kinds)][:max_total]


def key_frame_tooltip(frame: int, tag: str, kinds: Sequence[str], radius: float,
                      confidence: float, item: ReviewItem | None) -> str:
    """What a thumbnail shows, in plain words (no engine flag names).

    ``Frame 21 · _021 — first fit`` / ``R = 16.4 px · confidence 0.82`` /
    ``To check: outline not round`` (the last line only for a frame in Review).
    """
    title = f"Frame {frame + 1}" + (f" · {tag}" if tag else "")
    what = " + ".join(KIND_WORDS.get(k, k) for k in kinds)
    result = f"R = {radius:.1f} px" if np.isfinite(radius) and radius > 0 else "No result"
    if np.isfinite(confidence):
        result += f" · confidence {confidence:.2f}"
    lines = [f"{title} — {what}", result]
    if item is not None and item.reasons:
        lines.append("To check: " + ", ".join(reason_text(r, item.confidence)
                                              for r in item.reasons))
    return "\n".join(lines)
