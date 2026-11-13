"""Words of the Review dock (pure): row titles, tooltips, the detail line.

Rows stay short ("Frames 120–180", "Lost track"); the reasons behind them
and the confidence only appear on demand (tooltip, selected row).
"""

from __future__ import annotations

import os
import re
from collections import Counter
from collections.abc import Sequence

import numpy as np

from bubbletrack.controller.review_queue import GUESS, ReviewItem
from bubbletrack.controller.review_segments import ReviewSegment

_TRAILING_DIGITS = re.compile(r"(\d+)$")
_VIDEO_PREFIX = "video:"
SEP = " · "  # middle dot
DASH = "–"  # en dash, for spans
STEP_HINT = ("←/→ step through these frames, a click on the row goes back to "
             "its first one; J/K move between rows.")
DETAIL_REASONS = 3  # the detail line names this many; the tooltip has them all

REASON_LABELS: dict[str, str] = {
    "lost": "lost track",
    "cloud": "bubble cloud",
    "rejected": "no result",
    GUESS: "Auto's guess available",
    "disagree": "changed by tracking",
    "reacquired": "found again after a loss",
    "jump": "sudden jump",
    "low_confidence": "low confidence",
    "multiple_edges": "multiple edges",
    "non_circular": "not circular",
    "fragmented": "fragmented",
    "clipped_completed": "clipped (completed)",
    "shoulder": "shoulder",
    "collapse": "near collapse",
}


def file_number(path: str, index: int) -> str:
    """Trailing digits of the image file name, else the 1-based frame index."""
    if path.startswith(_VIDEO_PREFIX):
        return str(index + 1)
    stem = os.path.splitext(os.path.basename(path))[0]
    match = _TRAILING_DIGITS.search(stem)
    return match.group(1) if match else str(index + 1)


def file_tag(images: Sequence[str], frame: int) -> str:
    """'_054' for a numbered image file; '' when it would only repeat the index."""
    if not 0 <= frame < len(images) or images[frame].startswith(_VIDEO_PREFIX):
        return ""
    stem = os.path.splitext(os.path.basename(images[frame]))[0]
    return f"_{file_number(images[frame], frame)}" if _TRAILING_DIGITS.search(stem) else ""


def reason_text(reason: str, confidence: float) -> str:
    label = REASON_LABELS.get(reason, reason.replace("_", " "))
    if reason == "low_confidence" and np.isfinite(confidence):
        return f"{label} {confidence:.2f}"
    return label


def frame_title(frame: int, images: Sequence[str]) -> str:
    """'Frame 44 · _054' (1-based frame, file number when there is one)."""
    tag = file_tag(images, frame)
    return f"Frame {frame + 1}" + (f"{SEP}{tag}" if tag else "")


def item_text(item: ReviewItem, images: Sequence[str], limit: int | None = None) -> str:
    """'Frame 23 · _033 — low confidence 0.21, not circular'.

    With *limit*, only the first *limit* reasons are named ('+2 more').
    """
    named = item.reasons if limit is None else item.reasons[:limit]
    reasons = ", ".join(reason_text(r, item.confidence) for r in named)
    more = len(item.reasons) - len(named)
    return f"{frame_title(item.frame, images)} — {reasons}" + (f" +{more} more" if more else "")


def segment_title(segment: ReviewSegment, images: Sequence[str]) -> str:
    """'Frame 44 · _054' for one frame, 'Frames 120–180' for a run."""
    if segment.is_single:
        return frame_title(segment.start, images)
    return f"Frames {segment.start + 1}{DASH}{segment.end + 1}"


def segment_text(segment: ReviewSegment, images: Sequence[str]) -> str:
    """The row as one line (screen readers, tests)."""
    parts = [segment_title(segment, images), segment.label]
    if not segment.is_single:
        parts.append(f"{segment.count} frames")
    return SEP.join(parts)


def summary_text(segments: int, frames: int) -> str:
    """'12 segments · 340 frames' ('' when there is nothing to review)."""
    if not segments:
        return ""
    return (f"{segments} segment{'s' if segments != 1 else ''}{SEP}"
            f"{frames} frame{'s' if frames != 1 else ''}")


def tooltip_lines(segment: ReviewSegment, images: Sequence[str]) -> list[str]:
    """Everything behind a row: span, files, reasons, confidence, keys."""
    if segment.is_single:
        item = segment.items[0]
        why = ", ".join(reason_text(r, item.confidence) for r in item.reasons)
        return [segment_title(segment, images), segment.label, f"Why: {why}",
                *guess_lines(segment)]
    first, last = file_tag(images, segment.start), file_tag(images, segment.end)
    head = segment_title(segment, images)
    if first and last:
        head += f"{SEP}files {first}{DASH}{last}"
    lines = [head, f"{segment.label}{SEP}{segment.count} flagged frames",
             "Why: " + ", ".join(f"{REASON_LABELS.get(r, r.replace('_', ' '))} ×{n}"
                                 for r, n in _reason_counts(segment))]
    conf = [i.confidence for i in segment.items if np.isfinite(i.confidence)]
    if conf:
        lines.append(f"Confidence {min(conf):.2f}{DASH}{max(conf):.2f}")
    lines.extend(guess_lines(segment))
    lines.append(STEP_HINT)
    return lines


def guess_lines(segment: ReviewSegment) -> list[str]:
    """What Accept does on a row with Auto's guesses (none: no line)."""
    guessed = sum(GUESS in item.reasons for item in segment.items)
    if not guessed:
        return []
    if segment.is_single:
        return ["Accept (A) stores Auto's guess (dashed outline) as the result"]
    empty = sum("rejected" in item.reasons and GUESS not in item.reasons
                for item in segment.items)  # no result and no guess: "no bubble"
    text = f"Accept (A) stores Auto's guess on {guessed} frame{'s' if guessed != 1 else ''}"
    return [text + (f"; {empty} without one are locked as no bubble" if empty else "")]


def detail_text(item: ReviewItem | None, frame: int, images: Sequence[str],
                limit: int | None = DETAIL_REASONS) -> str:
    """The shown *frame* and why it is listed (its *item*, None if it is
    not): the leading *limit* reasons, None = all of them."""
    if item is None:
        return f"{frame_title(frame, images)} — not flagged"
    return item_text(item, images, limit)


def _reason_counts(segment: ReviewSegment) -> list[tuple[str, int]]:
    """Reasons by how many frames have them (ties: the more severe first)."""
    counts = Counter(r for item in segment.items for r in item.reasons)
    first_seen = {r: k for k, r in enumerate(dict.fromkeys(
        r for item in segment.items for r in item.reasons))}
    return sorted(counts.items(), key=lambda kv: (-kv[1], first_seen[kv[0]]))
