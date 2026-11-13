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
NOT_A_BUBBLE_HINT = "Not a bubble? Shift+drag on the chart > Mark as no bubble"  # image tip
REVIEW_NO_BUBBLE_HINT = "Not a bubble? No bubble (N); Shift+N for the whole row"

REASON_LABELS: dict[str, str] = {  # plain words; the flag names stay internal
    "lost": "Auto lost the bubble",
    "cloud": "bubble broke into a cloud",
    "rejected": "no result",
    GUESS: "Auto has a guess",
    "disagree": "tracking changed it",
    "reacquired": "bubble found again",
    "jump": "radius jumps",
    "low_confidence": "low confidence",
    "multiple_edges": "several edges",
    "non_circular": "outline not round",
    "fragmented": "outline in pieces",
    "clipped_completed": "cut by the image edge",
    "shoulder": "blurred wall",
    "collapse": "near collapse",
    "ml_outline": "outline from ML assist",
    "ml_disagrees": "ML assist sees it elsewhere",
    "ml_unfamiliar": "unlike the frames ML was taught",
}
GUESS_SUFFIX = " — Auto has a guess"


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


def problem_text(segment: ReviewSegment) -> str:
    """What is wrong with a row, in plain words ('No result — Auto has a guess')."""
    label = segment.label
    if segment.category == "no_result" and _guess_count(segment):
        label += GUESS_SUFFIX
    return label


def span_text(segment: ReviewSegment) -> str:
    """'Frame 5' or 'Frames 10–44' (1-based), for messages."""
    if segment.is_single:
        return f"Frame {segment.start + 1}"
    return f"Frames {segment.start + 1}{DASH}{segment.end + 1}"


def segment_text(segment: ReviewSegment, images: Sequence[str]) -> str:
    """The row as one line (screen readers, tests)."""
    parts = [segment_title(segment, images), problem_text(segment)]
    if not segment.is_single:
        parts.append(f"{segment.count} frames")
    return SEP.join(parts)


def progress_text(left: int, total: int) -> str:
    """'12 of 48 frames to check'; '48 frames to check' before any is done."""
    if total <= 0:
        return "Nothing to check"
    if left <= 0:
        return f"All {total} checked" if total > 1 else "Checked"
    noun = "frame" if total == 1 else "frames"
    return f"{left} {noun} to check" if left == total else f"{left} of {total} {noun} to check"


def position_text(segment: ReviewSegment, frame: int) -> str:
    """How much of the row is left: '47 left in this row' ('' for one frame).

    Checked frames leave the row, so a position ('1 of 47') would always
    read 1; what is left is what the user wants to know.
    """
    if segment.is_single:
        return ""
    return f"{segment.count} left in this row"


def tooltip_lines(segment: ReviewSegment, images: Sequence[str]) -> list[str]:
    """Everything behind a row: span, files, reasons, confidence, keys."""
    if segment.is_single:
        item = segment.items[0]
        why = ", ".join(reason_text(r, item.confidence) for r in item.reasons)
        return [segment_title(segment, images), problem_text(segment), f"Why: {why}",
                *guess_lines(segment)]
    first, last = file_tag(images, segment.start), file_tag(images, segment.end)
    head = segment_title(segment, images)
    if first and last:
        head += f"{SEP}files {first}{DASH}{last}"
    lines = [head, f"{problem_text(segment)}{SEP}{segment.count} flagged frames",
             "Why: " + ", ".join(f"{REASON_LABELS.get(r, r.replace('_', ' '))} ×{n}"
                                 for r, n in _reason_counts(segment))]
    conf = [i.confidence for i in segment.items if np.isfinite(i.confidence)]
    if conf:
        lines.append(f"Confidence {min(conf):.2f}{DASH}{max(conf):.2f}")
    lines.extend(guess_lines(segment))
    lines.append(STEP_HINT)
    return lines


def guess_lines(segment: ReviewSegment) -> list[str]:
    """What Accept does on a row with Auto's guesses (none: no line).

    A guess becomes a result once accepted: Accept stores the guess on
    screen; Accept all stores every guess of the row (it asks first about
    guesses nobody has looked at).
    """
    guessed = _guess_count(segment)
    if not guessed:
        return []
    if segment.is_single:
        return ["Auto's guess (dashed outline): Accept (A) stores it as the result",
                REVIEW_NO_BUBBLE_HINT]
    return [f"Auto's guess on {guessed} of {segment.count} frames (dashed): Accept (A) "
            "stores the one on screen, Accept all (Shift+A) every one", REVIEW_NO_BUBBLE_HINT]


def guess_hint(item: ReviewItem | None, segment: ReviewSegment | None) -> str:
    """What A does on the shown frame if it shows Auto's guess ('' otherwise)."""
    if segment is None or item is None or GUESS not in item.reasons:
        return ""
    more = _guess_count(segment) - 1
    return "A accepts the guess shown" + (
        f"; {more} more guess{'es' if more != 1 else ''} in this row" if more else "")


def accept_all_question(frames: int, unseen: int) -> str:
    """Accept all's question when it would store guesses nobody has looked at:
    'Accept 229 frames? 213 of them are Auto's guesses you have not looked at.'"""
    if frames == 1:
        return "Accept 1 frame? It is Auto's guess you have not looked at."
    what = "is Auto's guess" if unseen == 1 else "are Auto's guesses"
    return f"Accept {frames} frames? {unseen} of them {what} you have not looked at."


def _guess_count(segment: ReviewSegment) -> int:
    return sum(GUESS in item.reasons for item in segment.items)


def detail_text(item: ReviewItem | None, frame: int, images: Sequence[str],
                limit: int | None = DETAIL_REASONS,
                segment: ReviewSegment | None = None) -> str:
    """The shown *frame* and why it is listed (its *item*, None if it is
    not): the leading *limit* reasons, None = all of them.  With the
    selected *segment*, what A does there if the row has Auto's guesses."""
    if item is None:
        text = f"{frame_title(frame, images)} — not flagged"
    else:
        text = item_text(item, images, limit)
    hint = guess_hint(item, segment)
    return f"{text}. {hint}" if hint else text


def _reason_counts(segment: ReviewSegment) -> list[tuple[str, int]]:
    """Reasons by how many frames have them (ties: the more severe first)."""
    counts = Counter(r for item in segment.items for r in item.reasons)
    first_seen = {r: k for k, r in enumerate(dict.fromkeys(
        r for item in segment.items for r in item.reasons))}
    return sorted(counts.items(), key=lambda kv: (-kv[1], first_seen[kv[0]]))
