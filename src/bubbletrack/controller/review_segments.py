"""Review segments (pure): runs of frames that share one problem.

A hard recording can flag almost every frame; listing them one by one
buries the few decisions the user has to make.  Each flagged frame gets one
user-facing *category* (its most severe reason), and frames of the same
category that follow each other (a gap of at most ``MAX_GAP`` frames) form a
``ReviewSegment``: one row in the Review dock, one action for all of it.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

from bubbletrack.controller.review_queue import CONTEXT_REASONS, ReviewItem

MAX_GAP = 1  # frames that may be missing inside a segment
SEVERITY_HIGH = 80  # likely wrong (red)
SEVERITY_MEDIUM = 55  # worth a check (amber); below: for information (grey)

Tier = Literal["high", "medium", "low"]


@dataclass(frozen=True)
class Category:
    key: str
    label: str
    reasons: frozenset[str]


# Shown in this order in the filter menu.
CATEGORIES: tuple[Category, ...] = (
    Category("lost", "Lost track", frozenset({"lost", "cloud"})),
    Category("no_result", "No result", frozenset({"rejected"})),
    Category("tracking", "Changed by tracking", frozenset({"disagree", "reacquired"})),
    Category("jump", "Sudden jump", frozenset({"jump"})),
    Category("low_confidence", "Low confidence", frozenset({"low_confidence"})),
    Category("shape", "Irregular shape", frozenset({
        "non_circular", "multiple_edges", "fragmented", "clipped_completed", "shoulder"})),
    Category("collapse", "Near collapse", frozenset({"collapse"})),
    Category("ml", "ML result", frozenset({"ml_outline", "ml_disagrees"})),
)
# A result's source decides its row whatever else is flagged: every ML assist
# result to check (its own outline, or Auto's where ML saw the bubble elsewhere)
# stays in one run instead of splitting by reason; the tooltip names the reason.
SOURCE_CATEGORIES: tuple[tuple[str, str], ...] = (("ml_outline", "ml"), ("ml_disagrees", "ml"))
OTHER = Category("other", "Other", frozenset())
_BY_REASON = {reason: c.key for c in CATEGORIES for reason in c.reasons}
_LABELS = {c.key: c.label for c in (*CATEGORIES, OTHER)}


@dataclass(frozen=True)
class ReviewSegment:
    """Flagged frames of one category, in frame order (gaps allowed)."""

    category: str  # a Category key
    items: tuple[ReviewItem, ...]

    @property
    def frames(self) -> tuple[int, ...]:
        return tuple(item.frame for item in self.items)

    @property
    def start(self) -> int:
        return self.items[0].frame

    @property
    def end(self) -> int:
        return self.items[-1].frame

    @property
    def count(self) -> int:
        return len(self.items)

    @property
    def is_single(self) -> bool:
        return len(self.items) == 1

    @property
    def severity(self) -> int:
        return max(item.severity for item in self.items)

    @property
    def tier(self) -> Tier:
        return tier(self.severity)

    @property
    def label(self) -> str:
        return category_label(self.category)

    def contains(self, frame: int) -> bool:
        """True if *frame* lies in start..end (a member or a gap frame)."""
        return self.start <= frame <= self.end


def category_of(item: ReviewItem) -> str:
    """The category of an item's most severe reason that can list a frame
    (a source category, e.g. ML result, first)."""
    for reason, key in SOURCE_CATEGORIES:
        if reason in item.reasons:
            return key
    deciding = next((r for r in item.reasons if r not in CONTEXT_REASONS),
                    item.reasons[0] if item.reasons else "")
    return _BY_REASON.get(deciding, OTHER.key)


def category_label(key: str) -> str:
    return _LABELS.get(key, OTHER.label)


def tier(severity: int) -> Tier:
    if severity >= SEVERITY_HIGH:
        return "high"
    if severity >= SEVERITY_MEDIUM:
        return "medium"
    return "low"


def build_segments(items: Iterable[ReviewItem], max_gap: int = MAX_GAP,
                   ) -> list[ReviewSegment]:
    """Group *items* into segments, worst first (then by first frame).

    Frames of one category join a segment while at most *max_gap* frames
    are missing in between; a frame of another category in that gap does
    not split it (it is listed in its own segment).
    """
    runs: dict[str, list[list[ReviewItem]]] = {}
    for item in sorted(items, key=lambda i: i.frame):
        groups = runs.setdefault(category_of(item), [])
        if groups and item.frame - groups[-1][-1].frame - 1 <= max_gap:
            groups[-1].append(item)
        else:
            groups.append([item])
    segments = [ReviewSegment(key, tuple(group))
                for key, groups in runs.items() for group in groups]
    return sorted(segments, key=lambda s: (-s.severity, s.start))


def queue_frames(segments: Iterable[ReviewSegment]) -> list[int]:
    """Every listed frame in queue order: segment by segment, in frame order."""
    return [frame for segment in segments for frame in segment.frames]


def segment_of(segments: Sequence[ReviewSegment], frame: int | None, *,
               prefer: ReviewSegment | None = None) -> ReviewSegment | None:
    """The segment to select while *frame* is shown.

    *prefer* (the selected one) holds while its span holds the frame, so
    stepping through a span keeps its row; otherwise the segment that lists
    the frame, else the first one whose span holds it.
    """
    if frame is None:
        return None
    if prefer is not None and prefer in segments and prefer.contains(frame):
        return prefer
    member = next((s for s in segments if frame in s.frames), None)
    if member is not None:
        return member
    return next((s for s in segments if s.contains(frame)), None)


def anchor_frame(segment: ReviewSegment, frame: int) -> int:
    """The member at or before *frame*: where the queue continues from."""
    before = [f for f in segment.frames if f <= frame]
    return before[-1] if before else segment.start
