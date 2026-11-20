"""Review segments (pure): the rows of the Review dock.

A hard recording can flag almost every frame; listing them one by one
buries the few decisions the user has to make.  Each flagged frame gets one
*category* (its most severe reason), and each category belongs to one of
four *kinds* the user sees ("No bubble found", "Size looks off", "Not sure",
"Before / after the bubble"; the reasons behind them are in the tooltips).
Flagged frames that follow each other (a gap of at most ``MAX_GAP`` frames)
form one ``ReviewSegment``, whatever their categories: one row, coloured by
its worst frame and named by its most common kind, so rows do not overlap.
ML assist's results keep runs of their own (amber, never repainted red by a
neighbour), and the frames before the bubble (``BEFORE``) form one row.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

from bubbletrack.controller.review_queue import BEFORE, CONTEXT_REASONS, ReviewItem

MAX_GAP = 1  # frames that may be missing inside a segment
SEVERITY_HIGH = 80  # likely wrong (red)
SEVERITY_MEDIUM = 55  # worth a check (amber); below: a quick look (grey)

Tier = Literal["high", "medium", "low"]


@dataclass(frozen=True)
class Category:
    key: str
    label: str
    reasons: frozenset[str]


# The categories behind the four kinds the user sees (``KINDS``); the
# reasons behind them are in the tooltips.
CATEGORIES: tuple[Category, ...] = (
    Category("lost", "Auto lost the bubble", frozenset({"lost", "cloud"})),
    Category("no_result", "No result", frozenset({"rejected"})),
    Category("tracking", "Tracking changed it", frozenset({"disagree", "reacquired"})),
    Category("jump", "Radius jumps", frozenset({"jump"})),
    Category("low_confidence", "Not sure: low confidence", frozenset({"low_confidence"})),
    Category("shape", "Outline not round", frozenset({
        "non_circular", "multiple_edges", "fragmented", "shoulder"})),
    Category("clipped", "Cut by the image edge", frozenset({
        "clipped_not_round", "clipped_unmeasurable", "clipped_completed"})),
    Category("collapse", "Near collapse: tiny bubble", frozenset({"collapse"})),
    Category("ml", "Outline from ML assist", frozenset({"ml_outline", "ml_disagrees"})),
    Category(BEFORE, "Before the bubble", frozenset({BEFORE})),
)
# A result's source decides its row whatever else is flagged: every ML assist
# result to check (its own outline, or Auto's where ML saw the bubble elsewhere)
# stays in one run instead of splitting by reason; the tooltip names the reason.
# A frame Auto could not measure because the image edge cuts it is "Cut by the
# image edge", not just "No result": the row says why.
SOURCE_CATEGORIES: tuple[tuple[str, str], ...] = (
    ("ml_outline", "ml"), ("ml_disagrees", "ml"), ("clipped_unmeasurable", "clipped"))
OTHER = Category("other", "Worth a look", frozenset())
_BY_REASON = {reason: c.key for c in CATEGORIES for reason in c.reasons}
_LABELS = {c.key: c.label for c in (*CATEGORIES, OTHER)}
# ML assist's results are always checked, but they are not likely wrong: amber.
_CAPPED = {"ml": SEVERITY_HIGH - 1}  # category -> the highest severity it shows


@dataclass(frozen=True)
class Kind:
    """What the user sees of a category: one of four kinds of problem."""

    key: str
    label: str  # the filter's and the row's name
    short: str  # inside a mixed row's name: "Mostly not sure (11 not found)"
    categories: frozenset[str]


NOT_FOUND, SIZE_OFF, NOT_SURE = "not_found", "size_off", "not_sure"
# Shown in this order in the filter menu.
KINDS: tuple[Kind, ...] = (
    Kind(NOT_FOUND, "No bubble found", "not found",
         frozenset({"lost", "no_result", "clipped"})),
    Kind(SIZE_OFF, "Size looks off", "size off", frozenset({"jump", "tracking"})),
    Kind(NOT_SURE, "Not sure", "not sure",
         frozenset({"low_confidence", "shape", "ml", "collapse", OTHER.key})),
    Kind(BEFORE, "Before / after the bubble", "before / after", frozenset({BEFORE})),
)
_KIND_OF = {category: k.key for k in KINDS for category in k.categories}
_KINDS = {k.key: k for k in KINDS}
_KIND_RANK = {k.key: n for n, k in enumerate(KINDS)}


@dataclass(frozen=True)
class ReviewSegment:
    """Flagged frames of one span, in frame order (gaps allowed)."""

    category: str  # the Category key of its worst frame (the row opens on it)
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
    def entry(self) -> int:
        """The frame a selected row opens on: its first frame of the kind
        that names it, so the card says the row's problem."""
        return next(i.frame for i in self.items if category_of(i) == self.category)

    @property
    def count(self) -> int:
        return len(self.items)

    @property
    def is_single(self) -> bool:
        return len(self.items) == 1

    @property
    def severity(self) -> int:
        return max(item_severity(item) for item in self.items)

    @property
    def tier(self) -> Tier:
        return tier(self.severity)

    @property
    def to_check(self) -> bool:
        """A red or amber row (the filmstrip's key frames); Review counts the
        grey rows' frames to check as well."""
        return self.tier != "low"

    @property
    def kind_counts(self) -> tuple[tuple[str, int], ...]:
        """(kind, frames) of every kind in the row, the most common first
        (a tie: the kind of the worse frame first)."""
        counts = Counter(kind_of(item) for item in self.items)
        worst: dict[str, int] = {}
        for item in self.items:
            key = kind_of(item)
            worst[key] = max(worst.get(key, 0), item_severity(item))
        return tuple(sorted(counts.items(),
                            key=lambda kv: (-kv[1], -worst[kv[0]], _KIND_RANK[kv[0]])))

    @property
    def kind(self) -> str:
        """The row's most common kind (it names the row)."""
        return self.kind_counts[0][0]

    @property
    def label(self) -> str:
        """The row's name: its kind ('Not sure'), or for a mix the most
        common kind and the others ('Mostly not sure (11 not found)')."""
        (main, _), *others = self.kind_counts
        if not others:
            return kind_label(main)
        rest = ", ".join(f"{n} {_KINDS[key].short}" for key, n in others)
        return f"Mostly {kind_label(main).lower()} ({rest})"

    @property
    def categories(self) -> tuple[str, ...]:
        """Every category in the row, the worst first."""
        ranked = sorted(self.items, key=lambda i: -item_severity(i))
        return tuple(dict.fromkeys(category_of(i) for i in ranked))

    def contains(self, frame: int) -> bool:
        """True if *frame* lies in start..end (a member or a gap frame)."""
        return self.start <= frame <= self.end


def category_of(item: ReviewItem) -> str:
    """The category of an item's most severe reason that can list a frame
    (a source category, e.g. the ML assist outline, first)."""
    for reason, key in SOURCE_CATEGORIES:
        if reason in item.reasons:
            return key
    deciding = next((r for r in item.reasons if r not in CONTEXT_REASONS),
                    item.reasons[0] if item.reasons else "")
    return _BY_REASON.get(deciding, OTHER.key)


def kind_of(item: ReviewItem) -> str:
    """The kind the user sees for *item*.  A frame cut by the image edge that
    has a result (its visible wall is not round) is "Not sure", not "No
    bubble found"."""
    category = category_of(item)
    if category == "clipped" and "rejected" not in item.reasons:
        return NOT_SURE
    return _KIND_OF.get(category, NOT_SURE)


def kind_label(key: str) -> str:
    kind = _KINDS.get(key)
    return kind.label if kind is not None else _KINDS[NOT_SURE].label


def item_severity(item: ReviewItem) -> int:
    """The severity a frame shows with (an ML assist result is never red)."""
    cap = _CAPPED.get(category_of(item))
    return item.severity if cap is None else min(item.severity, cap)


def category_label(key: str) -> str:
    return _LABELS.get(key, OTHER.label)


def tier(severity: int) -> Tier:
    if severity >= SEVERITY_HIGH:
        return "high"
    if severity >= SEVERITY_MEDIUM:
        return "medium"
    return "low"


_TIER_RANK = {"high": 0, "medium": 1, "low": 2}
_GROUPS = {"ml": "ml", BEFORE: BEFORE}  # categories that keep rows of their own


def build_segments(items: Iterable[ReviewItem], max_gap: int = MAX_GAP,
                   ) -> list[ReviewSegment]:
    """Group *items* into segments: likely wrong first, then worth a check,
    then the rest (the row's dot colour), each in frame order, so working
    down the list moves forward through the recording.

    Flagged frames join a segment while at most *max_gap* frames are
    missing in between, whatever their categories.  Two sources keep rows
    of their own (other frames inside their span neither split nor join
    them): ML assist's results, so one red frame next to them does not
    repaint a long amber run, and the frames before the bubble (one row).
    """
    runs: dict[str, list[list[ReviewItem]]] = {}
    for item in sorted(items, key=lambda i: i.frame):
        group = _GROUPS.get(category_of(item), "")
        groups = runs.setdefault(group, [])
        if groups and (group == BEFORE or item.frame - groups[-1][-1].frame - 1 <= max_gap):
            groups[-1].append(item)
        else:
            groups.append([item])
    segments = [ReviewSegment(_worst_category(run), tuple(run))
                for groups in runs.values() for run in groups]
    return sorted(segments, key=lambda s: (_TIER_RANK[s.tier], s.start, -s.severity))


def _worst_category(items: Sequence[ReviewItem]) -> str:
    """The category of the most severe item (the earliest one on a tie)."""
    return category_of(max(items, key=lambda i: (item_severity(i), -i.frame)))


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


def next_in_segment(segment: ReviewSegment, frame: int, listed: Collection[int]) -> int | None:
    """After acting on *frame*: the segment's next member that is still
    *listed* (then the first one before it); None once none is left."""
    remaining = [f for f in segment.frames if f != frame and f in listed]
    return next((f for f in remaining if f > frame), remaining[0] if remaining else None)
