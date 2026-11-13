"""Review segments: consecutive frames with the same problem become one row."""

from __future__ import annotations

import pytest

from bubbletrack.controller.review_queue import ReviewItem
from bubbletrack.controller.review_segments import (
    CATEGORIES,
    ReviewSegment,
    build_segments,
    category_label,
    category_of,
    next_in_segment,
    queue_frames,
    segment_of,
    tier,
)


def _item(frame: int, *reasons: str, severity: int = 60) -> ReviewItem:
    return ReviewItem(frame, reasons, severity)


def _spans(segments) -> list[tuple[str, int, int, int]]:
    return [(s.category, s.start, s.end, s.count) for s in segments]


class TestCategories:
    @pytest.mark.parametrize("reason, category", [
        ("lost", "lost"), ("cloud", "lost"),
        ("rejected", "no_result"),
        ("disagree", "tracking"), ("reacquired", "tracking"),
        ("jump", "jump"),
        ("low_confidence", "low_confidence"),
        ("non_circular", "shape"), ("multiple_edges", "shape"),
        ("collapse", "collapse"),
        ("before", "before"),
        ("something_new", "other"),
    ])
    def test_reason_to_category(self, reason, category):
        assert category_of(_item(0, reason)) == category

    def test_the_most_severe_listed_reason_decides(self):
        assert category_of(_item(0, "lost", "rejected", "low_confidence")) == "lost"
        assert category_of(_item(0, "low_confidence", "non_circular")) == "low_confidence"

    def test_routine_context_reasons_never_decide(self):
        # fragmented outranks collapse, but alone it never lists a frame
        assert category_of(_item(0, "fragmented", "collapse")) == "collapse"
        assert category_of(_item(0, "fragmented", "shoulder")) == "shape"

    def test_a_handful_of_named_categories_in_plain_words(self):
        assert 5 <= len(CATEGORIES) <= 9
        assert category_label("lost") == "Auto lost the bubble"
        assert category_label("low_confidence") == "Not sure: low confidence"
        assert category_label("other") == "Worth a look"
        assert all(category_label(c.key) == c.label for c in CATEGORIES)
        jargon = ("segment", "guided", "disagree", "reacquired", "ML result", "track ")
        assert not any(word in c.label for c in CATEGORIES for word in jargon)


class TestGrouping:
    def test_no_items_no_segments(self):
        assert build_segments([]) == []

    def test_consecutive_frames_with_the_same_problem_merge(self):
        items = [_item(f, "low_confidence") for f in (3, 4, 5)]
        segments = build_segments(items)
        assert _spans(segments) == [("low_confidence", 3, 5, 3)]
        assert segments[0].frames == (3, 4, 5)

    def test_a_gap_of_one_frame_is_bridged_but_not_two(self):
        items = [_item(f, "lost") for f in (10, 12, 15)]
        assert _spans(build_segments(items)) == [("lost", 10, 12, 2), ("lost", 15, 15, 1)]

    def test_max_gap_is_configurable(self):
        items = [_item(f, "lost") for f in (10, 12)]
        assert len(build_segments(items, max_gap=0)) == 2

    def test_neighbouring_frames_with_other_problems_join_the_row(self):
        """One row per span: the worst problem names it, the others ride along."""
        items = [_item(3, "low_confidence", severity=70), _item(4, "lost", severity=100)]
        (segment,) = build_segments(items)
        assert _spans([segment]) == [("lost", 3, 4, 2)]
        assert segment.categories == ("lost", "low_confidence")

    def test_a_row_opens_on_the_frame_that_names_it(self):
        items = [_item(3, "low_confidence", severity=70), _item(4, "lost", severity=100),
                 _item(5, "lost", severity=100)]
        (segment,) = build_segments(items)
        assert segment.entry == 4
        assert build_segments([_item(7, "lost")])[0].entry == 7

    def test_an_interleaved_frame_of_another_kind_joins_the_run(self):
        items = [_item(10, "lost", severity=100), _item(11, "low_confidence"),
                 _item(12, "lost", severity=100)]
        (segment,) = build_segments(items)
        assert (segment.category, segment.frames) == ("lost", (10, 11, 12))

    def test_no_overlapping_rows(self):
        """The hard recordings listed 'Frames 26-48' and 'Frame 31' inside it."""
        items = ([_item(f, "lost", severity=100) for f in range(26, 49) if f not in (31, 35)]
                 + [_item(f, "disagree", severity=85) for f in (31, 35, 41, 42)]
                 + [_item(f, "low_confidence") for f in (5, 6, 8, 9, 10)]
                 + [_item(f, "non_circular", severity=55) for f in (7, 9, 11, 12)])
        segments = build_segments(items)
        spans = sorted((s.start, s.end) for s in segments)
        assert spans == [(5, 12), (26, 48)]
        assert all(a[1] < b[0] for a, b in zip(spans, spans[1:]))

    def test_frames_before_the_bubble_form_one_row_of_their_own(self):
        items = [_item(0, "before", severity=10), _item(1, "rejected", severity=90),
                 _item(2, "before", severity=10), _item(5, "before", severity=10),
                 _item(6, "rejected", severity=90)]
        segments = build_segments(items)
        before = [s for s in segments if s.category == "before"]
        assert [s.frames for s in before] == [(0, 2, 5)]
        assert before[0].label == "Before the bubble" and before[0].tier == "low"
        assert [s.frames for s in segments if s.category != "before"] == [(1,), (6,)]

    def test_ml_assist_rows_are_worth_a_check_not_likely_wrong(self):
        """ML assist's own outlines are always checked, but they are not errors."""
        items = [_item(f, "ml_outline", "low_confidence", "non_circular", severity=92)
                 for f in (3, 4)]
        (segment,) = build_segments(items)
        assert segment.category == "ml" and segment.tier == "medium"

    def test_ml_assist_results_keep_their_own_rows(self):
        """A lost frame next to ML's outlines must not turn a long ML row red."""
        items = [_item(f, "ml_outline", "low_confidence", severity=92) for f in range(3, 30)]
        segments = build_segments(items + [_item(30, "lost", severity=100)])
        assert [(s.category, s.tier, s.count) for s in segments] == [
            ("lost", "high", 1), ("ml", "medium", 27)]

    def test_input_order_does_not_matter(self):
        items = [_item(5, "lost"), _item(3, "lost"), _item(4, "lost")]
        assert build_segments(items) == build_segments(sorted(items, key=lambda i: i.frame))

    def test_single_frame_segment(self):
        (segment,) = build_segments([_item(44, "jump", severity=80)])
        assert (segment.start, segment.end, segment.count) == (44, 44, 1)
        assert segment.is_single and segment.severity == 80
        assert segment.label == "Radius jumps"


class TestRanking:
    def test_likely_wrong_rows_first_each_tier_in_frame_order(self):
        """Working down the list moves forward through the recording."""
        items = [
            _item(1, "low_confidence", severity=60), _item(2, "low_confidence", severity=88),
            _item(20, "lost", severity=100),
            _item(40, "non_circular", severity=55), _item(30, "jump", severity=88),
            _item(50, "collapse", severity=35), _item(45, "low_confidence", severity=70),
        ]
        segments = build_segments(items)
        assert [s.start for s in segments] == [1, 20, 30, 40, 45, 50]
        assert [s.tier for s in segments] == ["high"] * 3 + ["medium"] * 2 + ["low"]
        assert segments[0].severity == 88  # the worst frame of 1-2 makes the row red

    def test_queue_frames_lists_each_segment_in_frame_order(self):
        items = [_item(5, "lost", severity=100), _item(6, "lost", severity=100),
                 _item(1, "low_confidence"), _item(2, "low_confidence")]
        assert queue_frames(build_segments(items)) == [5, 6, 1, 2]


class TestLookup:
    @pytest.fixture
    def segments(self):
        return build_segments([_item(10, "lost", severity=100), _item(12, "lost", severity=100),
                               _item(11, "low_confidence"), _item(30, "jump", severity=80),
                               _item(0, "before", severity=10), _item(1, "rejected", severity=90),
                               _item(2, "before", severity=10)])

    def test_member_frame_finds_its_own_segment(self, segments):
        assert segment_of(segments, 11).frames == (10, 11, 12)
        assert segment_of(segments, 30).category == "jump"
        assert segment_of(segments, 1).category == "no_result"  # inside the before span
        assert segment_of(segments, 2).category == "before"

    def test_gap_frame_inside_a_span_finds_that_segment(self):
        segments = build_segments([_item(10, "lost"), _item(12, "lost")])
        assert segment_of(segments, 11) == segments[0]

    def test_preferred_segment_is_kept_while_its_span_holds_the_frame(self, segments):
        lost = segment_of(segments, 10)
        assert segment_of(segments, 11, prefer=lost) == lost

    def test_frames_outside_every_span(self, segments):
        assert segment_of(segments, 20) is None
        assert segment_of(segments, None) is None
        assert segment_of(segments, 5) is None

    def test_next_in_segment_is_the_next_member_still_listed(self):
        (segment,) = build_segments([_item(f, "lost") for f in (10, 12, 13, 14)])
        assert next_in_segment(segment, 12, {10, 13, 14}) == 13
        assert next_in_segment(segment, 12, {10, 14}) == 14  # 13 was settled meanwhile
        assert next_in_segment(segment, 11, {10, 12, 13, 14}) == 12  # from a gap frame

    def test_next_in_segment_wraps_to_the_members_before_then_ends(self):
        (segment,) = build_segments([_item(f, "lost") for f in (10, 12, 13)])
        assert next_in_segment(segment, 13, {10, 12}) == 10
        assert next_in_segment(segment, 13, {13, 40}) is None  # only itself, or other rows


class TestTier:
    @pytest.mark.parametrize("severity, expected", [
        (100, "high"), (80, "high"), (79, "medium"), (55, "medium"), (54, "low"), (0, "low"),
    ])
    def test_bands(self, severity, expected):
        assert tier(severity) == expected

    def test_segment_tier_follows_its_worst_frame(self):
        segment = ReviewSegment("lost", (ReviewItem(1, ("lost",), 30),
                                         ReviewItem(2, ("lost",), 100)))
        assert segment.tier == "high"
