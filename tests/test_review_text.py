"""Words of the Review dock: row titles, tooltips, the detail line, the header."""

from __future__ import annotations

from bubbletrack.controller.review_queue import ReviewItem
from bubbletrack.controller.review_segments import ReviewSegment
from bubbletrack.ui.review_text import (
    accept_all_question,
    detail_text,
    file_number,
    file_tag,
    item_text,
    position_text,
    problem_text,
    progress_text,
    reason_text,
    segment_text,
    segment_title,
    span_text,
    tooltip_lines,
)

IMAGES = tuple(f"C:/d/frame_{i + 11:03d}.tif" for i in range(200))  # frame 0 = _011


def _segment(category, *frames, reasons=("lost",), confidence=float("nan")):
    return ReviewSegment(category, tuple(ReviewItem(f, reasons, 100, confidence)
                                         for f in frames))


class TestFileNumbers:
    def test_file_number_from_trailing_digits(self):
        assert file_number("C:/data/run_7/frame_033.tiff", 22) == "033"
        assert file_number("img00012.tif", 3) == "00012"

    def test_file_number_falls_back_to_one_based_index(self):
        assert file_number("C:/data/bubble.tif", 4) == "5"
        assert file_number("video:12", 12) == "13"

    def test_file_tag_only_for_numbered_files(self):
        assert file_tag(IMAGES, 43) == "_054"
        assert file_tag(("C:/d/bubble.tif",), 0) == ""  # would just repeat the index
        assert file_tag(("video:0",), 0) == ""
        assert file_tag((), 3) == ""


class TestReasons:
    def test_friendly_names(self):
        assert reason_text("non_circular", 0.9) == "outline not round"
        assert reason_text("lost", 0.9) == "Auto lost the bubble"
        assert reason_text("disagree", 0.9) == "tracking changed it"
        assert reason_text("rejected", 0.1) == "no result"
        assert reason_text("brand_new_flag", 0.9) == "brand new flag"

    def test_low_confidence_shows_the_value(self):
        assert reason_text("low_confidence", 0.214) == "low confidence 0.21"
        assert reason_text("low_confidence", float("nan")) == "low confidence"

    def test_item_text(self):
        item = ReviewItem(22, ("low_confidence", "non_circular"), 80, 0.214)
        text = item_text(item, ("x",) * 22 + ("C:/d/frame_033.tif",))
        assert text == "Frame 23 \u00b7 _033 \u2014 low confidence 0.21, outline not round"

    def test_item_text_without_a_file_list(self):
        assert item_text(ReviewItem(4, ("jump",), 80), ()) == "Frame 5 \u2014 radius jumps"


class TestRows:
    def test_single_frame_title_carries_the_file_number(self):
        assert segment_title(_segment("lost", 43), IMAGES) == "Frame 44 \u00b7 _054"
        assert segment_title(_segment("lost", 43), ()) == "Frame 44"

    def test_multi_frame_title_is_the_span(self):
        assert segment_title(_segment("lost", 119, 121, 179), IMAGES) == "Frames 120\u2013180"

    def test_row_text_names_the_problem_and_the_count(self):
        assert segment_text(_segment("lost", 43), IMAGES) == \
            "Frame 44 \u00b7 _054 \u00b7 Auto lost the bubble"
        assert segment_text(_segment("jump", 9, 10), IMAGES) == \
            "Frames 10\u201311 \u00b7 Radius jumps \u00b7 2 frames"

    def test_a_no_result_row_says_when_auto_has_a_guess(self):
        plain = _segment("no_result", 3, 4, reasons=("rejected",))
        guessed = ReviewSegment("no_result", (ReviewItem(3, ("rejected",), 90),
                                              ReviewItem(4, ("rejected", "guess"), 89)))
        assert problem_text(plain) == "No result"
        assert problem_text(guessed) == "No result \u2014 Auto has a guess"

    def test_progress(self):
        assert progress_text(48, 48) == "48 frames to check"
        assert progress_text(12, 48) == "12 of 48 frames to check"
        assert progress_text(0, 48) == "All 48 checked"
        assert progress_text(0, 0) == "Nothing to check"

    def test_position_and_span(self):
        segment = _segment("lost", 10, 12, 13)
        assert position_text(segment, 12) == "3 left in this row"
        assert position_text(segment, 11) == "3 left in this row"  # a gap frame too
        assert position_text(_segment("lost", 4), 4) == ""
        assert span_text(segment) == "Frames 11\u201314"
        assert span_text(_segment("lost", 4)) == "Frame 5"

    def test_accept_all_question(self):
        assert accept_all_question(229, 213) == \
            "Accept 229 frames? 213 of them are Auto's guesses you have not looked at."
        assert accept_all_question(5, 1) == \
            "Accept 5 frames? 1 of them is Auto's guess you have not looked at."


class TestDetails:
    def test_tooltip_of_a_run_counts_reasons_and_gives_the_confidence_range(self):
        segment = ReviewSegment("lost", (
            ReviewItem(10, ("lost", "low_confidence"), 100, 0.12),
            ReviewItem(11, ("lost",), 100, 0.4),
            ReviewItem(13, ("lost", "low_confidence", "non_circular"), 100, 0.05),
        ))
        lines = tooltip_lines(segment, IMAGES)
        assert lines[0] == "Frames 11\u201314 \u00b7 files _021\u2013_024"
        assert lines[1] == "Auto lost the bubble \u00b7 3 flagged frames"
        assert lines[2] == ("Why: Auto lost the bubble \u00d73, low confidence \u00d72, "
                            "outline not round \u00d71")
        assert lines[3] == "Confidence 0.05\u20130.40"
        assert any("\u2190/\u2192" in line for line in lines)  # how to step through it

    def test_tooltip_of_one_frame_lists_its_reasons(self):
        segment = _segment("low_confidence", 43, reasons=("low_confidence", "non_circular"),
                           confidence=0.3)
        assert tooltip_lines(segment, IMAGES) == [
            "Frame 44 \u00b7 _054", "Not sure: low confidence",
            "Why: low confidence 0.30, outline not round"]

    def test_detail_of_the_shown_frame(self):
        item = ReviewItem(12, ("lost",), 100)
        assert detail_text(item, 12, IMAGES) == \
            "Frame 13 \u00b7 _023 \u2014 Auto lost the bubble"
        assert detail_text(None, 11, IMAGES) == "Frame 12 \u00b7 _022 \u2014 not flagged"

    def test_detail_names_the_leading_reasons_only(self):
        reasons = ("lost", "rejected", "low_confidence", "non_circular", "fragmented",
                   "clipped_completed")
        segment = _segment("lost", 4, reasons=reasons, confidence=0.18)
        assert detail_text(segment.items[0], 4, IMAGES) == \
            "Frame 5 \u00b7 _015 \u2014 Auto lost the bubble, no result, low confidence 0.18 +3 more"
        assert detail_text(segment.items[0], 4, IMAGES, None).endswith("cut by the image edge")
        assert "cut by the image edge" in tooltip_lines(segment, IMAGES)[-1]
