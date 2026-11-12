"""Review queue rules: which frames are listed, why, and in what order."""

from __future__ import annotations

import numpy as np
import pytest

from bubbletrack.controller.review_queue import (
    ReviewItem,
    build_review_items,
    jump_frames,
    next_after_action,
)
from bubbletrack.model.provenance import FitSource
from bubbletrack.model.state import AppState


def _state(radii, conf=None) -> AppState:
    """Results for len(radii) frames; conf=None leaves frames unscored (legacy)."""
    state = AppState().with_results_initialized(len(radii))
    state.radius[:] = radii
    state.fit_source[:] = [FitSource.AUTO if r > 0 else FitSource.NONE for r in radii]
    if conf is not None:
        state.fit_confidence[:] = conf
    return state


def _frames(items) -> list[int]:
    return [item.frame for item in items]


def _by_frame(items) -> dict[int, ReviewItem]:
    return {item.frame: item for item in items}


class TestRules:
    def test_no_results_means_no_items(self):
        assert build_review_items(AppState()) == []
        assert build_review_items(AppState().with_results_initialized(5)) == []

    def test_rejected_detection_is_listed_but_never_fitted_frames_are_not(self):
        state = _state([40.0, -1.0, -1.0, 40.0], conf=[0.9, 0.2, np.nan, 0.9])
        items = _by_frame(build_review_items(state))
        assert set(items) == {1}
        assert items[1].reasons == ("rejected", "low_confidence")

    def test_low_confidence_threshold(self):
        state = _state([40.0, 41.0, 42.0], conf=[0.49, 0.5, 0.9])
        items = _by_frame(build_review_items(state))
        assert set(items) == {0}
        assert items[0].reasons == ("low_confidence",)
        assert items[0].confidence == pytest.approx(0.49)

    def test_review_flags_are_listed_and_other_flags_ignored(self):
        state = _state([40.0, 41.0, 42.0, 43.0], conf=[0.9] * 4)
        state.fit_flags[1] = frozenset({"non_circular", "no_reference"})
        state.fit_flags[2] = frozenset({"no_reference", "low_snr"})
        state.fit_flags[3] = frozenset({"fragmented", "shoulder", "non_circular"})
        items = _by_frame(build_review_items(state))
        assert set(items) == {1, 3}
        assert items[1].reasons == ("non_circular",)
        assert items[3].reasons == ("non_circular", "fragmented", "shoulder")  # most severe first
        assert items[3].severity > items[1].severity  # context adds to the rank

    def test_context_reasons_alone_do_not_list_a_frame(self):
        state = _state([40.0, 41.0, 42.0], conf=[0.9] * 3)
        state.fit_flags[0] = frozenset({"fragmented"})
        state.fit_flags[1] = frozenset({"clipped_completed", "shoulder"})
        state.fit_flags[2] = frozenset({"fragmented"})
        extra = [ReviewItem(2, ("collapse",), 0)]
        assert _frames(build_review_items(state, extra)) == [2]

    def test_sequence_flags_are_accepted(self):
        state = _state([40.0] * 4, conf=[0.9] * 4)
        for i, flag in enumerate(("disagree", "lost", "reacquired", "cloud")):
            state.fit_flags[i] = frozenset({flag})
        assert _frames(build_review_items(state)) == [1, 0, 2, 3]

    def test_locked_frames_are_excluded(self):
        state = _state([40.0, -1.0, 40.0], conf=[0.1, 0.1, 0.9])
        state.fit_locked[0] = True
        state.fit_locked[1] = True
        assert build_review_items(state) == []

    def test_manual_fix_hides_the_stale_engine_verdict(self):
        state = _state([40.0, 41.0], conf=[0.1, 0.1])
        state.fit_flags[0] = frozenset({"non_circular"})
        state.fit_source[0] = FitSource.MANUAL
        assert _frames(build_review_items(state)) == [1]


class TestJump:
    def test_spike_against_both_neighbours_is_a_jump(self):
        state = _state([50.0, 50.0, 80.0, 50.0, 50.0])
        items = build_review_items(state)
        assert [(i.frame, i.reasons) for i in items] == [(2, ("jump",))]

    def test_dip_is_a_jump_too(self):
        assert jump_frames(np.array([50.0, 50.0, 30.0, 50.0])).tolist() == [2]

    def test_steady_fast_change_is_not_a_jump(self):
        # Collapse: every step > 15 %, but each frame lies between its neighbours.
        assert jump_frames(np.array([100.0, 80.0, 60.0, 45.0, 30.0])).tolist() == []

    def test_small_or_one_sided_changes_are_not_jumps(self):
        assert jump_frames(np.array([50.0, 56.0, 50.0])).tolist() == []  # 12 %
        assert jump_frames(np.array([50.0, 50.0, 80.0, 80.0])).tolist() == []  # a step
        assert jump_frames(np.array([80.0, 50.0, 50.0])).tolist() == []  # ends: one neighbour

    def test_neighbours_skip_frames_without_a_result(self):
        assert jump_frames(np.array([50.0, -1.0, 80.0, np.nan, 50.0])).tolist() == [2]

    def test_jump_rule_applies_only_to_unscored_frames(self):
        state = _state([50.0, 50.0, 80.0, 50.0, 50.0], conf=[0.9] * 5)
        assert build_review_items(state) == []


class TestOrderingAndMerge:
    def test_most_likely_wrong_first_then_by_frame(self):
        state = _state([40.0, 40.0, -1.0, 40.0, 40.0, 40.0],
                       conf=[0.9, 0.45, 0.1, 0.05, 0.9, 0.9])
        state.fit_flags[0] = frozenset({"non_circular"})
        state.fit_flags[4] = frozenset({"non_circular"})
        items = build_review_items(state)
        assert _frames(items) == [2, 3, 1, 0, 4]
        severities = [i.severity for i in items]
        assert severities == sorted(severities, reverse=True)

    def test_more_reasons_rank_higher_at_equal_worst_reason(self):
        state = _state([40.0, 40.0, 40.0], conf=[0.9] * 3)
        state.fit_flags[1] = frozenset({"non_circular"})
        state.fit_flags[2] = frozenset({"non_circular", "shoulder"})
        assert _frames(build_review_items(state)) == [2, 1]

    def test_extra_items_merge_reasons_per_frame(self):
        state = _state([40.0, 40.0, 40.0, 40.0], conf=[0.3, 0.9, 0.9, 0.9])
        state.fit_locked[3] = True
        extra = [ReviewItem(0, ("disagree",), 85), ReviewItem(2, ("collapse",), 35),
                 ReviewItem(2, ("collapse", "cloud"), 10), ReviewItem(3, ("lost",), 100),
                 ReviewItem(9, ("lost",), 100)]
        items = _by_frame(build_review_items(state, extra))
        assert set(items) == {0, 2}  # locked and out-of-range frames dropped
        assert items[0].reasons == ("disagree", "low_confidence")
        assert items[2].reasons == ("collapse", "cloud")

    def test_extra_severity_is_kept_when_higher(self):
        state = _state([40.0, 40.0], conf=[0.9, 0.9])
        items = build_review_items(state, [ReviewItem(1, ("custom",), 500)])
        assert items == [ReviewItem(1, ("custom",), 500, 0.9)]


class TestNextAfterAction:
    def test_moves_to_the_following_item(self):
        assert next_after_action([5, 2, 9], 2, [5, 9]) == 9

    def test_skips_items_that_disappeared(self):
        assert next_after_action([5, 2, 9, 4], 2, [5, 4]) == 4

    def test_wraps_to_earlier_items(self):
        assert next_after_action([5, 2, 9], 9, [5, 2]) == 5

    def test_stays_when_the_frame_is_the_only_item_left(self):
        assert next_after_action([5, 2], 2, [2]) == 2

    def test_empty_queue(self):
        assert next_after_action([2], 2, []) is None

    def test_new_items_when_nothing_old_is_left(self):
        assert next_after_action([2], 2, [2, 3]) == 3


class TestEquality:
    def test_unscored_items_compare_equal(self):
        # NaN confidences must not make an unchanged queue look changed
        # (the panel skips its rebuild when the queue is the same).
        state = _state([50.0, 50.0, 80.0, 50.0, 50.0])
        assert build_review_items(state) == build_review_items(state)
        nan_a, nan_b = float("nan"), float("nan")
        assert ReviewItem(1, ("jump",), 80, nan_a) == ReviewItem(1, ("jump",), 80, nan_b)
        assert hash(ReviewItem(1, ("jump",), 80, nan_a)) == hash(ReviewItem(1, ("jump",), 80))

    def test_different_confidences_differ(self):
        assert ReviewItem(1, ("jump",), 80, 0.2) != ReviewItem(1, ("jump",), 80, 0.3)
        assert ReviewItem(1, ("jump",), 80, 0.2) != ReviewItem(1, ("jump",), 80)
