"""Review's recommended action for the frame on screen (pure): Accept, Fix or No bubble."""

from __future__ import annotations

import numpy as np
import pytest

from bubbletrack.controller.review_advice import (
    ACCEPT, FIX, NO_BUBBLE, NO_BUBBLE_ALL, NONE, advise,
)
from bubbletrack.controller.review_queue import BEFORE, GUESS, ReviewItem
from bubbletrack.model.provenance import FitSource
from bubbletrack.model.state import AppState
from bubbletrack.model.suggestion import make_suggestion, set_suggestion

ASK = "No result here. Is there a bubble? Fix it (F), or No bubble (N) if the frame is empty."


def _state(radii) -> AppState:
    state = AppState().with_results_initialized(len(radii))
    state.radius[:] = radii
    state.fit_source[:] = [FitSource.AUTO if r > 0 else FitSource.NONE for r in radii]
    state.fit_confidence[:] = [0.9 if r > 0 else 0.1 for r in radii]
    return state


def _guess(state, frame: int, radius: float, confidence: float = 0.5) -> None:
    set_suggestion(state, frame, make_suggestion(radius, 50.0, 60.0, np.zeros((4, 2)),
                                                 confidence))


def _item(frame: int, *reasons: str, confidence: float = float("nan")) -> ReviewItem:
    return ReviewItem(frame, reasons, 80, confidence)


def _empty(*frames: int):
    """looks_empty: these frames look like the reference frame."""
    return lambda frame: frame in frames


class TestGuesses:
    def test_a_sure_guess_the_size_of_the_bubble_around_is_worth_accepting(self):
        state = _state([40, 41, -1, 43, 44])
        _guess(state, 2, 42.0)
        advice = advise(state, 2, _item(2, "rejected", GUESS))
        assert (advice.action, advice.guess) == (ACCEPT, True)
        assert advice.text == ("No result here. Auto's guess is dashed: "
                               "accept it if it sits on the wall.")

    def test_an_uncertain_guess_recommends_a_fix(self):
        state = _state([40, 41, -1, 43, 44])
        _guess(state, 2, 42.0, confidence=0.27)
        advice = advise(state, 2, _item(2, "rejected", GUESS))
        assert (advice.action, advice.guess, advice.can_accept) == (FIX, True, True)
        assert advice.text == ("No result here. Auto's guess (dashed) is uncertain (0.27). "
                               "Accept it only if it sits on the wall.")

    def test_the_first_sentence_is_the_rows_problem(self):
        state = _state([40, 41, -1, 43, 44])
        _guess(state, 2, 42.0, confidence=0.6)
        advice = advise(state, 2, _item(2, "lost", "rejected", GUESS))
        assert advice.text.startswith("Auto lost the bubble here. Auto's guess is dashed")

    def test_a_guess_among_guesses_of_the_same_size(self):
        state = _state([-1] * 7)
        for f, r in enumerate([45, 47, 46, 44, 48, 46, 45]):
            _guess(state, f, r)
        assert advise(state, 3, _item(3, "rejected", GUESS)).action == ACCEPT

    def test_a_guess_unlike_the_frames_around_is_not_recommended(self):
        state = _state([40, 41, -1, 43, 44])
        _guess(state, 2, 8.0)  # a speckle, not the bubble
        advice = advise(state, 2, _item(2, "rejected", GUESS))
        assert advice.action == FIX and advice.guess
        assert "does not match" in advice.text and "No bubble (N)" in advice.text

    def test_without_neighbours_to_compare_a_sure_guess_is_trusted(self):
        state = _state([-1, -1])
        _guess(state, 0, 30.0)
        assert advise(state, 0, _item(0, "rejected", GUESS)).action == ACCEPT


class TestNoResult:
    """C1: with no result and no guess, Fix; No bubble only outside the bubble's
    frames, and only when the frame looks like the reference frame."""

    def test_inside_the_bubbles_frames_always_fix(self):
        state = _state([-1, 40, 41, -1, 43, -1])
        advice = advise(state, 3, _item(3, "rejected"), _empty(3))
        assert (advice.action, advice.text) == (FIX, ASK)
        assert not advice.can_accept  # nothing to accept

    def test_after_the_last_result_a_soft_bubble_still_means_fix(self):
        """05 blood: ~200 frames show a soft bubble Auto never measured."""
        state = _state([40, 41, -1, -1, -1, -1, -1, -1])
        assert advise(state, 6, _item(6, "rejected"), _empty()).action == FIX
        assert advise(state, 6, _item(6, "rejected")).action == FIX  # cannot tell: Fix

    def test_outside_the_bubbles_frames_an_empty_looking_frame_means_no_bubble(self):
        state = _state([-1, -1, 40, 41, -1, -1])
        for frame in (0, 5):
            advice = advise(state, frame, _item(frame, "rejected"), _empty(0, 5))
            assert advice.action == NO_BUBBLE and not advice.can_accept
            assert advice.text.startswith("No result here.")
            assert "looks like the reference frame" in advice.text

    def test_without_any_result_only_empty_looking_frames_are_no_bubble(self):
        state = _state([-1, -1, -1])
        assert advise(state, 0, _item(0, "rejected"), _empty(0)).action == NO_BUBBLE
        assert advise(state, 1, _item(1, "rejected"), _empty(0)).action == FIX

    def test_a_lost_frame_without_result_names_the_row_problem(self):
        state = _state([40, -1, 42])
        advice = advise(state, 1, _item(1, "lost", "rejected"), _empty())
        assert advice.text.startswith("Auto lost the bubble here. Is there a bubble?")


class TestBeforeTheBubble:
    def test_recommends_no_bubble_on_the_whole_row(self):
        state = _state([-1, -1, -1, 40, 41])
        advice = advise(state, 1, _item(1, BEFORE), _empty(0, 1, 2))
        assert advice.action == NO_BUBBLE_ALL and not advice.can_accept
        assert advice.text.startswith("Before the bubble")


class TestResults:
    @pytest.mark.parametrize("reason, said", [("lost", "Auto lost the bubble here."),
                                              ("jump", "Radius jumps here.")])
    def test_likely_wrong_results_recommend_a_fix(self, reason, said):
        state = _state([40, 41, 42, 43, 44])
        advice = advise(state, 2, _item(2, reason))
        assert advice.action == FIX and advice.text.startswith(said)
        assert "Fix it if the outline is off" in advice.text
        assert advice.can_accept

    def test_the_card_says_the_rows_problem_not_another_reason(self):
        """M2: a 'Tracking changed it' row is not described as 'Radius jumps'."""
        state = _state([40, 41, 42, 43, 44])
        advice = advise(state, 2, _item(2, "disagree", "jump"))
        assert advice.action == FIX and advice.text.startswith("Tracking changed it here.")
        assert "jump" not in advice.text.lower()

    def test_doubtful_results_recommend_a_look_then_accept(self):
        state = _state([40, 41, 42, 43, 44])
        advice = advise(state, 2, _item(2, "low_confidence", "fragmented", confidence=0.21))
        assert advice.action == ACCEPT
        assert advice.text.startswith("Not sure here: low confidence (0.21).")

    def test_routine_flags_never_lead_the_sentence(self):
        state = _state([40, 41, 42])
        advice = advise(state, 1, _item(1, "fragmented", "non_circular"))
        assert advice.text.startswith("Outline not round here.")
        assert "R comes from the least-squares circle through the bubble outline" in advice.text

    def test_an_ml_outline_names_ml_assist(self):
        state = _state([40, 41, 42])
        advice = advise(state, 1, _item(1, "ml_outline", "lost"))
        assert advice.text.startswith("Outline from ML assist here.")

    def test_an_unflagged_frame_recommends_nothing(self):
        state = _state([40, 41, 42])
        advice = advise(state, 1, None)
        assert advice.action == NONE and advice.text == "Not flagged — nothing to do here."

    def test_settled_frames_recommend_nothing_and_offer_no_accept(self):
        state = _state([40, -1, 42])
        state.fit_locked[[0, 1, 2]] = True
        state.fit_source[2] = FitSource.GUIDED
        accepted, empty, fixed = (advise(state, f, None) for f in (0, 1, 2))
        assert accepted.action == empty.action == fixed.action == NONE
        assert accepted.text.startswith("Accepted (locked)") and not accepted.can_accept
        assert fixed.text.startswith("Fixed by you (locked)") and not fixed.can_accept
        assert empty.text.startswith("Marked as no bubble (locked)") and not empty.can_accept

    def test_texts_are_plain_words(self):
        state = _state([40, 41, 42])
        for reasons in (("disagree",), ("reacquired",), ("ml_outline",), ("collapse",)):
            text = advise(state, 1, _item(1, *reasons)).text
            assert not any(word in text for word in ("disagree", "reacquired", "ml_", "_"))
