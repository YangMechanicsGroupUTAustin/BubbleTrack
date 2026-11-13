"""Review's recommended action for the frame on screen (pure): Accept, Fix or No bubble."""

from __future__ import annotations

import numpy as np
import pytest

from bubbletrack.controller.review_advice import ACCEPT, FIX, NO_BUBBLE, NONE, advise
from bubbletrack.controller.review_queue import GUESS, ReviewItem
from bubbletrack.model.provenance import FitSource
from bubbletrack.model.state import AppState
from bubbletrack.model.suggestion import make_suggestion, set_suggestion


def _state(radii) -> AppState:
    state = AppState().with_results_initialized(len(radii))
    state.radius[:] = radii
    state.fit_source[:] = [FitSource.AUTO if r > 0 else FitSource.NONE for r in radii]
    state.fit_confidence[:] = [0.9 if r > 0 else 0.1 for r in radii]
    return state


def _guess(state, frame: int, radius: float) -> None:
    set_suggestion(state, frame, make_suggestion(radius, 50.0, 60.0, np.zeros((4, 2)), 0.05))


def _item(frame: int, *reasons: str, confidence: float = float("nan")) -> ReviewItem:
    return ReviewItem(frame, reasons, 80, confidence)


class TestGuesses:
    def test_a_guess_the_size_of_the_bubble_around_is_worth_accepting(self):
        state = _state([40, 41, -1, 43, 44])
        _guess(state, 2, 42.0)
        advice = advise(state, 2, _item(2, "rejected", GUESS))
        assert (advice.action, advice.guess) == (ACCEPT, True)
        assert "Auto has a guess" in advice.text and "Accept it if" in advice.text

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
        assert "does not match" in advice.text and "No bubble" in advice.text

    def test_a_stray_guess_is_never_called_no_bubble(self):
        """In doubt the advice is Fix: a wrong "no bubble" throws a radius away."""
        state = _state([-1] * 6)
        _guess(state, 0, 8.8)  # before the bubble: Auto guessed at speckle
        _guess(state, 2, 5.7)
        _guess(state, 3, 40.2)
        assert advise(state, 0, _item(0, "rejected", GUESS)).action == FIX

    def test_without_neighbours_to_compare_the_guess_is_trusted(self):
        state = _state([-1, -1])
        _guess(state, 0, 30.0)
        assert advise(state, 0, _item(0, "rejected", GUESS)).action == ACCEPT


class TestNoResult:
    def test_empty_frames_around_mean_no_bubble(self):
        state = _state([-1, -1, -1, -1, -1, -1, -1, 40, 41])
        advice = advise(state, 2, _item(2, "rejected"))
        assert advice.action == NO_BUBBLE and "probably no bubble" in advice.text
        assert not advice.can_accept  # nothing to accept

    def test_a_gap_inside_the_bubble_needs_a_fix(self):
        state = _state([40, 41, -1, 43, 44])
        advice = advise(state, 2, _item(2, "rejected"))
        assert advice.action == FIX and "frames around it have one" in advice.text
        assert not advice.can_accept

    def test_only_guesses_around_an_empty_frame_say_so(self):
        """Not "the frames around it show a bubble": they only have Auto's guesses."""
        state = _state([-1] * 7)
        for f in (0, 1, 2, 4, 5, 6):
            _guess(state, f, 45.0)
        advice = advise(state, 3, _item(3, "rejected"))
        assert advice.action == FIX and "only has guesses" in advice.text


class TestResults:
    @pytest.mark.parametrize("reason, said", [("lost", "Auto lost the bubble"),
                                              ("jump", "Radius jumps")])
    def test_likely_wrong_results_recommend_a_fix(self, reason, said):
        state = _state([40, 41, 42, 43, 44])
        advice = advise(state, 2, _item(2, reason))
        assert advice.action == FIX and "Fix it if the outline is off" in advice.text
        assert advice.can_accept

    def test_the_fix_advice_names_the_reason_that_makes_it_likely_wrong(self):
        """Not "Tracking changed it. Fix it ...": the jump is why Fix is advised."""
        state = _state([40, 41, 42, 43, 44])
        advice = advise(state, 2, _item(2, "disagree", "jump"))
        assert advice.action == FIX and advice.text.startswith("Radius jumps.")

    def test_doubtful_results_recommend_a_look_then_accept(self):
        state = _state([40, 41, 42, 43, 44])
        advice = advise(state, 2, _item(2, "low_confidence", "fragmented", confidence=0.21))
        assert advice.action == ACCEPT
        assert advice.text.startswith("Low confidence (0.21).")

    def test_routine_flags_never_lead_the_sentence(self):
        state = _state([40, 41, 42])
        advice = advise(state, 1, _item(1, "fragmented", "non_circular"))
        assert advice.text.startswith("Outline not round.")

    def test_an_unflagged_frame(self):
        state = _state([40, 41, 42])
        advice = advise(state, 1, None)
        assert advice.action == ACCEPT and advice.text.startswith("Not flagged.")

    def test_settled_frames_recommend_nothing(self):
        state = _state([40, -1, 42])
        state.fit_locked[[0, 1]] = True
        accepted, empty = advise(state, 0, None), advise(state, 1, None)
        assert accepted.action == empty.action == NONE
        assert accepted.text.startswith("Accepted (locked)") and accepted.can_accept
        assert empty.text == "Marked as no bubble (locked)." and not empty.can_accept

    def test_texts_are_plain_words(self):
        state = _state([40, 41, 42])
        for reasons in (("disagree",), ("reacquired",), ("ml_outline",), ("collapse",)):
            text = advise(state, 1, _item(1, *reasons)).text
            assert not any(word in text for word in ("disagree", "reacquired", "ml_", "_"))
