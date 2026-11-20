"""Review's recommended action for the frame on screen (pure): Accept, Fix or No bubble."""

from __future__ import annotations

import numpy as np
import pytest

from bubbletrack.controller.review_advice import (
    ACCEPT, FIX, NO_BUBBLE, NO_BUBBLE_ALL, NONE, accept_all_frames, advise,
)
from bubbletrack.controller.review_queue import BEFORE, GUESS, ReviewItem
from bubbletrack.model.provenance import FitSource
from bubbletrack.model.state import AppState
from bubbletrack.model.suggestion import make_suggestion, set_suggestion

ASK = "Auto found no bubble here. Fix it, or mark No bubble if the frame is empty."


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
        assert advice.text == ("Auto found no bubble here. Its guess (dashed outline) fits the "
                               "frames around it: accept it if it sits on the wall.")

    def test_an_uncertain_guess_recommends_a_fix_and_says_so(self):
        """A2: no 'Accept it only if…' next to a filled Fix; the number is in the tooltip."""
        state = _state([40, 41, -1, 43, 44])
        _guess(state, 2, 42.0, confidence=0.27)
        advice = advise(state, 2, _item(2, "rejected", GUESS))
        assert (advice.action, advice.guess, advice.can_accept) == (FIX, True, True)
        assert advice.text == ("Auto found no bubble here. Its guess (dashed outline) is too "
                               "unsure to use: fix it.")
        assert "0.27" not in advice.text and "0.27" in advice.tip

    def test_a_lost_frame_says_no_bubble_found(self):
        state = _state([40, 41, -1, 43, 44])
        _guess(state, 2, 42.0, confidence=0.6)
        advice = advise(state, 2, _item(2, "lost", "rejected", GUESS))
        assert advice.text.startswith("Auto found no bubble here. Its guess (dashed outline) fits")

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
            assert advice.text == ("Auto found no bubble here. The frame looks empty: "
                                   "mark No bubble.")

    def test_without_any_result_only_empty_looking_frames_are_no_bubble(self):
        state = _state([-1, -1, -1])
        assert advise(state, 0, _item(0, "rejected"), _empty(0)).action == NO_BUBBLE
        assert advise(state, 1, _item(1, "rejected"), _empty(0)).action == FIX

    def test_a_lost_frame_without_result_asks_for_a_fix(self):
        state = _state([40, -1, 42])
        advice = advise(state, 1, _item(1, "lost", "rejected"), _empty())
        assert (advice.action, advice.text) == (FIX, ASK)


class TestBeforeTheBubble:
    def test_recommends_no_bubble_on_the_whole_row(self):
        state = _state([-1, -1, -1, 40, 41])
        advice = advise(state, 1, _item(1, BEFORE), _empty(0, 1, 2))
        assert advice.action == NO_BUBBLE_ALL and not advice.can_accept
        assert advice.text == "Before the bubble: the frame looks empty. Mark No bubble."


class TestResults:
    @pytest.mark.parametrize("reason, said", [("lost", "Auto lost track of the bubble here."),
                                              ("jump", "The size may be off here.")])
    def test_results_that_clash_with_the_frames_around_recommend_a_fix(self, reason, said):
        state = _state([40, 41, 42, 43, 44])
        advice = advise(state, 2, _item(2, reason))
        assert advice.action == FIX
        assert advice.text == f"{said} It does not match the frames around it: fix it."
        assert advice.can_accept

    def test_tracking_changed_it_is_a_size_problem(self):
        state = _state([40, 41, 42, 43, 44])
        advice = advise(state, 2, _item(2, "disagree", "jump"))
        assert advice.action == FIX and advice.text.startswith("The size may be off here.")
        accept = advise(state, 2, _item(2, "disagree"))
        assert accept.action == ACCEPT and accept.text.startswith("The size may be off here.")

    def test_doubtful_results_recommend_accept_without_the_number(self):
        state = _state([40, 41, 42, 43, 44])
        advice = advise(state, 2, _item(2, "low_confidence", "fragmented", confidence=0.21))
        assert advice.action == ACCEPT
        assert advice.text == ("Auto is not sure here. Accept it if the outline sits on the "
                               "bubble wall.")

    def test_a_not_round_outline_recommends_accept_and_keeps_the_option_in_the_tooltip(self):
        state = _state([40, 41, 42])
        advice = advise(state, 1, _item(1, "fragmented", "non_circular"))
        assert advice.action == ACCEPT
        assert advice.text.startswith("Auto is not sure here. The outline is not round")
        assert "least-squares" not in advice.text and "Bubble may be non-round" in advice.tip

    def test_an_ml_outline_is_not_sure(self):
        state = _state([40, 41, 42])
        assert advise(state, 1, _item(1, "ml_outline")).text.startswith("Auto is not sure here.")

    def test_an_unflagged_frame_recommends_nothing(self):
        state = _state([40, 41, 42])
        advice = advise(state, 1, None)
        assert advice.action == NONE and advice.text == "Not flagged: nothing to do here."

    def test_settled_frames_recommend_nothing_and_offer_no_accept(self):
        state = _state([40, -1, 42])
        state.fit_locked[[0, 1, 2]] = True
        state.fit_source[2] = FitSource.GUIDED
        accepted, empty, fixed = (advise(state, f, None) for f in (0, 1, 2))
        assert accepted.action == empty.action == fixed.action == NONE
        assert accepted.text == "Accepted: nothing to do here." and not accepted.can_accept
        assert fixed.text == "Fixed by you: nothing to do here." and not fixed.can_accept
        assert empty.text == "Marked as no bubble: nothing to do here." and not empty.can_accept

    def test_texts_are_plain_words(self):
        state = _state([40, 41, 42])
        for reasons in (("disagree",), ("reacquired",), ("ml_outline",), ("collapse",)):
            text = advise(state, 1, _item(1, *reasons)).text
            assert not any(word in text for word in ("disagree", "reacquired", "ml_", "_"))


class TestTheButtonAgreesWithTheText:
    """A2: the filled button and the advice say the same thing."""

    CASES = (
        ([40, 41, 42, 43, 44], 2, ("low_confidence",), None),
        ([40, 41, 42, 43, 44], 2, ("jump",), None),
        ([40, 41, 42, 43, 44], 2, ("lost",), None),
        ([40, 41, 42, 43, 44], 2, ("disagree",), None),
        ([40, 41, 42, 43, 44], 2, ("non_circular",), None),
        ([40, 41, 42, 43, 44], 2, ("clipped_not_round",), None),
        ([40, 41, -1, 43, 44], 2, ("rejected", GUESS), (42.0, 0.6)),
        ([40, 41, -1, 43, 44], 2, ("rejected", GUESS), (42.0, 0.1)),
        ([40, 41, -1, 43, 44], 2, ("rejected", GUESS), (8.0, 0.6)),
        ([40, 41, -1, 43, 44], 2, ("rejected",), None),
        ([-1, 41, 42, 43, -1], 4, ("rejected",), None),
        ([-1, -1, 42, 43, 44], 0, (BEFORE,), None),
    )
    VERB = {ACCEPT: "accept it", FIX: "fix it", NO_BUBBLE: "mark no bubble",
            NO_BUBBLE_ALL: "mark no bubble"}
    OTHERS = {ACCEPT: ("fix it",), FIX: ("accept it",), NO_BUBBLE: ("accept it", "fix it"),
              NO_BUBBLE_ALL: ("accept it", "fix it")}

    @pytest.mark.parametrize("radii, frame, reasons, guess", CASES)
    def test_the_text_tells_to_do_what_the_filled_button_does(self, radii, frame, reasons, guess):
        state = _state(radii)
        if guess is not None:
            _guess(state, frame, *guess)
        advice = advise(state, frame, _item(frame, *reasons), _empty(0, 4))
        text = advice.text.lower()
        assert self.VERB[advice.action] in text, advice
        assert not any(other in text for other in self.OTHERS[advice.action]), advice
        assert not any(ch.isdigit() for ch in advice.text)  # A1: no engine numbers
        for word in ("reference frame", "least-squares", "dashed guess", "confidence"):
            assert word not in text

    def test_accept_is_recommended_for_results_and_fitting_guesses_only(self):
        state = _state([40, 41, -1, -1, 44, 45])
        _guess(state, 2, 42.0, confidence=0.6)  # fits
        _guess(state, 3, 9.0, confidence=0.6)  # a speckle
        actions = [advise(state, f, _item(f, *r)).action for f, r in (
            (0, ("low_confidence",)), (1, ("jump",)), (2, ("rejected", GUESS)),
            (3, ("rejected", GUESS)), (4, ("disagree",)))]
        assert actions == [ACCEPT, FIX, ACCEPT, FIX, ACCEPT]


class TestAcceptAllFrames:
    """B5: Accept all stores only the frames whose recommendation is Accept."""

    def test_fix_recommended_and_empty_frames_are_left_out(self):
        state = _state([40, 41, 42, -1, -1, 45, 46])
        _guess(state, 3, 43.0, confidence=0.6)  # fits: Accept
        _guess(state, 4, 43.0, confidence=0.1)  # too unsure: Fix
        state.radius[6] = -1  # no result, no guess: Fix or No bubble
        items = [_item(0, "low_confidence"), _item(1, "jump"), _item(2, "lost"),
                 _item(3, "rejected", GUESS), _item(4, "rejected", GUESS),
                 _item(5, "disagree"), _item(6, "rejected")]
        assert accept_all_frames(state, items) == [0, 3, 5]

    def test_a_row_of_frames_to_accept_is_accepted_whole(self):
        state = _state([40, 41, 42])
        items = [_item(f, "low_confidence") for f in range(3)]
        assert accept_all_frames(state, items) == [0, 1, 2]
