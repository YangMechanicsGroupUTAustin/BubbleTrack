"""ML assist's teaching steps (no Qt, no scikit-learn): which frames are examples,
when two of them are different enough to train, what Train and apply measures,
and the one-sentence next step the panel shows in every state."""

from __future__ import annotations

import numpy as np
import pytest

from bubbletrack.controller import ml_teach_steps as T
from bubbletrack.model.provenance import FitSource
from bubbletrack.model.state import AppState, update_state
from bubbletrack.model.teach import BACKGROUND, BUBBLE, MLExamples

N = 40


def _state(n: int = N) -> AppState:
    state = AppState(images=tuple(f"f{i}.tif" for i in range(n))).with_results_initialized(n)
    state.radius[:] = np.linspace(20.0, 40.0, n)
    state.circle_fit_par[:] = [[50.0, 60.0]] * n
    state.fit_source[:] = FitSource.AUTO
    state.fit_confidence[:] = 0.8
    return update_state(state, gridx=(1, 100), gridy=(1, 120))


def _confirm(state, *frames):
    for f in frames:
        state.fit_locked[f] = True                 # accepted (Looks right, or Review's Accept)


# --------------------------------------------------------------------------- #
#  Examples and the two-different-frames rule
# --------------------------------------------------------------------------- #

def test_confirmed_and_fixed_frames_are_examples_with_their_size():
    state = _state()
    _confirm(state, 3)
    state.fit_source[30] = FitSource.MANUAL
    assert [(e.frame, round(e.radius)) for e in T.example_frames(state)] == [(3, 22), (30, 35)]


def test_a_painted_frame_counts_once_it_has_bubble_and_background_strokes():
    state = _state()
    only_bubble = MLExamples().paint(5, BUBBLE, [(10, 10)])
    state = update_state(state, ml_examples=only_bubble)
    assert T.example_frames(state) == ()
    both = only_bubble.paint(5, BACKGROUND, [(40, 40)])
    state = update_state(state, ml_examples=both)
    (painted,) = T.example_frames(state)
    assert painted.frame == 5 and not painted.sized


def test_two_alike_frames_are_not_enough_two_different_ones_are():
    state = _state()
    _confirm(state, 10, 11)                       # similar size, next to each other
    progress = T.progress(state)
    assert not progress.ready and progress.done == 1
    _confirm(state, 37)                           # a quarter of the recording later
    progress = T.progress(state)
    assert progress.ready and progress.done == 2
    assert {e.frame for e in progress.pair} in ({10, 37}, {11, 37})


def test_a_locked_no_bubble_frame_is_not_an_example():
    state = _state()
    state.radius[4] = -1.0
    _confirm(state, 4)
    assert T.progress(state).done == 0


# --------------------------------------------------------------------------- #
#  The outline a card asks about
# --------------------------------------------------------------------------- #

def test_a_card_judges_the_result_else_autos_guess():
    from bubbletrack.model.suggestion import make_suggestion

    state = _state()
    state.radius[5] = -1.0
    state.fit_confidence[5] = 0.2
    state.fit_suggestion[5] = make_suggestion(30.0, 52.0, 61.0, np.ones((4, 2)), 0.2)
    state.radius[6] = -1.0                        # nothing at all
    result, guess = T.outline(state, 4), T.outline(state, 5)
    assert not result.guess and result.radius == pytest.approx(state.radius[4])
    assert guess.guess and (guess.radius, guess.rc, guess.cc) == (30.0, 52.0, 61.0)
    assert T.outline(state, 6) is None
    radius, conf, trusted = T.outlines(state)
    assert radius[5] == 30.0 and conf[5] == 0.2 and not trusted[5] and trusted[4]
    assert np.isnan(radius[6])


# --------------------------------------------------------------------------- #
#  What Train and apply measures
# --------------------------------------------------------------------------- #

def test_train_scope_is_review_frames_and_frames_without_a_result_minus_the_users():
    state = _state()
    state.radius[[2, 3, 25]] = -1.0               # no result
    state.radius[3] = -1.0
    _confirm(state, 3)                            # "no bubble here": the user's decision
    state.fit_source[8] = FitSource.MANUAL        # listed below, but the user's fix
    scope = T.train_scope(state, listed={8, 12, 13})
    assert scope == (2, 12, 13, 25)               # never 0..39's unflagged measured frames


def test_train_scope_without_results_is_empty():
    assert T.train_scope(AppState(), listed=()) == ()


# --------------------------------------------------------------------------- #
#  The panel's words, one state at a time
# --------------------------------------------------------------------------- #

def test_step_line():
    state = _state()
    assert T.step_text(T.progress(state)) == (
        "Step 1 of 2: confirm 2 frames where the outline is right — 0 of 2 done")
    _confirm(state, 10)
    assert T.step_text(T.progress(state)).endswith("— 1 of 2 done")
    _confirm(state, 37)
    assert T.step_text(T.progress(state)) == "Step 2 of 2: Train and apply — 2 different frames confirmed"
    _confirm(state, 20)
    assert T.step_text(T.progress(state)) == "Step 2 of 2: Train and apply — 3 frames confirmed"


@pytest.mark.parametrize("confirmed, candidates, scope, expected, warn", [
    ((), 2, 9, "Judge each frame: Looks right ✓ if the red outline sits on the bubble wall, "
               "Looks wrong ✗ to see another frame.", False),
    ((10,), 1, 9, "One more: judge the other frame, which differs from frame 11.", False),
    ((10, 11), 1, 9, "Your 2 confirmed frames are alike (similar size, close in time): "
                     "judge the other frame, which differs.", False),
    ((), 0, 9, "No frame with an outline is left to judge: open “Paint examples by hand” below "
               "and paint the bubble and the background on two frames.", True),
    ((10,), 0, 9, "No other frame with an outline is left: open “Paint examples by hand” below "
                  "and paint a frame that differs from frame 11 (another size or another part "
                  "of the recording).", True),
    ((10, 37), 0, 9, "Press Train and apply: ML assist learns from your frames and measures "
                     "the 9 frames that need review.", False),
    ((10, 37), 0, 0, "Nothing needs ML assist: every frame has a result and none is listed "
                     "in Review.", True),
])
def test_next_step_says_what_to_do_in_one_sentence(confirmed, candidates, scope, expected, warn):
    state = _state()
    _confirm(state, *confirmed)
    assert T.next_text(T.progress(state), candidates=candidates, scope=scope) == (expected, warn)


def test_train_button_says_why_it_waits():
    state = _state()
    _confirm(state, 10)
    assert T.train_state(T.progress(state), scope=9) == (
        False, "Confirm 2 different frames first (1 of 2 done)")
    _confirm(state, 37)
    enabled, tip = T.train_state(T.progress(state), scope=9)
    assert enabled and "9 frames that need review" in tip
    assert T.train_state(T.progress(state), scope=0)[0] is False


def test_summary_after_train_and_apply():
    assert T.summary_text(41, 52, 9, stopped=False) == (
        "ML assist measured 41 of 52 frames to check; 9 are listed in Review")
    assert T.summary_text(52, 52, 1, stopped=False).endswith("; 1 is listed in Review")
    assert T.summary_text(52, 52, 0, stopped=False).endswith("; none is left to check in Review")
    assert T.summary_text(20, 52, 30, stopped=True) == (
        "ML assist stopped: 20 of 52 frames to check measured; 30 are listed in Review")
