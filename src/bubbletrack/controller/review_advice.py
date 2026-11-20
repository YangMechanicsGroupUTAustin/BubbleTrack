"""What to do with the frame on screen (pure): Review's recommended action.

``advise`` looks at the frame's result, Auto's guess and the frames around
it, and recommends Accept, Fix or No bubble in plain words.  The first
sentence says the frame's kind of problem (``problem_sentence``), the second
what the recommended (filled) button does, so text and button agree:

* Accept: the frame has a result, or a guess that fits the frames around it
  and that Auto is fairly sure of;
* Fix: no result and no usable guess, or a result or guess that clashes with
  the frames around it (a jump, a lost track, a guess of another size);
* No bubble: the frames before the bubble (the whole row), and an empty
  frame outside the bubble's frames that looks like the reference frame
  (``review_background``).

When in doubt the advice is Fix, never No bubble: a wrong "no bubble"
throws a real radius away.  Engine numbers stay out of the text: ``tip``
holds them (and engine options) for the tooltip.  An unflagged or settled
frame gets no recommendation.  ``accept_all_frames`` is what Accept all
stores: only frames whose recommendation is Accept.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np

from bubbletrack.controller.review_queue import BEFORE, ReviewItem
from bubbletrack.controller.review_segments import NOT_FOUND, category_of, kind_of
from bubbletrack.model.provenance import is_user_confirmed
from bubbletrack.model.state import AppState
from bubbletrack.model.suggestion import Suggestion, shown_suggestion
from bubbletrack.ui.review_text import LOST_SENTENCE, problem_sentence

ACCEPT, FIX, NO_BUBBLE, NO_BUBBLE_ALL = "accept", "fix", "no_bubble", "no_bubble_all"
NONE = ""  # nothing to do (not flagged, or settled): no button is filled
NEIGHBOURHOOD = 3  # frames on each side that a guess is compared with
GUESS_TOLERANCE = 0.3  # a guess within 30 % of the radius around it fits
UNCERTAIN_GUESS = 0.3  # below this confidence a guess is not recommended
# A result with these clashes with the frames around it: a jump, or a lost
# track (the tracker found nothing consistent with the frames around).
_CLASHES = ("lost", "jump")
NON_CIRCULAR, CLIPPED_NOT_ROUND, UNMEASURABLE = "non_circular", "clipped_not_round", "clipped_unmeasurable"

# The second sentence: what the recommended button does.
ACCEPT_RESULT = "Accept it if the outline sits on the bubble wall."
ACCEPT_GUESS = ("Its guess (dashed outline) fits the frames around it: accept it if it sits "
                "on the wall.")
FIX_CLASH = "It does not match the frames around it: fix it."
FIX_GUESS_CLASH = ("Its guess (dashed outline) does not match the frames around it: fix it, "
                   "or mark No bubble if there is none.")
FIX_GUESS_UNSURE = "Its guess (dashed outline) is too unsure to use: fix it."
FIX_EMPTY = "Fix it, or mark No bubble if the frame is empty."
NO_BUBBLE_EMPTY = "The frame looks empty: mark No bubble."
MARK_BEFORE = "Mark No bubble."
CLIPPED_NOT_ROUND_TEXT = ("The image edge cuts it and the wall you can see is not round, so "
                          "the circle may be off: accept it if it follows the wall.")
NOT_ROUND_TEXT = ("The outline is not round, so the circle may be off: accept it if it sits "
                  "on the wall.")
NOT_FLAGGED = "Not flagged: nothing to do here."
# "Bubble may be non-round" and the image edge cuts the bubble: no result.
UNMEASURABLE_SENTENCE = "The image edge cuts the bubble here, so its area can't be measured."
# Tooltip only: engine details behind the card's words.
NOT_ROUND_TIP = ("R is the circle through the outline. If your bubbles are not round, tick "
                 "\"Bubble may be non-round\" (Auto box, Advanced) and Fit again.")
UNMEASURABLE_TIP = ("The wall you can see is not round; Auto's guess takes the hidden part "
                    "as round.")

LooksEmpty = Callable[[int], "bool | None"]  # frame -> looks like the reference? (None: unknown)


@dataclass(frozen=True)
class Advice:
    action: str  # ACCEPT | FIX | NO_BUBBLE | NO_BUBBLE_ALL | NONE: the recommended one
    text: str  # the problem, then what the recommended button does
    guess: bool = False  # the frame shows Auto's guess: Accept stores it
    empty: bool = False  # no result and no guess: nothing to accept
    settled: bool = False  # accepted, fixed or marked already: nothing to accept
    tip: str = ""  # engine details for the tooltip (numbers, options)

    @property
    def can_accept(self) -> bool:
        return not (self.empty or self.settled)


def advise(state: AppState, frame: int, item: ReviewItem | None,
           looks_empty: LooksEmpty | None = None) -> Advice:
    """The recommended action for *frame* (listed as *item*, None if not flagged).

    *looks_empty(frame)* tells whether an empty frame looks like the
    reference frame (None or a None answer: cannot tell, so Fix).
    """
    radius = state.radius
    if radius is None or not 0 <= frame < len(radius):
        return Advice(FIX, f"{problem_sentence(NOT_FOUND)} {FIX_EMPTY}", empty=True)
    locked = state.fit_locked
    if locked is not None and locked[frame]:
        return _settled(state, frame)
    kind = kind_of(item) if item is not None else NOT_FOUND
    if kind == BEFORE:
        return Advice(NO_BUBBLE_ALL, f"{problem_sentence(BEFORE)} {MARK_BEFORE}", empty=True)
    problem = problem_sentence(kind)
    unmeasurable = item is not None and UNMEASURABLE in item.reasons and not radius[frame] > 0
    tip = ""
    if unmeasurable:
        problem, tip = UNMEASURABLE_SENTENCE, UNMEASURABLE_TIP
    guess = shown_suggestion(state, frame)
    if guess is not None:
        return _advise_guess(state, frame, guess, problem, tip)
    if not radius[frame] > 0:
        if unmeasurable:  # a bubble cut by the image edge: never "no bubble"
            return Advice(FIX, f"{problem} Fix it.", empty=True, tip=tip)
        return _advise_empty(state, frame, problem, looks_empty)
    if item is None:
        return Advice(NONE, NOT_FLAGGED)
    return _advise_result(item, LOST_SENTENCE if kind == NOT_FOUND else problem)


def accept_all_frames(state: AppState, items: Iterable[ReviewItem],
                      looks_empty: LooksEmpty | None = None) -> list[int]:
    """The frames of a row that Accept all stores, in frame order: those with
    a result or a guess whose recommendation is Accept (or none: decided
    already).  Frames Review recommends fixing stay listed."""
    frames = []
    for item in sorted(items, key=lambda i: i.frame):
        advice = advise(state, item.frame, item, looks_empty)
        if advice.can_accept and advice.action in (ACCEPT, NONE):
            frames.append(item.frame)
    return frames


def _settled(state: AppState, frame: int) -> Advice:
    """A locked frame: the user decided already; Fix stays possible."""
    if not state.radius[frame] > 0:
        return Advice(NONE, "Marked as no bubble: nothing to do here.", empty=True)
    source = state.fit_source[frame] if state.fit_source is not None else 0
    who = "Fixed by you" if is_user_confirmed(source) else "Accepted"
    return Advice(NONE, f"{who}: nothing to do here.", settled=True)


def _advise_result(item: ReviewItem, problem: str) -> Advice:
    """A flagged frame with a result: Accept, unless it clashes with the frames around."""
    if any(reason in item.reasons for reason in _CLASHES):
        return Advice(FIX, f"{problem} {FIX_CLASH}")
    category = category_of(item)
    if category == "clipped" and CLIPPED_NOT_ROUND in item.reasons:
        return Advice(ACCEPT, f"{problem} {CLIPPED_NOT_ROUND_TEXT}")
    if category == "shape" and NON_CIRCULAR in item.reasons:
        return Advice(ACCEPT, f"{problem} {NOT_ROUND_TEXT}", tip=NOT_ROUND_TIP)
    return Advice(ACCEPT, f"{problem} {ACCEPT_RESULT}")


def _advise_guess(state: AppState, frame: int, guess: Suggestion, problem: str,
                  tip: str = "") -> Advice:
    """Accept a guess that fits the frames around and that Auto is fairly sure of."""
    if np.isfinite(guess.confidence):
        tip = "\n".join(t for t in (tip, f"Auto's guess: confidence {guess.confidence:.2f}") if t)
    sizes = _sizes_around(state, frame)
    if len(sizes) >= 2 and abs(guess.radius - float(np.median(sizes))) > \
            GUESS_TOLERANCE * float(np.median(sizes)):
        return Advice(FIX, f"{problem} {FIX_GUESS_CLASH}", guess=True, tip=tip)
    if np.isfinite(guess.confidence) and guess.confidence < UNCERTAIN_GUESS:
        return Advice(FIX, f"{problem} {FIX_GUESS_UNSURE}", guess=True, tip=tip)
    return Advice(ACCEPT, f"{problem} {ACCEPT_GUESS}", guess=True, tip=tip)


def _advise_empty(state: AppState, frame: int, problem: str,
                  looks_empty: LooksEmpty | None) -> Advice:
    fitted = np.flatnonzero(np.asarray(state.radius, dtype=float) > 0)
    outside = len(fitted) == 0 or frame < fitted[0] or frame > fitted[-1]
    if outside and looks_empty is not None and looks_empty(frame) is True:
        return Advice(NO_BUBBLE, f"{problem} {NO_BUBBLE_EMPTY}", empty=True)
    return Advice(FIX, f"{problem} {FIX_EMPTY}", empty=True)


def _around(n: int, frame: int, k: int = NEIGHBOURHOOD) -> list[int]:
    return [i for i in range(max(frame - k, 0), min(frame + k, n - 1) + 1) if i != frame]


def _sizes_around(state: AppState, frame: int) -> list[float]:
    """The radii around *frame*: results, else the guesses shown."""
    sizes = []
    for i in _around(len(state.radius), frame):
        if state.radius[i] > 0:
            sizes.append(float(state.radius[i]))
        elif (guess := shown_suggestion(state, i)) is not None:
            sizes.append(float(guess.radius))
    return sizes
