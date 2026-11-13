"""What to do with the frame on screen (pure): Review's recommended action.

``advise`` looks at the frame's result, Auto's guess and the frames around
it, and recommends Accept, Fix or No bubble in plain words.  The first
sentence says the problem in the row's own words (``problem_sentence``),
the rest what to do.  The Review dock fills the recommended button; an
unflagged or settled frame gets none.

When in doubt the advice is Fix, never No bubble: a wrong "no bubble"
throws a real radius away.  So an empty frame (no result, no guess) is
"No bubble" only outside the bubble's frames (before its first or after
its last result) and when it looks like the reference frame
(``review_background``); and a guess is recommended only when Auto is
fairly sure of it and it fits the frames around it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from bubbletrack.controller.review_queue import BEFORE, ReviewItem
from bubbletrack.controller.review_segments import category_of
from bubbletrack.model.provenance import is_user_confirmed
from bubbletrack.model.state import AppState
from bubbletrack.model.suggestion import Suggestion, shown_suggestion
from bubbletrack.ui.review_text import problem_sentence

ACCEPT, FIX, NO_BUBBLE, NO_BUBBLE_ALL = "accept", "fix", "no_bubble", "no_bubble_all"
NONE = ""  # nothing to do (not flagged, or settled): no button is filled
NEIGHBOURHOOD = 3  # frames on each side that a guess is compared with
GUESS_TOLERANCE = 0.3  # a guess within 30 % of the radius around it fits
UNCERTAIN_GUESS = 0.3  # below this confidence a guess is not recommended
_LIKELY_WRONG = ("lost", "jump")  # a result with these is more often wrong than right
NON_CIRCULAR, CLIPPED_NOT_ROUND, UNMEASURABLE = "non_circular", "clipped_not_round", "clipped_unmeasurable"

NOT_FLAGGED = "Not flagged — nothing to do here."
ASK_EMPTY = "Is there a bubble? Fix it (F), or No bubble (N) if the frame is empty."
LIKE_REFERENCE = ("It looks like the reference frame, without a bubble: "
                  "No bubble (N), or Fix it (F) if there is one.")
BEFORE_TEXT = "Before the bubble: no result, and the frame looks like the reference frame."
NOT_ROUND_TEXT = ("R comes from the circle that fits the wall, which may be off for this shape. "
                  "If your bubbles are not round, tick \"Bubble may be non-round\" (Auto box, "
                  "Advanced) and Fit again; otherwise accept it if the circle sits on the wall.")
CLIPPED_NOT_ROUND_TEXT = ("The wall you can see is not round, so the circle may be off. Accept it "
                          "only if the circle follows the wall; otherwise Fix it (F).")
UNMEASURABLE_TEXT = ("Its area can't be measured: the wall you can see is not round. Fix it (F), "
                     "or accept Auto's guess (dashed), which takes the hidden part as round.")

LooksEmpty = Callable[[int], "bool | None"]  # frame -> looks like the reference? (None: unknown)


@dataclass(frozen=True)
class Advice:
    action: str  # ACCEPT | FIX | NO_BUBBLE | NO_BUBBLE_ALL | NONE: the recommended one
    text: str  # the problem, then what to do (one or two short sentences)
    guess: bool = False  # the frame shows Auto's guess: Accept stores it
    empty: bool = False  # no result and no guess: nothing to accept
    settled: bool = False  # accepted, fixed or marked already: nothing to accept

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
        return Advice(FIX, f"{problem_sentence('no_result')} {ASK_EMPTY}", empty=True)
    locked = state.fit_locked
    if locked is not None and locked[frame]:
        return _settled(state, frame)
    category = category_of(item) if item is not None else "no_result"
    if category == BEFORE:
        return Advice(NO_BUBBLE_ALL, BEFORE_TEXT, empty=True)
    confidence = item.confidence if item is not None else float("nan")
    problem = problem_sentence(category, confidence)
    guess = shown_suggestion(state, frame)
    if item is not None and UNMEASURABLE in item.reasons and not radius[frame] > 0:
        return Advice(FIX, f"{problem} {UNMEASURABLE_TEXT}", guess=guess is not None,
                      empty=guess is None)
    if guess is not None:
        return _advise_guess(state, frame, guess, problem)
    if not radius[frame] > 0:
        return _advise_empty(state, frame, problem, looks_empty)
    if item is None:
        return Advice(NONE, NOT_FLAGGED)
    if any(reason in item.reasons for reason in _LIKELY_WRONG):
        return Advice(FIX, f"{problem} Fix it if the outline is off the bubble wall.")
    # The circle may be off the wall here: the text says when to accept, no button is pushed.
    if category == "clipped" and CLIPPED_NOT_ROUND in item.reasons:
        return Advice(NONE, f"{problem} {CLIPPED_NOT_ROUND_TEXT}")
    if category == "shape" and NON_CIRCULAR in item.reasons:
        return Advice(NONE, f"{problem} {NOT_ROUND_TEXT}")
    return Advice(ACCEPT, f"{problem} Accept it if the outline sits on the bubble wall.")


def _settled(state: AppState, frame: int) -> Advice:
    """A locked frame: the user decided already; Fix stays possible."""
    if not state.radius[frame] > 0:
        return Advice(NONE, "Marked as no bubble (locked): nothing to do here.", empty=True)
    source = state.fit_source[frame] if state.fit_source is not None else 0
    who = "Fixed by you" if is_user_confirmed(source) else "Accepted"
    return Advice(NONE, f"{who} (locked): nothing to do here.", settled=True)


def _advise_guess(state: AppState, frame: int, guess: Suggestion, problem: str) -> Advice:
    sizes = _sizes_around(state, frame)
    if len(sizes) >= 2 and abs(guess.radius - float(np.median(sizes))) > \
            GUESS_TOLERANCE * float(np.median(sizes)):
        return Advice(FIX, f"{problem} Auto's guess (dashed) does not match the frames "
                           "around it. Fix it, or No bubble (N) if there is none.", guess=True)
    if np.isfinite(guess.confidence) and guess.confidence < UNCERTAIN_GUESS:
        return Advice(FIX, f"{problem} Auto's guess (dashed) is uncertain "
                           f"({guess.confidence:.2f}). Accept it only if it sits on the wall.",
                      guess=True)
    return Advice(ACCEPT, f"{problem} Auto's guess is dashed: accept it if it sits on the wall.",
                  guess=True)


def _advise_empty(state: AppState, frame: int, problem: str,
                  looks_empty: LooksEmpty | None) -> Advice:
    fitted = np.flatnonzero(np.asarray(state.radius, dtype=float) > 0)
    outside = len(fitted) == 0 or frame < fitted[0] or frame > fitted[-1]
    if outside and looks_empty is not None and looks_empty(frame) is True:
        return Advice(NO_BUBBLE, f"{problem} {LIKE_REFERENCE}", empty=True)
    return Advice(FIX, f"{problem} {ASK_EMPTY}", empty=True)


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
