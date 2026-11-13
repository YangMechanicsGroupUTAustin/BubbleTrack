"""What to do with the frame on screen (pure): Review's recommended action.

``advise`` looks at the frame's result, Auto's guess and the frames around
it, and recommends Accept, Fix or No bubble with one short plain sentence:
what is wrong, and what to do.  The Review dock makes that action its
highlighted button; the other two stay one click (or key) away.

The frames around (``NEIGHBOURHOOD`` each side) decide the empty cases: a
guess about the size of the bubble around it is worth accepting; with
nothing at all around (no result, no guess) the frame is probably before or
after the bubble.  When in doubt the advice is Fix, never No bubble: a
wrong "no bubble" would throw a real radius away.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bubbletrack.controller.review_queue import CONTEXT_REASONS, GUESS, ReviewItem
from bubbletrack.model.state import AppState
from bubbletrack.model.suggestion import shown_suggestion
from bubbletrack.ui.review_text import reason_text

ACCEPT, FIX, NO_BUBBLE = "accept", "fix", "no_bubble"
NEIGHBOURHOOD = 3  # frames on each side that the advice compares with
GUESS_TOLERANCE = 0.3  # a guess within 30 % of the radius around it fits
_LIKELY_WRONG = ("lost", "jump")  # a result with these is more often wrong than right


@dataclass(frozen=True)
class Advice:
    action: str  # ACCEPT | FIX | NO_BUBBLE: the recommended one
    text: str  # what is wrong and what to do (one or two short sentences)
    guess: bool = False  # the frame shows Auto's guess: Accept stores it


def advise(state: AppState, frame: int, item: ReviewItem | None) -> Advice:
    """The recommended action for *frame* (listed as *item*, None if not flagged)."""
    radius = state.radius
    if radius is None or not 0 <= frame < len(radius):
        return Advice(FIX, "No result.")
    guess = shown_suggestion(state, frame)
    if guess is not None:
        return _advise_guess(state, frame, guess.radius)
    if not radius[frame] > 0:
        if not _sizes_around(state, frame):
            return Advice(NO_BUBBLE, "No result, and nothing on the frames around it either: "
                                     "probably no bubble here.")
        return Advice(FIX, "No result, but the frames around it show a bubble. "
                           "Fix it, or choose No bubble if there is none.")
    if item is None:
        return Advice(ACCEPT, "Not flagged. Accept it if the outline sits on the bubble wall.")
    problem = _problem(item)
    if any(r in item.reasons for r in _LIKELY_WRONG):
        return Advice(FIX, f"{problem}. Fix it if the outline is off the bubble wall.")
    return Advice(ACCEPT, f"{problem}. Accept it if the outline sits on the bubble wall.")


def _problem(item: ReviewItem) -> str:
    """The item's leading reason in plain words, capitalised."""
    reason = next((r for r in item.reasons if r not in CONTEXT_REASONS and r != GUESS),
                  item.reasons[0] if item.reasons else "")
    text = reason_text(reason, item.confidence) if reason else "Worth a look"
    if reason == "low_confidence" and np.isfinite(item.confidence):
        text = f"Low confidence ({item.confidence:.2f})"
    return text[:1].upper() + text[1:]


def _advise_guess(state: AppState, frame: int, guess_radius: float) -> Advice:
    sizes = _sizes_around(state, frame)
    if len(sizes) < 2 or abs(guess_radius - float(np.median(sizes))) <= \
            GUESS_TOLERANCE * float(np.median(sizes)):
        return Advice(ACCEPT, "No result, but Auto has a guess (dashed outline). "
                              "Accept it if it sits on the bubble wall.", guess=True)
    return Advice(FIX, "Auto's guess (dashed) does not match the frames around it. "
                       "Fix it, or choose No bubble if there is none.", guess=True)


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
