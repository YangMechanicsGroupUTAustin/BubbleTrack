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
NONE = ""  # a settled (locked) frame: no action is recommended
NEIGHBOURHOOD = 3  # frames on each side that the advice compares with
GUESS_TOLERANCE = 0.3  # a guess within 30 % of the radius around it fits
_LIKELY_WRONG = ("lost", "jump")  # a result with these is more often wrong than right


@dataclass(frozen=True)
class Advice:
    action: str  # ACCEPT | FIX | NO_BUBBLE: the recommended one
    text: str  # what is wrong and what to do (one or two short sentences)
    guess: bool = False  # the frame shows Auto's guess: Accept stores it
    empty: bool = False  # no result and no guess: nothing to accept

    @property
    def can_accept(self) -> bool:
        return not self.empty


def advise(state: AppState, frame: int, item: ReviewItem | None) -> Advice:
    """The recommended action for *frame* (listed as *item*, None if not flagged)."""
    radius = state.radius
    if radius is None or not 0 <= frame < len(radius):
        return Advice(FIX, "No result.", empty=True)
    locked = state.fit_locked
    if locked is not None and locked[frame]:  # already settled: nothing to recommend
        if radius[frame] > 0:
            return Advice(NONE, "Accepted (locked). Fix it if the outline is off the wall.")
        return Advice(NONE, "Marked as no bubble (locked).", empty=True)
    guess = shown_suggestion(state, frame)
    if guess is not None:
        return _advise_guess(state, frame, guess.radius)
    if not radius[frame] > 0:
        return _advise_empty(state, frame)
    if item is None:
        return Advice(ACCEPT, "Not flagged. Accept it if the outline sits on the bubble wall.")
    wrong = next((r for r in _LIKELY_WRONG if r in item.reasons), None)
    if wrong is not None:  # say the reason that makes it likely wrong
        return Advice(FIX, f"{_capital(reason_text(wrong, item.confidence))}. "
                           "Fix it if the outline is off the bubble wall.")
    return Advice(ACCEPT, f"{_problem(item)}. Accept it if the outline sits on the bubble wall.")


def _advise_empty(state: AppState, frame: int) -> Advice:
    if any(state.radius[i] > 0 for i in _around(len(state.radius), frame)):
        return Advice(FIX, "No result, but the frames around it have one. "
                           "Fix it, or choose No bubble if there is none.", empty=True)
    if _sizes_around(state, frame):
        return Advice(FIX, "No result; Auto only has guesses on the frames around it. "
                           "Fix it, or choose No bubble if there is none.", empty=True)
    return Advice(NO_BUBBLE, "No result, and nothing on the frames around it either: "
                             "probably no bubble here.", empty=True)


def _problem(item: ReviewItem) -> str:
    """The item's leading reason in plain words, capitalised."""
    reason = next((r for r in item.reasons if r not in CONTEXT_REASONS and r != GUESS),
                  item.reasons[0] if item.reasons else "")
    text = reason_text(reason, item.confidence) if reason else "Worth a look"
    if reason == "low_confidence" and np.isfinite(item.confidence):
        text = f"Low confidence ({item.confidence:.2f})"
    return _capital(text)


def _capital(text: str) -> str:
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
