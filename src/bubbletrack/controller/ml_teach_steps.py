"""ML assist's teaching steps, in plain words (Qt-free).

Step 1: confirm two frames where the outline is right, of different
situations (``model.teach.suggest.differ``).  Every frame the user decided
with a bubble counts (Looks right, Review's Accept, a fix, a manual fit), and
so does a frame painted by hand once it has bubble *and* background strokes.

Step 2: Train and apply.  It measures the frames that need review: the frames
listed in Review and the frames without a result.  The user's frames (manual,
fixed, locked) and the frames Auto measured without a flag are never touched.

The panel shows ``step_text`` (where the user is), the cards, and
``next_text`` (what to do next, one sentence) in every state.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable
from dataclasses import dataclass

import numpy as np

from bubbletrack.model.ml_assist import protected_frames, verified_circles
from bubbletrack.model.suggestion import shown_suggestion, suggested_mask
from bubbletrack.model.teach import MLExamples
from bubbletrack.model.teach.suggest import FrameSize, find_pair
from bubbletrack.ui.wording import plural

NEEDED = 2  # confirmed frames of different situations before training
TRAIN = "Train and apply"
PAINT_SECTION = "Paint examples by hand"


@dataclass(frozen=True)
class Progress:
    """The examples so far and whether two of them differ."""

    examples: tuple[FrameSize, ...]
    """Every example frame (confirmed outline, or painted), by frame."""
    pair: tuple[FrameSize, FrameSize] | None
    """Two examples of different situations (step 1 done), else None."""

    @property
    def ready(self) -> bool:
        return self.pair is not None

    @property
    def done(self) -> int:
        return NEEDED if self.ready else min(len(self.examples), NEEDED - 1)

    @property
    def frames(self) -> frozenset[int]:
        return frozenset(e.frame for e in self.examples)


def example_frames(state) -> tuple[FrameSize, ...]:
    """Confirmed frames (with their radius) and painted frames (size unknown)."""
    circles = verified_circles(state)
    out = [FrameSize(f, r) for f, (_, _, r) in circles.items()]
    painted = state.ml_examples
    if isinstance(painted, MLExamples):
        out += [FrameSize(f) for f, s in painted.strokes.items()
                if f not in circles and s.bubble and s.background]
    return tuple(sorted(out, key=lambda e: e.frame))


def progress(state, prefer: Collection[int] = ()) -> Progress:
    """Where step 1 stands; pairs holding a frame of *prefer* (on show) win."""
    examples = example_frames(state)
    n = 0 if state.radius is None else len(state.radius)
    return Progress(examples, find_pair(examples, n, prefer))


@dataclass(frozen=True, eq=False)
class Outline:
    """The outline a card asks about (1-based full-image coordinates)."""

    radius: float
    rc: float
    cc: float
    xy: np.ndarray | None
    """(N, 2) [row, col] points; None: draw the circle."""
    guess: bool
    """Auto's guess on a frame without a result (accepting it stores it)."""


def outline(state, frame: int) -> Outline | None:
    """Frame *frame*'s result, else Auto's shown guess, else None."""
    if state.radius is None or not 0 <= frame < len(state.radius):
        return None
    r = float(state.radius[frame])
    if r > 0:
        rc, cc = (float(v) for v in state.circle_fit_par[frame])
        xy = state.circle_xy[frame] if state.circle_xy is not None else None
        return Outline(r, rc, cc, None if xy is None else np.asarray(xy, dtype=float), False)
    guess = shown_suggestion(state, frame)
    if guess is None:
        return None
    return Outline(guess.radius, guess.rc, guess.cc, np.asarray(guess.edge_xy, dtype=float), True)


def outlines(state) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per frame: (radius of the outline to judge (NaN: none), its confidence,
    True for a measured result / False for Auto's guess), for ``pick_candidate``."""
    if state.radius is None:
        return np.empty(0), np.empty(0), np.empty(0, bool)
    n = len(state.radius)
    measured = np.asarray(state.radius, dtype=float) > 0
    radius = np.where(measured, np.asarray(state.radius, dtype=float), np.nan)
    conf = (np.full(n, np.nan) if state.fit_confidence is None
            else np.asarray(state.fit_confidence, dtype=float).copy())
    for f in np.flatnonzero(suggested_mask(state, n)):
        guess = shown_suggestion(state, int(f))
        if guess is not None:
            radius[f], conf[f] = guess.radius, guess.confidence
    return radius, conf, measured


def train_scope(state, listed: Iterable[int]) -> tuple[int, ...]:
    """The frames Train and apply measures: *listed* (in Review) and the frames
    without a result, minus the user's own (manual, fixed, locked)."""
    if state.radius is None:
        return ()
    n = len(state.radius)
    empty = np.flatnonzero(~(np.asarray(state.radius, dtype=float) > 0))
    frames = {int(f) for f in listed if 0 <= int(f) < n} | {int(f) for f in empty}
    return tuple(sorted(frames - protected_frames(state)))


# -- words ------------------------------------------------------------------ #

def step_text(p: Progress) -> str:
    """'Step 1 of 2: confirm 2 frames where the outline is right — 1 of 2 done'."""
    if not p.ready:
        return (f"Step 1 of 2: confirm {NEEDED} frames where the outline is right"
                f" — {p.done} of {NEEDED} done")
    n = len(p.examples)
    what = f"{NEEDED} different frames confirmed" if n == NEEDED else f"{n} frames confirmed"
    return f"Step 2 of 2: {TRAIN} — {what}"


def next_text(p: Progress, *, candidates: int, scope: int) -> tuple[str, bool]:
    """(the next step in one sentence, warn) for the panel's state."""
    if p.ready:
        if not scope:
            return ("Nothing needs ML assist: every frame has a result and none is listed "
                    "in Review."), True
        return (f"Press {TRAIN}: ML assist learns from your frames and measures the "
                f"{plural(scope, 'frame')} that need review."), False
    first = p.examples[0].frame + 1 if p.examples else None
    if candidates:
        if first is None:
            return ("Judge each frame: Looks right ✓ if the red outline sits on the bubble wall, "
                    "Looks wrong ✗ to see another frame."), False
        if len(p.examples) == 1:
            return f"One more: judge the other frame, which differs from frame {first}.", False
        return (f"Your {len(p.examples)} confirmed frames are alike (similar size, close in time): "
                "judge the other frame, which differs."), False
    if first is None:
        return (f"No frame with an outline is left to judge: open “{PAINT_SECTION}” below and "
                "paint the bubble and the background on two frames."), True
    return (f"No other frame with an outline is left: open “{PAINT_SECTION}” below and paint "
            f"a frame that differs from frame {first} (another size or another part of the "
            "recording)."), True


def train_state(p: Progress, *, scope: int) -> tuple[bool, str]:
    """(enabled, tooltip) of the Train and apply button."""
    if not p.ready:
        return False, f"Confirm {NEEDED} different frames first ({p.done} of {NEEDED} done)"
    if not scope:
        return False, "Nothing to measure: every frame has a result and none is listed in Review"
    return True, (f"ML assist learns from your confirmed (and painted) frames, then measures\n"
                  f"the {plural(scope, 'frame')} that need review: the frames listed in Review and\n"
                  "the frames without a result. Your confirmed frames and the frames\n"
                  "Auto measured without a flag are not touched.")


def summary_text(measured: int, total: int, listed: int, *, stopped: bool) -> str:
    """'ML assist measured 41 of 52 frames to check; 9 are listed in Review'."""
    if listed == 0:
        tail = "none is left to check in Review"
    else:
        tail = f"{listed} {'is' if listed == 1 else 'are'} listed in Review"
    if stopped:
        return f"ML assist stopped: {measured} of {total} frames to check measured; {tail}"
    return f"ML assist measured {measured} of {total} frames to check; {tail}"
