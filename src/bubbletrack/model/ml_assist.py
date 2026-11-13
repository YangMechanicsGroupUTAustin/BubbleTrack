"""ML assist and the app's results (pure, Qt-free).

Which frames are verified examples, what a frame's stored plain-Auto result
is, which frames to suggest, and the short texts the UI shows.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from bubbletrack.model.ml_pipeline import (
    FLAG_GUIDED, FLAG_OUTLINE, FLAG_UNFAMILIAR, ML_FLAGS, RESULT_FLAGS, MLJob, StoredAuto,
)
from bubbletrack.model.provenance import USER_CONFIRMED_SOURCES, FitSource
from bubbletrack.model.teach.examples import BACKGROUND, BUBBLE, LABELS, MLExamples

SUGGESTED_FRAMES = 2  # the study: two verified frames of different sizes teach best
HINT_SHARE = 0.2  # suggest ML assist when this share of the bubble's frames has no result ...
HINT_MIN_FRAMES = 10  # ... and at least this many of them
BRUSH_FRACTION = 0.12  # default brush radius, relative to the typical bubble radius
BRUSH_RANGE = (2, 40)  # px
DEFAULT_BUBBLE_RADIUS = 30.0  # px, before anything is fitted
UNAVAILABLE_TIP = "ML assist needs scikit-learn (pip install scikit-learn)."
PLAIN_AUTO_SOURCES = frozenset({FitSource.AUTO, FitSource.REFIT, FitSource.NEIGHBOR, FitSource.TRACKED})
"""Computed sources whose result is plain Auto's own measurement."""


def ml_available() -> bool:
    """True when scikit-learn can be imported (training needs it)."""
    return importlib.util.find_spec("sklearn") is not None


def _n(state) -> int:
    return 0 if state.radius is None else len(state.radius)


def _locked(state) -> np.ndarray:
    n = _n(state)
    return np.zeros(n, bool) if state.fit_locked is None else np.asarray(state.fit_locked, bool)


def _sources(state) -> np.ndarray:
    n = _n(state)
    return np.zeros(n, int) if state.fit_source is None else np.asarray(state.fit_source, int)


def protected_frames(state) -> frozenset[int]:
    """Frames ML assist never measures: manual fits, circle fixes and locked frames."""
    confirmed = np.isin(_sources(state), [int(s) for s in USER_CONFIRMED_SOURCES])
    return frozenset(int(i) for i in np.flatnonzero(confirmed | _locked(state)))


def verified_circles(state) -> dict[int, tuple[float, float, float]]:
    """{frame: (rc, cc, R)} (1-based) of the frames the user decided with a bubble.

    Manual fits and circle fixes, and locked (accepted) frames with a result;
    a locked frame without one ("no bubble here") is not a bubble example.
    """
    if state.radius is None or state.circle_fit_par is None:
        return {}
    out = {}
    for i in sorted(protected_frames(state)):
        r = float(state.radius[i])
        rc, cc = (float(v) for v in state.circle_fit_par[i])
        if np.isfinite(r) and r > 0 and np.isfinite(rc) and np.isfinite(cc):
            out[i] = (rc, cc, r)
    return out


def examples_for_run(state, user_examples: MLExamples | None = None) -> MLExamples:
    """The training examples of a run: the user's strokes (*user_examples*, by
    default the recording's ``state.ml_examples``) with every verified frame.

    The one place where the example source is decided (teaching preview,
    Apply and Fit All all train from it), so another source can plug in here.
    """
    if user_examples is None:
        user_examples = state.ml_examples
    base = user_examples if isinstance(user_examples, MLExamples) else MLExamples()
    return base.with_verified(verified_circles(state))


def make_job(state, examples: MLExamples, read_raw: Callable[[int], np.ndarray],
             reference: np.ndarray | None) -> MLJob:
    """The pool job that trains on *examples* (their frames are read here)."""
    frames = {f: read_raw(f) for f in examples.frames}
    return MLJob(examples.to_dict(), frames, tuple(state.gridx), tuple(state.gridy), reference)


def stroke_count(examples: MLExamples) -> int:
    """Painted strokes (connected painted areas per frame and label)."""
    total = 0
    for strokes in examples.strokes.values():
        for label in LABELS:
            px = np.array(sorted(strokes.of(label)), dtype=np.int64)
            if not len(px):
                continue
            px -= px.min(axis=0)
            mask = np.zeros(tuple(px.max(axis=0) + 1), bool)
            mask[px[:, 0], px[:, 1]] = True
            total += int(ndimage.label(mask, structure=np.ones((3, 3)))[1])
    return total


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def readiness(examples: MLExamples) -> tuple[bool, str]:
    """(ready, quality line): training needs a bubble and a background example."""
    verified = len(examples.verified)
    painted = [examples.strokes[f] for f in examples.strokes]
    bubble = verified > 0 or any(s.bubble for s in painted)
    background = verified > 0 or any(s.background for s in painted)
    ready = bubble and background
    head = f"{_plural(verified, 'verified frame')}, {_plural(stroke_count(examples), 'stroke')}"
    return ready, f"{head} — " + ("ready" if ready else "needs at least one bubble and one "
                                                        "background example")


def stored_auto(state, idx: int) -> StoredAuto | None:
    """Frame *idx*'s stored plain-Auto result, or None when it has none to offer.

    Accepted Auto results (Fit All, refits, tracking) and Auto rejections are
    usable; ML results, user decisions, unscored and locked frames are not.
    """
    conf = float(state.fit_confidence[idx]) if state.fit_confidence is not None else np.nan
    flags = frozenset(state.fit_flags[idx]) if state.fit_flags is not None else frozenset()
    if not np.isfinite(conf) or flags & RESULT_FLAGS or _locked(state)[idx]:
        return None
    try:
        source = FitSource(int(_sources(state)[idx]))
    except ValueError:
        return None
    metrics = {k: float(v[idx]) for k, v in (state.fit_metrics or {}).items() if np.isfinite(v[idx])}
    radius = float(state.radius[idx])
    if radius <= 0:
        if source != FitSource.NONE:
            return None
        return StoredAuto(np.nan, np.nan, np.nan, conf, flags - ML_FLAGS, None, metrics)
    xy = state.circle_xy[idx] if state.circle_xy is not None else None
    rc, cc = (float(v) for v in state.circle_fit_par[idx])
    if source not in PLAIN_AUTO_SOURCES or xy is None or not (np.isfinite(rc) and np.isfinite(cc)):
        return None
    return StoredAuto(radius, rc, cc, conf, flags - ML_FLAGS, np.asarray(xy, float), metrics)


def stored_auto_map(state, frames: Iterable[int]) -> dict[int, StoredAuto]:
    out = {}
    for i in frames:
        stored = stored_auto(state, i)
        if stored is not None:
            out[int(i)] = stored
    return out


def suggestions(state, n: int = SUGGESTED_FRAMES, *, reference: int | None = None) -> tuple[int, ...]:
    """Frames to look at first (``suggest_frames`` on Auto's results).

    Verified and locked frames and the reference frame are left out.
    """
    from bubbletrack.model.teach import suggest_frames

    if state.radius is None or not len(state.radius):
        return ()
    radius = np.where(np.asarray(state.radius) > 0, state.radius, np.nan)
    conf = state.fit_confidence if state.fit_confidence is not None else np.zeros(len(radius))
    conf = np.where(np.isfinite(conf), conf, 0.0)
    exclude = set(protected_frames(state)) | ({reference} if reference is not None else set())
    if len(exclude) >= len(radius):
        return ()
    return suggest_frames(radius, conf, n=n, exclude=exclude)


def default_brush(state) -> int:
    """Brush radius (px) scaled to the recording's typical bubble: the verified
    frames' when there are any (Auto's sizes are unreliable where help is needed)."""
    verified = [r for _, _, r in verified_circles(state).values()]
    radius = np.asarray(verified or (state.radius if state.radius is not None else []), float)
    fitted = radius[np.isfinite(radius) & (radius > 0)]
    typical = float(np.median(fitted)) if len(fitted) else DEFAULT_BUBBLE_RADIUS
    return int(np.clip(round(BRUSH_FRACTION * typical), *BRUSH_RANGE))


@dataclass(frozen=True)
class Failures:
    """Frames without a result inside the bubble's lifetime in a run."""

    missing: int
    """Frames with no result between the first and the last frame with one."""
    span: int
    """Frames between them (user-decided ones left out)."""

    @property
    def share(self) -> float:
        return self.missing / self.span if self.span else 0.0

    def worth_a_hint(self, share: float | None = None, min_frames: int | None = None) -> bool:
        share = HINT_SHARE if share is None else share
        min_frames = HINT_MIN_FRAMES if min_frames is None else min_frames
        return self.missing >= min_frames and self.share >= share


def failures(state, frames: Iterable[int]) -> Failures:
    """Missing results of a run between its first and last frame with a result.

    Frames before the first and after the last bubble are routine (before
    cavitation, after the cloud), and listed frames with a result are not
    failures; manual and locked frames are the user's and do not count.
    """
    if state.radius is None:
        return Failures(0, 0)
    order = sorted(int(i) for i in frames)
    fitted = [i for i in order if state.radius[i] > 0]
    if not fitted:
        return Failures(0, 0)
    skip = protected_frames(state)
    inside = [i for i in order if fitted[0] <= i <= fitted[-1] and i not in skip]
    return Failures(sum(1 for i in inside if not state.radius[i] > 0), len(inside))


def run_counts(state, frames: Iterable[int]) -> dict[str, int]:
    """How the frames of an ML-assisted run came out (protected ones left out)."""
    skip = protected_frames(state)
    counts = {"guided": 0, "auto": 0, "outline": 0, "none": 0, "unfamiliar": 0}
    flags_of = state.fit_flags or [frozenset()] * _n(state)
    for i in frames:
        if i in skip:
            continue
        flags = flags_of[i]
        if not state.radius[i] > 0:
            counts["none"] += 1
        elif int(state.fit_source[i]) == FitSource.ML or FLAG_OUTLINE in flags:
            counts["outline"] += 1
        elif FLAG_GUIDED in flags:
            counts["guided"] += 1
        else:
            counts["auto"] += 1
        counts["unfamiliar"] += FLAG_UNFAMILIAR in flags
    return counts


def run_summary(counts: dict[str, int]) -> str:
    """The note of an ML-assisted run, e.g. 'ML assist: 40 ML outlines to review'."""
    parts = [f"{_plural(counts['outline'], 'ML outline')} to review"]
    if counts["unfamiliar"]:
        parts.append(f"{_plural(counts['unfamiliar'], 'frame')} unlike the taught ones")
    return "ML assist: " + ", ".join(parts)


def state_text(on: bool, examples: MLExamples | None) -> str:
    """The Auto box's state line: 'Off', 'Off (2 examples)', 'On (3 examples)'
    (examples = frames with verified circles or strokes)."""
    n = len(examples.frames) if isinstance(examples, MLExamples) else 0
    head = "On" if on else "Off"
    return f"{head} ({_plural(n, 'example')})" if n or on else head


__all__ = [
    "BACKGROUND", "BUBBLE", "HINT_SHARE", "SUGGESTED_FRAMES", "UNAVAILABLE_TIP", "default_brush",
    "examples_for_run", "failures", "make_job", "ml_available", "protected_frames", "readiness", "run_counts",
    "run_summary", "state_text", "stored_auto", "stored_auto_map", "stroke_count", "suggestions",
    "verified_circles",
]
