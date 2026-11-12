"""One tracking pass (forward or backward from a seed) and the shared re-detection cache.

Per frame, in order (the "cv1" prototype rules):

1. the raw observation inside the tight gate                      -> ``raw``
   (optionally, below ``refine_below_confidence``, also ``redetect`` and keep the more confident)
2. else ``redetect(frame, prior)`` inside the tight gate          -> ``tracked``
3. else that re-detection inside the wide gate (confidence 0.3)   -> ``tracked`` + ``jump``
4. else the raw observation inside the wide gate (confidence 0.5) -> ``raw`` + ``jump``
5. else either one centred inside the tight gate, any radius (0.5) -> ``jump``
   (re-nucleation / growth from a residual bubble, or a collapse seen backwards)
6. else ``lost``: keep predicting; the gates grow.

An acceptance right after a loss is flagged ``reacquired`` and needs
``release_confidence``.  When the predicted radius collapses (below
``release_ratio`` of the last one) the radius window is released to
[0, ceiling] so the minimum and the rebound are accepted.

After such a release, ``cloud_frames`` consecutive cloud-like frames end a
forward pass: flagged ``fragmented`` by the detector, or small with nothing solid
on the track and no object persisting from the previous frame (a weak residual
bubble persists, fragments do not).  The remaining frames are ``none`` +
``cloud``.  A backward pass is never cut short (a wrong stop would erase whole
earlier cycles); at either sequence end or a "no bubble" anchor, a trailing run
of cloud-like frames (after a release) or of lost frames where nothing at all was
detected is ``none``: no bubble yet / any more.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from bubbletrack.model.sequence.config import SequenceConfig
from bubbletrack.model.sequence.gating import (
    Prediction, TrackPoint, centre_error, centre_within, confident, predict, radius_error, within,
)
from bubbletrack.model.sequence.types import (
    FLAG_CLOUD, FLAG_JUMP, FLAG_LOST, FLAG_REACQUIRED, NO_OBSERVATION, SOURCE_LOST, SOURCE_NONE,
    SOURCE_RAW, SOURCE_TRACKED, Detector, Observation,
)

logger = logging.getLogger(__name__)

_FRAGMENTED = "fragmented"
_HISTORY = 3


@dataclass(frozen=True)
class Step:
    """One frame's outcome in one pass."""

    obs: Observation
    """The measurement reported (raw, re-detected, or the unverified raw one when lost)."""
    source: str
    flags: frozenset = frozenset()
    again: Observation | None = None
    """The re-detection made for this frame, if any (used by the cloud test)."""

    @property
    def accepted(self) -> bool:
        return self.source in (SOURCE_RAW, SOURCE_TRACKED)


class RedetectCache:
    """Calls ``redetect`` at most once per (frame, similar prior); reuses any stored
    result of the same frame that already fits the current gate."""

    def __init__(self, redetect: Detector) -> None:
        self._redetect = redetect
        self._store: dict[int, list[tuple[tuple[float, float, float], Observation]]] = {}
        self.calls = 0

    def get(self, frame: int, pred: Prediction, cfg: SequenceConfig) -> Observation:
        prior = pred.prior(cfg)
        for old_prior, obs in self._store.get(frame, ()):
            if within(obs, pred) or _same_prior(old_prior, prior):
                return obs
        obs = self._call(frame, prior)
        self._store.setdefault(frame, []).append((prior, obs))
        return obs

    def _call(self, frame: int, prior: tuple[float, float, float]) -> Observation:
        self.calls += 1
        try:
            obs = self._redetect(frame, prior)
        except Exception:  # a bad frame must not abort the whole solve
            logger.warning("redetect failed on frame %d with prior %s", frame, prior, exc_info=True)
            return NO_OBSERVATION
        if not isinstance(obs, Observation):
            raise TypeError(f"redetect must return an Observation, got {type(obs).__name__}")
        return obs


def _same_prior(a: tuple[float, float, float], b: tuple[float, float, float]) -> bool:
    return math.hypot(a[0] - b[0], a[1] - b[1]) <= 1.0 and abs(a[2] - b[2]) <= max(0.5, 0.05 * a[2])


def _better(again: Observation, raw: Observation, pred: Prediction) -> bool:
    return within(again, pred) and again.confidence > raw.confidence


def measure(frame: int, raw: Observation, pred: Prediction, coast: int, cache: RedetectCache,
            cfg: SequenceConfig) -> Step:
    """Apply the acceptance rules to one frame."""
    floor = cfg.release_confidence if pred.released or coast > 0 else cfg.min_confidence
    extra = frozenset({FLAG_REACQUIRED}) if coast > 0 else frozenset()
    can_redetect = coast < cfg.max_redetect_coast
    if within(raw, pred) and confident(raw, floor):
        if can_redetect and raw.confidence < cfg.refine_below_confidence:  # weak: try the prior too
            again = cache.get(frame, pred, cfg)
            if _better(again, raw, pred):
                return Step(again, SOURCE_TRACKED, extra, again)
        return Step(raw, SOURCE_RAW, extra)
    wide = cfg.wide_gate_factor
    again = cache.get(frame, pred, cfg) if can_redetect else None
    if again is not None:
        if within(again, pred) and confident(again, floor):
            return Step(again, SOURCE_TRACKED, extra, again)
        if within(again, pred, wide) and confident(again, max(floor, cfg.wide_confidence)):
            return Step(again, SOURCE_TRACKED, extra | {FLAG_JUMP}, again)
    if within(raw, pred, wide) and confident(raw, cfg.reacquire_confidence):
        return Step(raw, SOURCE_RAW, extra | {FLAG_JUMP}, again)
    for obs, source in ((again, SOURCE_TRACKED), (raw, SOURCE_RAW)):    # same centre, new size
        if obs is not None and centre_within(obs, pred) and confident(obs, cfg.reacquire_confidence):
            return Step(obs, source, extra | {FLAG_JUMP}, again)
    return Step(raw, SOURCE_LOST, frozenset({FLAG_LOST}), again)


def _fragmented(step: Step, raw: Observation, pred: Prediction, rmax: float,
                cfg: SequenceConfig) -> bool:
    """Fragments around a small main body (below ``cloud_radius_frac`` x Rmax).

    Fragments next to a full-size bubble are satellites or debris (the engine
    counts any other dark region within ~2 R), not a cloud.  Measured on LIC:
    LIC 151 -> 182 / 235 solver-correct frames, lic_hard 43 -> 70 / 182; a
    confidence test instead (>= reacquire_confidence) gave 158 and 43."""
    small = cfg.cloud_radius_frac * rmax
    for obs in (raw, step.obs):
        if _FRAGMENTED in obs.flags and (obs.radius if obs.ok else pred.last_radius) < small:
            return True
    return False


def _small_and_unsupported(step: Step, raw: Observation, pred: Prediction, rmax: float,
                           cfg: SequenceConfig) -> bool:
    """Small (below ``cloud_radius_frac`` x the pass's Rmax) with nothing solid on the track.

    Solid = any measurement of the frame (raw or re-detected) centred inside the
    tight gate with at least ``min_confidence`` (its radius may jump: collapses are
    fast), or a confident raw bubble of non-cloud size inside the wide gate."""
    small = cfg.cloud_radius_frac * rmax
    r_eff = step.obs.radius if step.accepted else pred.last_radius
    if not r_eff < small:
        return False
    seen = (step.obs, raw) if step.again is None else (step.obs, raw, step.again)
    if any(centre_within(m, pred) and confident(m, cfg.min_confidence) for m in seen):
        return False
    return not (within(raw, pred, cfg.wide_gate_factor) and raw.radius >= small
                and confident(raw, cfg.low_confidence))


def _near_track(step: Step, raw: Observation, pred: Prediction, cfg: SequenceConfig) -> tuple[Observation, ...]:
    """Every detection of the frame (raw or re-detected) inside the wide centre gate."""
    found = (step.obs, raw) if step.again is None else (step.obs, raw, step.again)
    return tuple(m for m in found if centre_within(m, pred, cfg.wide_gate_factor))


def _persistent(seen: tuple[Observation, ...], before: tuple[Observation, ...]) -> bool:
    """Did the same small object appear in the previous frame (a residual bubble, not fragments)?"""
    for a in seen:
        for b in before:
            r = max(a.radius, b.radius)
            if (math.hypot(a.rc - b.rc, a.cc - b.cc) <= max(3.0, 0.3 * r)
                    and abs(a.radius - b.radius) <= max(2.0, 0.25 * r)):
                return True
    return False


def run_pass(raw: tuple[Observation, ...], start: TrackPoint, direction: int, stop: int,
             cache: RedetectCache, cfg: SequenceConfig, *, stop_is_end: bool = True) -> dict[int, Step]:
    """Track from ``start`` (exclusive) towards ``stop`` (exclusive) in ``direction`` (+1 / -1).

    ``stop_is_end``: ``stop`` is the sequence boundary or a "no bubble" anchor (the trailing
    frames may then become ``none``), not another seed whose own passes cover them."""
    history: tuple[TrackPoint, ...] = (start,)
    c_err: tuple[float, ...] = ()
    r_err: tuple[float, ...] = ()
    coast, armed, rmax = 0, False, start.radius
    run: list[int] = []
    seen: tuple[Observation, ...] = ()
    steps: dict[int, Step] = {}
    frame = start.frame + direction
    while frame != stop and 0 <= frame < len(raw):
        pred = predict(history, frame, c_err, r_err, coast, cfg)
        armed = armed or pred.released
        step = measure(frame, raw[frame], pred, coast, cache, cfg)
        steps[frame] = step
        if step.accepted:
            obs = step.obs
            c_err = (c_err + (centre_error(obs, pred),))[-cfg.error_window:]
            if not pred.released:
                r_err = (r_err + (radius_error(obs, pred),))[-cfg.error_window:]
            history = history[-(_HISTORY - 1):] + (TrackPoint(frame, obs.rc, obs.cc, obs.radius),)
            rmax = max(rmax, obs.radius)
            coast = 0
        else:
            coast += 1
        before, seen = seen, _near_track(step, raw[frame], pred, cfg)
        cloudy = armed and (_fragmented(step, raw[frame], pred, rmax, cfg)
                            or (_small_and_unsupported(step, raw[frame], pred, rmax, cfg)
                                and not _persistent(seen, before)))
        run = run + [frame] if cloudy else []
        if direction > 0 and len(run) >= cfg.cloud_frames:
            return _terminate(steps, run[0], direction, stop, len(raw), cloud=True)
        frame += direction
    if not stop_is_end:                # stopped at another seed: that one's passes take over
        return steps
    first = _empty_tail(steps, raw, direction)
    if run and (first is None or (run[0] - first) * direction < 0):
        first = run[0]
    if first is None:
        return steps
    return _terminate(steps, first, direction, stop, len(raw), cloud=armed and direction > 0)


def _empty_tail(steps: dict[int, Step], raw: tuple[Observation, ...], direction: int) -> int | None:
    """First frame of the trailing run of lost frames where nothing at all was detected."""
    first = None
    for f in sorted(steps, reverse=direction > 0):
        step = steps[f]
        if step.source != SOURCE_LOST or raw[f].ok or (step.again is not None and step.again.ok):
            break
        first = f
    return first


def _terminate(steps: dict[int, Step], first: int, direction: int, stop: int, n: int, *,
               cloud: bool) -> dict[int, Step]:
    """Frames from ``first`` to the pass end have no bubble (``cloud`` after a collapse)."""
    flags = frozenset({FLAG_CLOUD}) if cloud else frozenset()
    end = min(n, stop) if direction > 0 else max(-1, stop)
    kept = {f: s for f, s in steps.items() if (f - first) * direction < 0}
    kept.update({f: Step(NO_OBSERVATION, SOURCE_NONE, flags) for f in range(first, end, direction)})
    return kept
