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

After such a release, ``cloud_frames`` consecutive cloud-like frames are a
post-collapse fragment cloud.  A frame is cloud-like when no single body persists
on the track (no detection near it lies within max(2 px, 0.2 R) of one in the
previous frame: a bubble of any size stays put, the main body of a cloud hops
between fragments) and there are fragments: the detector flags the tracked
(accepted) body ``fragmented``, or the tracked body is small (below
``cloud_radius_frac`` x the bubble's Rmax; for a lost frame its last measured
radius, while the loss is recent).  An unverified detection elsewhere says nothing
about the tracked bubble.  A forward pass marks the cloud ``none`` + ``cloud``
until a bubble re-forms (a rebound or the next pulse: two consecutive confident
raw detections of one body); tracking restarts there.  A backward pass is never
cut short (a wrong stop would erase whole earlier cycles); at either sequence end
or a "no bubble" anchor, a trailing run of cloud-like frames (after a release) or
of lost frames where nothing at all was detected is ``none``: no bubble yet / any
more.
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
_SAME_PLACE_PX = 2.0
_SAME_PLACE_FRAC = 0.2
"""Two detections in consecutive frames are one body when their centres lie within
max(2 px, 0.2 R) (radius free: collapses are fast).  On the LIC recordings 95 % of
consecutive verified frames move by at most 0.1 R (98 % by 0.2 R); the main body of a
fragment cloud hops by a large fraction of R."""


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


def _fragmented(step: Step) -> bool:
    """The detector reports fragments around the tracked body (the frame's accepted measurement).

    An unverified detection elsewhere says nothing about the tracked bubble: on edge-1.1.0
    the single-frame result often picks a debris particle beside a well-tracked bubble."""
    return step.accepted and _FRAGMENTED in step.obs.flags


def _small(step: Step, pred: Prediction, coast: int, rmax: float, cfg: SequenceConfig) -> bool:
    """The tracked body is below ``cloud_radius_frac`` x the bubble's Rmax.

    A lost frame only shows the last measured radius: it counts while the loss is recent
    (at most ``max_redetect_coast`` lost frames); after a long loss the bubble may have
    regrown or drifted."""
    if step.accepted:
        radius = step.obs.radius
    elif coast <= cfg.max_redetect_coast:
        radius = pred.last_radius
    else:
        return False
    return radius < cfg.cloud_radius_frac * rmax


def _near_track(step: Step, raw: Observation, pred: Prediction, cfg: SequenceConfig) -> tuple[Observation, ...]:
    """Every detection of the frame (raw or re-detected) inside the wide centre gate."""
    found = (step.obs, raw) if step.again is None else (step.obs, raw, step.again)
    return tuple(m for m in found if centre_within(m, pred, cfg.wide_gate_factor))


def _same_place(a: Observation, b: Observation) -> bool:
    """One body in consecutive frames (see ``_SAME_PLACE_PX``)."""
    reach = max(_SAME_PLACE_PX, _SAME_PLACE_FRAC * max(a.radius, b.radius))
    return math.hypot(a.rc - b.rc, a.cc - b.cc) <= reach


def _persistent(seen: tuple[Observation, ...], before: tuple[Observation, ...]) -> bool:
    """Did a body near the track stay put since the previous frame (a bubble, not fragments)?"""
    return any(_same_place(a, b) for a in seen for b in before)


@dataclass(frozen=True)
class _Segment:
    """One bubble followed by a pass, up to the pass end or (forwards) a confirmed cloud."""

    steps: dict
    last: int
    """Last frame measured."""
    run: tuple[int, ...]
    """Trailing cloud-like frames."""
    armed: bool
    """A collapse was seen."""
    cloud: bool
    """The segment ended at a cloud starting at ``run[0]``."""


def _follow(raw: tuple[Observation, ...], start: TrackPoint, direction: int, stop: int,
            cache: RedetectCache, cfg: SequenceConfig) -> _Segment:
    """Track one bubble from ``start`` (exclusive) towards ``stop`` (exclusive)."""
    history: tuple[TrackPoint, ...] = (start,)
    c_err: tuple[float, ...] = ()
    r_err: tuple[float, ...] = ()
    coast, armed, rmax = 0, False, start.radius
    run: tuple[int, ...] = ()
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
        cloudy = (armed and not _persistent(seen, before)
                  and (_fragmented(step) or _small(step, pred, coast, rmax, cfg)))
        run = run + (frame,) if cloudy else ()
        if direction > 0 and len(run) >= cfg.cloud_frames:
            return _Segment(steps, frame, run, armed, cloud=True)
        frame += direction
    return _Segment(steps, frame - direction, run, armed, cloud=False)


def _reformed(raw: tuple[Observation, ...], lo: int, hi: int, cfg: SequenceConfig) -> int | None:
    """First frame in ``lo..hi-1`` where a bubble re-forms after a cloud (a rebound or the next
    pulse): it and the next frame hold one body with ``reacquire_confidence`` (raw only)."""
    for f in range(lo, hi - 1):
        a, b = raw[f], raw[f + 1]
        if (a.ok and b.ok and confident(a, cfg.reacquire_confidence)
                and confident(b, cfg.reacquire_confidence) and _same_place(a, b)):
            return f
    return None


def run_pass(raw: tuple[Observation, ...], start: TrackPoint, direction: int, stop: int,
             cache: RedetectCache, cfg: SequenceConfig, *, stop_is_end: bool = True) -> dict[int, Step]:
    """Track from ``start`` (exclusive) towards ``stop`` (exclusive) in ``direction`` (+1 / -1).

    ``stop_is_end``: ``stop`` is the sequence boundary or a "no bubble" anchor (the trailing
    frames may then become ``none``), not another seed whose own passes cover them."""
    steps: dict[int, Step] = {}
    seg = _follow(raw, start, direction, stop, cache, cfg)
    while seg.cloud:                   # forwards only: no bubble until one re-forms
        end = min(len(raw), stop)
        back = _reformed(raw, seg.last + 1, end, cfg)
        steps.update(seg.steps)
        steps.update(_no_bubble(range(seg.run[0], end if back is None else back), cloud=True))
        if back is None:
            return steps
        obs = raw[back]
        steps[back] = Step(obs, SOURCE_RAW, frozenset({FLAG_REACQUIRED}))
        seg = _follow(raw, TrackPoint(back, obs.rc, obs.cc, obs.radius), direction, stop, cache, cfg)
    steps.update(seg.steps)
    if not stop_is_end:                # stopped at another seed: that one's passes take over
        return steps
    first = _empty_tail(steps, raw, direction)
    if seg.run and (first is None or (seg.run[0] - first) * direction < 0):
        first = seg.run[0]
    if first is None:
        return steps
    return _terminate(steps, first, direction, stop, len(raw), cloud=seg.armed and direction > 0)


def _empty_tail(steps: dict[int, Step], raw: tuple[Observation, ...], direction: int) -> int | None:
    """First frame of the trailing run of lost frames where nothing at all was detected."""
    first = None
    for f in sorted(steps, reverse=direction > 0):
        step = steps[f]
        if step.source != SOURCE_LOST or raw[f].ok or (step.again is not None and step.again.ok):
            break
        first = f
    return first


def _no_bubble(frames: range, *, cloud: bool) -> dict[int, Step]:
    """``none`` steps for ``frames`` (flag ``cloud`` after a collapse)."""
    flags = frozenset({FLAG_CLOUD}) if cloud else frozenset()
    return {f: Step(NO_OBSERVATION, SOURCE_NONE, flags) for f in frames}


def _terminate(steps: dict[int, Step], first: int, direction: int, stop: int, n: int, *,
               cloud: bool) -> dict[int, Step]:
    """Frames from ``first`` to the pass end have no bubble (``cloud`` after a collapse)."""
    end = min(n, stop) if direction > 0 else max(-1, stop)
    kept = {f: s for f, s in steps.items() if (f - first) * direction < 0}
    return kept | _no_bubble(range(first, end, direction), cloud=cloud)
