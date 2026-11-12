"""``solve_sequence``: bidirectional tracking over per-frame results, then cycles and review.

1. Seeds: every anchor with a circle; without anchors, the most confident raw
   frame (preferring one that agrees with a neighbour).  Anchors without a
   circle (radius <= 0 / NaN: "no bubble here") are barriers; a stretch between
   barriers that no anchor reaches gets its own automatic seed.
2. From each seed, one pass forwards and one backwards, each stopping at the
   next anchor (so anchors reset the track state) or the sequence end.
3. A frame covered by two passes takes the nearer seed's accepted result
   (flag ``disagree`` when the two passes found different circles).
4. Collapses and cycles from the verified radii; flags; the review list.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

from bubbletrack.model.sequence.config import DEFAULT_SEQUENCE_CONFIG, SequenceConfig
from bubbletrack.model.sequence.cycles import detect_cycles
from bubbletrack.model.sequence.gating import TrackPoint, disagree, predict, within
from bubbletrack.model.sequence.review import build_review
from bubbletrack.model.sequence.tracker import RedetectCache, Step, run_pass
from bubbletrack.model.sequence.types import (
    FLAG_COLLAPSE, FLAG_DISAGREE, FLAG_LOW_CONFIDENCE, SOURCE_ANCHOR, SOURCE_LOST, SOURCE_NONE,
    SOURCE_RAW, SOURCE_TRACKED, Detector, FrameSolution, Observation, SequenceResult,
)

_NAN = float("nan")


def _validate(raw: Sequence[Observation], anchors) -> tuple[tuple[Observation, ...], dict]:
    frames = tuple(raw)
    for i, obs in enumerate(frames):
        if not isinstance(obs, Observation):
            raise TypeError(f"raw[{i}] must be an Observation, got {type(obs).__name__}")
    out = {}
    for key, circle in dict(anchors or {}).items():
        try:
            idx = int(key)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"anchor frame must be an integer index, got {key!r}") from exc
        if idx != key or not 0 <= idx < len(frames):
            raise ValueError(f"anchor frame {key!r} is outside 0..{len(frames) - 1}")
        try:
            rc, cc, r = (float(v) for v in circle)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"anchor {idx} must be (rc, cc, R), got {circle!r}") from exc
        if math.isfinite(r) and r > 0 and not (math.isfinite(rc) and math.isfinite(cc)):
            raise ValueError(f"anchor {idx} has a radius but no finite centre: {circle!r}")
        out[idx] = (rc, cc, r)
    return frames, out


def _has_circle(circle: tuple[float, float, float]) -> bool:
    return math.isfinite(circle[2]) and circle[2] > 0


def _support(raw: tuple[Observation, ...], i: int, cfg: SequenceConfig) -> int:
    """How many immediate neighbours agree with frame ``i`` (centre and radius gates)."""
    point = (TrackPoint(i, raw[i].rc, raw[i].cc, raw[i].radius),)
    return sum(1 for j in (i - 1, i + 1)
               if 0 <= j < len(raw) and within(raw[j], predict(point, j, (), (), 0, cfg)))


def choose_seed(raw: tuple[Observation, ...], lo: int, hi: int, cfg: SequenceConfig) -> int | None:
    """Most confident ok frame in ``lo..hi-1``, preferring one with temporal support
    (a neighbour inside the gate); NaN confidence ranks as 0, ties go to the earlier frame."""
    best, best_key = None, None
    for i in range(lo, hi):
        obs = raw[i]
        if not obs.ok:
            continue
        sup = _support(raw, i, cfg)
        conf = obs.confidence if math.isfinite(obs.confidence) else 0.0
        key = (sup > 0, conf, sup, -i)
        if best_key is None or key > best_key:
            best, best_key = i, key
    return best


def _seeds(raw: tuple[Observation, ...], anchors: dict, cfg: SequenceConfig) -> tuple[dict[int, TrackPoint], set[int]]:
    """Anchor seeds, plus an automatic seed in every stretch no anchor's passes reach
    (stretches are bounded by anchors; an anchor without a circle stops passes)."""
    seeds = {i: TrackPoint(i, c[0], c[1], c[2]) for i, c in sorted(anchors.items()) if _has_circle(c)}
    bounds = [-1, *sorted(anchors), len(raw)]
    auto = set()
    for a, b in zip(bounds, bounds[1:]):
        if b - a > 1 and a not in seeds and b not in seeds:
            s = choose_seed(raw, a + 1, b, cfg)
            if s is not None:
                auto.add(s)
                seeds[s] = TrackPoint(s, raw[s].rc, raw[s].cc, raw[s].radius)
    return seeds, auto


def _passes(raw, seeds: dict[int, TrackPoint], hard: list[int], cache, cfg) -> list[tuple[int, dict]]:
    out = []
    for s, point in seeds.items():
        nxt = min((h for h in hard if h > s), default=len(raw))
        prv = max((h for h in hard if h < s), default=-1)
        out.append((s, run_pass(raw, point, +1, nxt, cache, cfg)))
        out.append((s, run_pass(raw, point, -1, prv, cache, cfg)))
    return out


def _coverage(n: int, passes: list[tuple[int, dict]]) -> list[list[tuple[int, int, Step]]]:
    """Per frame: (distance to the seed, pass index, step) of every pass that reached it."""
    cover: list[list[tuple[int, int, Step]]] = [[] for _ in range(n)]
    for k, (seed, steps) in enumerate(passes):
        for i, step in steps.items():
            cover[i].append((abs(i - seed), k, step))
    return cover


def _merge(cands: list[tuple[int, int, Step]], cfg: SequenceConfig) -> tuple[Step | None, bool]:
    """(chosen step, passes disagree): the nearest seed's accepted result, else its loss."""
    cands = sorted(cands, key=lambda c: (c[0], c[1]))
    if not cands:
        return None, False
    accepted = [c[2] for c in cands if c[2].accepted]
    if not accepted:
        return cands[0][2], False
    split = len(accepted) > 1 and disagree(accepted[0].obs, accepted[1].obs, cfg)
    return accepted[0], split


def _solution(step: Step | None, raw: Observation, split: bool, cfg: SequenceConfig) -> FrameSolution:
    if step is None or step.source == SOURCE_NONE:
        flags = step.flags if step is not None else frozenset()
        return FrameSolution(_NAN, _NAN, _NAN, 0.0, SOURCE_NONE, flags, -1)
    flags = set(step.flags) | set(step.obs.flags)
    if step.source == SOURCE_LOST:
        r, rc, cc = (raw.radius, raw.rc, raw.cc) if raw.ok else (_NAN, _NAN, _NAN)
        return FrameSolution(r, rc, cc, 0.0, SOURCE_LOST, flags | set(raw.flags), -1)
    obs = step.obs
    if split or (step.source == SOURCE_TRACKED and disagree(obs, raw, cfg)):
        flags.add(FLAG_DISAGREE)
    if obs.confidence < cfg.low_confidence:
        flags.add(FLAG_LOW_CONFIDENCE)
    return FrameSolution(obs.radius, obs.rc, obs.cc, obs.confidence, step.source, flags, -1)


def _anchor_solution(circle: tuple[float, float, float]) -> FrameSolution:
    if _has_circle(circle):
        return FrameSolution(circle[2], circle[0], circle[1], 1.0, SOURCE_ANCHOR, frozenset(), -1)
    return FrameSolution(_NAN, _NAN, _NAN, 1.0, SOURCE_ANCHOR, frozenset(), -1)


def _with_cycles(frames: list[FrameSolution], cfg: SequenceConfig):
    verified = np.array([f.radius if f.source in (SOURCE_RAW, SOURCE_TRACKED, SOURCE_ANCHOR) else _NAN
                         for f in frames])
    collapses, cycles = detect_cycles(verified, config=cfg)
    cycle_of = np.full(len(frames), -1)
    for k, (a, b) in enumerate(cycles):
        cycle_of[a:b + 1] = k
    out = []
    for i, f in enumerate(frames):
        flags = f.flags | {FLAG_COLLAPSE} if i in collapses else f.flags
        out.append(FrameSolution(f.radius, f.rc, f.cc, f.confidence, f.source, flags, int(cycle_of[i])))
    return tuple(out), collapses, cycles


def solve_sequence(raw: Sequence[Observation], redetect: Detector, *,
                   anchors: Mapping[int, tuple[float, float, float]] | None = None,
                   config: SequenceConfig = DEFAULT_SEQUENCE_CONFIG) -> SequenceResult:
    """Track the bubble through a sequence of single-frame results.

    Parameters
    ----------
    raw : sequence of Observation
        One single-frame result per frame (Fit All).  Never modified.
    redetect : Detector
        ``redetect(idx, prior)`` re-measures frame ``idx`` near ``prior = (rc, cc, R)``.
        Called only for frames whose raw result is inconsistent with the track.
    anchors : {frame: (rc, cc, R)}, optional
        Manual or accepted frames: copied unchanged and used as hard seeds.  An
        anchor with R <= 0 or NaN means "no bubble in this frame" and stops passes.
    config : SequenceConfig

    Returns
    -------
    SequenceResult
    """
    frames_in, anchor_map = _validate(raw, anchors)
    n = len(frames_in)
    cache = RedetectCache(redetect)
    seeds, auto = _seeds(frames_in, anchor_map, config)
    cover = _coverage(n, _passes(frames_in, seeds, sorted(anchor_map), cache, config))
    frames: list[FrameSolution] = []
    for i in range(n):
        if i in anchor_map:
            frames.append(_anchor_solution(anchor_map[i]))
        elif i in auto:
            frames.append(_solution(Step(frames_in[i], SOURCE_RAW), frames_in[i], False, config))
        else:
            step, split = _merge(cover[i], config)
            frames.append(_solution(step, frames_in[i], split, config))
    solved, collapses, cycles = _with_cycles(frames, config)
    review = build_review(solved, collapses, config)
    return SequenceResult(frames=solved, collapse_frames=collapses, cycles=cycles, review=review,
                          seed_frame=min(seeds, default=-1), redetect_calls=cache.calls)
