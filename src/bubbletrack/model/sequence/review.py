"""Prioritised review list: the frames a user should look at, likely-wrong first.

Priority of a frame = the weight of its most important reason (plus a small
bonus per extra reason), multiplied by ``review_collapse_weight`` near a
collapse, where fast dynamics make flags expected.  Ties: lower confidence
first, then frame order.  Anchors and ``none`` frames (other than the first
frame of a cloud) are never listed.
"""

from __future__ import annotations

import math

from bubbletrack.model.provenance import AREA_RADIUS_FLAG
from bubbletrack.model.sequence.config import SequenceConfig
from bubbletrack.model.sequence.types import (
    FLAG_CLOUD, FLAG_COLLAPSE, FLAG_DISAGREE, FLAG_JUMP, FLAG_LOST, FLAG_LOW_CONFIDENCE, FLAG_REACQUIRED,
    CONTEXT_FLAGS, RAW_REVIEW_FLAGS, SOURCE_ANCHOR, SOURCE_NONE, FrameSolution,
)

REASON_WEIGHT = {
    FLAG_LOST: 100.0,          # no verified measurement
    FLAG_REACQUIRED: 80.0,     # first frame after a loss or a cloud: may be another object
    FLAG_JUMP: 70.0,           # accepted only by the wide gate
    FLAG_LOW_CONFIDENCE: 60.0,
    FLAG_DISAGREE: 50.0,       # the single-frame result was replaced (or the passes differ)
    "fragmented": 40.0,
    "clipped_not_round": 36.0,
    "non_circular": 35.0,
    "clipped_completed": 30.0,
    FLAG_CLOUD: 20.0,          # where a cloud starts (tracking pauses)
    FLAG_COLLAPSE: 10.0,       # collapse frame or its neighbour
}
_EXTRA_REASON_BONUS = 1.0
_SOLVER_REASONS = (FLAG_LOST, FLAG_REACQUIRED, FLAG_JUMP, FLAG_LOW_CONFIDENCE, FLAG_DISAGREE)


def _reasons(i: int, frame: FrameSolution, prev: FrameSolution | None, near_collapse: bool) -> list[str]:
    if frame.source == SOURCE_ANCHOR:
        return []
    if frame.source == SOURCE_NONE:
        first_cloud = FLAG_CLOUD in frame.flags and (prev is None or FLAG_CLOUD not in prev.flags)
        return [FLAG_CLOUD] if first_cloud else []
    out = [r for r in _SOLVER_REASONS if r in frame.flags]
    out += [r for r in RAW_REVIEW_FLAGS if r in frame.flags
            and not (r == "non_circular" and AREA_RADIUS_FLAG in frame.flags)]
    if near_collapse:
        out.append(FLAG_COLLAPSE)
    return out


def build_review(frames: tuple[FrameSolution, ...], collapses: tuple[int, ...],
                 cfg: SequenceConfig) -> tuple[tuple[int, tuple[str, ...]], ...]:
    """``(frame, reasons)`` pairs sorted by priority (most important first)."""
    near = {c + d for c in collapses for d in range(-cfg.review_collapse_radius, cfg.review_collapse_radius + 1)}
    ranked = []
    for i, frame in enumerate(frames):
        reasons = _reasons(i, frame, frames[i - 1] if i > 0 else None, i in near)
        if CONTEXT_FLAGS.issuperset(reasons):  # nothing (or only routine flags) to review
            continue
        reasons.sort(key=lambda r: -REASON_WEIGHT[r])
        score = REASON_WEIGHT[reasons[0]] + _EXTRA_REASON_BONUS * (len(reasons) - 1)
        if i in near:
            score *= cfg.review_collapse_weight
        conf = frame.confidence if math.isfinite(frame.confidence) else 1.0
        ranked.append((-score, conf, i, tuple(reasons)))
    ranked.sort()
    return tuple((i, reasons) for _, _, i, reasons in ranked)
