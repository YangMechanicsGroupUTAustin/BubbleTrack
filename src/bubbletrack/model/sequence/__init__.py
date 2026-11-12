"""Sequence solver: use neighbouring frames to track, repair and review per-frame results.

After Fit All, every frame has a single-frame :class:`Observation`.  The solver
tracks the bubble forwards and backwards from a seed with a constant-velocity
prediction and adaptive gates, re-runs the detector with a circle prior only
on frames whose raw result is inconsistent with the track, finds collapses
and cycles, stops at a post-collapse fragment cloud, and returns a prioritised
review list.  Manual or accepted frames are hard anchors.

Public API::

    from bubbletrack.model.sequence import Observation, SequenceConfig, solve_sequence

    raw = [Observation(r, rc, cc, conf, flags) for ...]          # one per frame
    res = solve_sequence(raw, redetect, anchors={12: (rc, cc, R)})
    res.frames[i].radius, res.frames[i].source, res.review[:10]

Pure Python / numpy: no Qt imports (model layer).
"""

from bubbletrack.model.sequence.config import DEFAULT_SEQUENCE_CONFIG, SequenceConfig
from bubbletrack.model.sequence.cycles import detect_cycles
from bubbletrack.model.sequence.solver import solve_sequence
from bubbletrack.model.sequence.types import (
    CONTEXT_FLAGS, FLAG_CLOUD, FLAG_COLLAPSE, FLAG_DISAGREE, FLAG_JUMP, FLAG_LOST, FLAG_LOW_CONFIDENCE,
    FLAG_REACQUIRED, RAW_REVIEW_FLAGS, SOURCE_ANCHOR, SOURCE_LOST, SOURCE_NONE, SOURCE_RAW, SOURCE_TRACKED, SOURCES,
    Detector, FrameSolution, Observation, SequenceResult,
)

__all__ = [
    "CONTEXT_FLAGS", "DEFAULT_SEQUENCE_CONFIG", "Detector", "FLAG_CLOUD", "FLAG_COLLAPSE", "FLAG_DISAGREE", "FLAG_JUMP",
    "FLAG_LOST", "FLAG_LOW_CONFIDENCE", "FLAG_REACQUIRED", "FrameSolution", "Observation", "RAW_REVIEW_FLAGS",
    "SOURCES", "SOURCE_ANCHOR", "SOURCE_LOST", "SOURCE_NONE", "SOURCE_RAW", "SOURCE_TRACKED", "SequenceConfig",
    "SequenceResult", "detect_cycles", "solve_sequence",
]
