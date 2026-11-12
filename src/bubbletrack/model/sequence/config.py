"""Tunable constants of the sequence solver (all relative to the bubble size where possible).

Defaults follow the validated research prototypes: the constant-velocity
tracker "cv1" (damping 0.8, a centre gate that grows with the measured speed
and the recent prediction error, a 3x wide gate, re-acquisition at confidence
>= 0.5) and the raw-vs-tracked disagreement test (5 %).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields


@dataclass(frozen=True)
class SequenceConfig:
    """All tunables of :func:`bubbletrack.model.sequence.solve_sequence`."""

    # ---- prediction -------------------------------------------------------
    velocity_damping: float = 0.8
    """Centre / radius rates (between the last two accepted frames) are damped by this factor
    (collapses reverse them)."""
    max_extrapolation: int = 3
    """Rates are extrapolated over at most this many frames (constant position beyond)."""
    min_prior_radius: float = 1.0
    """Smallest radius (px) passed to the detector as a prior."""

    # ---- tight gate -------------------------------------------------------
    gate_centre_px: float = 6.0
    gate_centre_frac: float = 0.4
    """Centre gate floor = max(gate_centre_px, gate_centre_frac * R)."""
    gate_speed_k: float = 0.5
    """... + gate_speed_k * |centre velocity| * dt."""
    gate_error_k: float = 2.0
    """... + gate_error_k * median of the recent prediction errors (centre and radius gates)."""
    gate_unknown_frac: float = 0.5
    """... + gate_unknown_frac * R * dt on the first step from a seed (no velocity measured yet)."""
    gate_radius_px: float = 2.0
    gate_radius_frac: float = 0.12
    """Radius gate floor = max(gate_radius_px, gate_radius_frac * R)."""
    gate_rate_k: float = 0.5
    """... + gate_rate_k * |radius rate| * dt."""
    error_window: int = 5
    """Recent prediction errors (accepted frames) used by the gate."""
    coast_growth: float = 0.25
    """Both gates grow by this fraction of their size floor per consecutive lost frame."""
    wide_gate_factor: float = 3.0
    """Wide gate (re-acquisition, sudden jumps): the radius gate times this factor; the centre
    gate plus ``factor - 1`` size floors (speed and error terms are not scaled, so a fast
    bubble does not open a huge window onto other objects)."""

    # ---- acceptance -------------------------------------------------------
    min_confidence: float = 0.15
    """Minimum confidence to accept an observation inside the tight gate."""
    wide_confidence: float = 0.3
    """Minimum confidence to accept an observation inside the wide gate (flag ``jump``)."""
    reacquire_confidence: float = 0.5
    """Minimum confidence of a raw observation accepted by the wide gate (sudden jump, re-acquisition)."""
    refine_below_confidence: float = 0.0
    """Also re-detect a raw observation that passes the tight gate with a confidence below this,
    keeping the more confident of the two.  Off by default: on the LIC sets it costs about
    +50 % redetect calls for one more correct labelled frame in 417."""
    max_redetect_coast: int = 8
    """After this many consecutive lost frames ``redetect`` is no longer called (raw re-acquisition only)."""

    # ---- collapse release -------------------------------------------------
    release_ratio: float = 0.75
    """When the predicted radius drops below this fraction of the last one, a collapse
    (or, backwards, nucleation) is under way: the radius window is released to
    [0, last + |rate| + gate] (minimum and rebound) and the prior keeps the last radius."""
    release_confidence: float = 0.3
    """Minimum confidence to accept an observation while the radius prior is released, or in
    the tight gate right after a loss."""

    # ---- cloud termination ------------------------------------------------
    cloud_radius_frac: float = 0.15
    """After a collapse, radii below this fraction of the pass's Rmax count as cloud-like ..."""
    cloud_frames: int = 3
    """... and this many consecutive cloud-like frames stop the track (frames become ``none``)."""
    low_confidence: float = 0.3
    """Accepted frames below this confidence are flagged ``low_confidence``."""

    # ---- disagreement -----------------------------------------------------
    disagree_frac: float = 0.05
    disagree_px: float = 1.0
    """Radii disagree when they differ by more than max(disagree_px, disagree_frac * R) ..."""
    disagree_centre_frac: float = 0.1
    disagree_centre_px: float = 2.0
    """... or centres by more than max(disagree_centre_px, disagree_centre_frac * R)."""

    # ---- cycles -----------------------------------------------------------
    collapse_rel_prominence: float = 0.3
    """A collapse is a radius minimum at least this fraction below the LOWER neighbouring peak ..."""
    collapse_min_prominence_px: float = 2.0
    """... and at least this many pixels below it ..."""
    collapse_min_width: int = 2
    """... whose valley spans at least this many frames below ``collapse_width_level`` of its depth
    (rejects a one-frame dip: a wrong detection mid-cycle) ..."""
    collapse_width_level: float = 0.75
    collapse_min_gap: int = 3
    """... and at least this many frames from a deeper collapse."""

    # ---- review -----------------------------------------------------------
    review_collapse_radius: int = 1
    """Frames within this distance of a collapse are reported (ranked low)."""
    review_collapse_weight: float = 0.5
    """Review priority of any frame near a collapse is multiplied by this (fast dynamics are expected)."""

    def __post_init__(self) -> None:
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{f.name} must be a finite number, got {value!r}")
            if value < 0:
                raise ValueError(f"{f.name} must be >= 0, got {value!r}")
        for name in ("velocity_damping", "release_ratio", "cloud_radius_frac", "min_confidence",
                     "wide_confidence", "reacquire_confidence", "release_confidence", "low_confidence",
                     "refine_below_confidence",
                     "collapse_rel_prominence", "collapse_width_level", "review_collapse_weight"):
            if getattr(self, name) > 1:
                raise ValueError(f"{name} must be in [0, 1], got {getattr(self, name)!r}")
        for name, low in (("max_extrapolation", 1), ("error_window", 1), ("cloud_frames", 1),
                          ("collapse_min_width", 1), ("collapse_min_gap", 1), ("max_redetect_coast", 0),
                          ("review_collapse_radius", 0)):
            value = getattr(self, name)
            if int(value) != value or value < low:
                raise ValueError(f"{name} must be an integer >= {low}, got {value!r}")
            object.__setattr__(self, name, int(value))
        if self.wide_gate_factor < 1:
            raise ValueError(f"wide_gate_factor must be >= 1, got {self.wide_gate_factor!r}")


DEFAULT_SEQUENCE_CONFIG = SequenceConfig()
