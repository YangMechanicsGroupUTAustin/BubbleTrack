"""Constant-velocity prediction and adaptive gates for one tracking pass.

The track keeps the accepted circles of its pass.  The next frame is predicted
from the last one plus the damped centre velocity and radius rate measured
between the last two accepted frames.  The centre gate grows with the measured
speed, the recent prediction errors, the number of lost frames and (on the
first step from a seed, before any rate is known) with the radius, so a fast
drift or growth is followed without widening the gate for a still bubble.

The radius window is the prediction +- the radius gate, with two collapse
rules: a shrinking bubble may also reverse (rebound up to the last radius plus
the measured rate), and when the prediction falls below ``release_ratio`` of
the last radius the window is released to [0, ceiling].
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from bubbletrack.model.sequence.config import SequenceConfig
from bubbletrack.model.sequence.types import Observation


@dataclass(frozen=True)
class TrackPoint:
    frame: int
    rc: float
    cc: float
    radius: float


@dataclass(frozen=True)
class Prediction:
    rc: float
    cc: float
    radius: float
    """Predicted radius (>= 0)."""
    last_radius: float
    centre_gate: float
    centre_floor: float
    """Size-only part of the centre gate; the wide gate adds ``factor - 1`` floors (not the speed terms)."""
    radius_lo: float
    radius_hi: float
    radius_gate: float
    """Half-width of the radius window; the wide gate widens both bounds by ``factor - 1`` of it."""
    released: bool
    """A collapse (or, backwards, nucleation) is under way: ``radius_lo`` is 0."""

    def prior(self, cfg: SequenceConfig) -> tuple[float, float, float]:
        """Circle prior for the detector; released -> the last radius, not the collapsing prediction."""
        r = self.last_radius if self.released else self.radius
        return (self.rc, self.cc, max(r, cfg.min_prior_radius))


def _rates(history: tuple[TrackPoint, ...]) -> tuple[float, float, float] | None:
    """(d rc, d cc, d R) per frame between the last two accepted points (None for a lone seed)."""
    if len(history) < 2:
        return None
    last, prev = history[-1], history[-2]
    span = abs(last.frame - prev.frame)
    return ((last.rc - prev.rc) / span, (last.cc - prev.cc) / span, (last.radius - prev.radius) / span)


def _median(values: tuple[float, ...], window: int) -> float:
    return float(np.median(values[-window:])) if values else 0.0


def predict(history: tuple[TrackPoint, ...], frame: int, centre_errors: tuple[float, ...],
            radius_errors: tuple[float, ...], coast: int, cfg: SequenceConfig) -> Prediction:
    """Prediction and gates for ``frame`` from the accepted ``history`` (latest last).

    ``coast`` = consecutive lost frames just before ``frame``: each widens both gates by
    ``coast_growth`` size floors."""
    last = history[-1]
    dt_e = min(abs(frame - last.frame), cfg.max_extrapolation)
    rates = _rates(history)
    d_rc, d_cc, vr_raw = (0.0, 0.0, 0.0) if rates is None else rates
    vc = (d_rc * cfg.velocity_damping, d_cc * cfg.velocity_damping)
    vr = vr_raw * cfg.velocity_damping
    r_l = last.radius
    grow = 1.0 + cfg.coast_growth * coast
    c_floor = max(cfg.gate_centre_px, cfg.gate_centre_frac * r_l)
    r_floor = max(cfg.gate_radius_px, cfg.gate_radius_frac * r_l)
    unknown = cfg.gate_unknown_frac * r_l * dt_e if rates is None else 0.0
    centre_gate = (c_floor * grow + cfg.gate_speed_k * math.hypot(*vc) * dt_e
                   + cfg.gate_error_k * _median(centre_errors, cfg.error_window) + unknown)
    radius_gate = (r_floor * grow + cfg.gate_rate_k * abs(vr) * dt_e
                   + cfg.gate_error_k * _median(radius_errors, cfg.error_window) + unknown)
    r_pred = max(r_l + vr * dt_e, 0.0)
    rebound = r_l + abs(vr_raw) * dt_e
    released = rates is not None and r_l + vr_raw * dt_e < cfg.release_ratio * r_l
    lo, hi = r_pred - radius_gate, r_pred + radius_gate
    if vr_raw < 0:
        hi = max(hi, rebound + r_floor)
    if released:
        lo, hi = 0.0, rebound + radius_gate
    return Prediction(rc=last.rc + vc[0] * dt_e, cc=last.cc + vc[1] * dt_e, radius=r_pred, last_radius=r_l,
                      centre_gate=centre_gate, centre_floor=c_floor * grow, radius_lo=lo, radius_hi=hi,
                      radius_gate=radius_gate, released=released)


def centre_error(obs: Observation, pred: Prediction) -> float:
    return math.hypot(obs.rc - pred.rc, obs.cc - pred.cc)


def radius_error(obs: Observation, pred: Prediction) -> float:
    return abs(obs.radius - pred.radius)


def centre_within(obs: Observation, pred: Prediction, factor: float = 1.0) -> bool:
    return obs.ok and centre_error(obs, pred) <= pred.centre_gate + (factor - 1.0) * pred.centre_floor


def within(obs: Observation, pred: Prediction, factor: float = 1.0) -> bool:
    """Is ``obs`` inside the gate?  ``factor > 1`` is the wide gate: the centre gate gains
    ``factor - 1`` size floors (its speed terms are not scaled, so a fast bubble does not open
    a huge window onto other objects); the radius window widens by ``factor - 1`` gates."""
    if not centre_within(obs, pred, factor):
        return False
    slack = (factor - 1.0) * pred.radius_gate
    return pred.radius_lo - slack <= obs.radius <= pred.radius_hi + slack


def confident(obs: Observation, threshold: float) -> bool:
    """Confidence passes ``threshold``; an unknown (NaN) confidence never gates."""
    return not (obs.confidence < threshold)


def disagree(a: Observation, b: Observation, cfg: SequenceConfig) -> bool:
    """Do two observations describe different circles (per the ``disagree_*`` tolerances)?"""
    if not (a.ok and b.ok):
        return a.ok != b.ok
    ref = max(a.radius, b.radius)
    if abs(a.radius - b.radius) > max(cfg.disagree_px, cfg.disagree_frac * ref):
        return True
    return math.hypot(a.rc - b.rc, a.cc - b.cc) > max(cfg.disagree_centre_px, cfg.disagree_centre_frac * ref)
