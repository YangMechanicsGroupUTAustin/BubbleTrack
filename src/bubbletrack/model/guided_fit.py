"""A circle the user drew ("Drag a circle"): measure the wall near it, and decide.

The user presses on the bubble centre and drags to its wall.  The circle goes
to the Auto engine as a prior (``EdgeParams(prior=(rc, cc, R), seed=None)``):
it only picks WHICH region is the bubble; the wall itself is measured in the
image.  The result is accepted when it is consistent with the drawn circle
and its confidence reaches a floor lower than the engine's own gate, because
the drag is evidence of its own.

The thresholds come from the prior-drag study (``tasks/studies/
prior_drag_RESULTS.md``, engine edge-1.1.1, labelled LIC / LIC-hard frames):

* the prior is a tie-breaker: with the centre within 0.5 R of the true centre
  and a radius of 0.5-2x, the result equals the exact-circle result on
  98-100 % of the frames;
* accepting a result "consistent with the drag" (centre within 0.5 R_drag of
  the drawn centre AND |R - R_drag| <= 0.5 R_drag) at confidence >= 0.15
  instead of the engine gate 0.3 raises the right-and-accepted rate on LIC
  from 85 -> 91 % while wrong answers stay at 2 %; circles dragged on
  no-bubble frames are accepted 3 % of the time instead of 18 %.

A click without a drag gives no size: the prior radius is then the nearest
fitted frame's (the result is checked against it like a drag), or a small
stand-in the engine copes with, and only the centre is checked (against the
measured radius).  Everything here is Qt-free.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import numpy as np

from bubbletrack.model.edge_pipeline import EdgeFrameFit, EdgeJob, fit_edge

CENTRE_TOLERANCE = 0.5
"""Result centre within this fraction of the drawn radius from the drawn centre."""
RADIUS_TOLERANCE = 0.5
"""|R - R_drawn| within this fraction of the drawn radius."""
MIN_CONFIDENCE = 0.15
"""Confidence floor for a consistent result (the engine's own gate is 0.3)."""
CLICK_FALLBACK_RADIUS_PX = 5.0
"""Prior radius for a click when no frame is fitted yet (edge-1.1.1 handles it)."""


@dataclass(frozen=True)
class UserCircle:
    """A drawn circle in 1-based full-image coordinates (the engine's)."""

    rc: float
    cc: float
    radius: float
    dragged: bool = True  # False: a click; the radius is borrowed
    sized: bool = True  # False: the radius is a stand-in, not checked

    @property
    def prior(self) -> tuple[float, float, float]:
        return (self.rc, self.cc, self.radius)


def drawn_circle(rc: float, cc: float, radius: float) -> UserCircle:
    """A press-drag-release circle."""
    if not (math.isfinite(radius) and radius > 0):
        raise ValueError(f"a drawn circle needs a positive radius, got {radius!r}")
    return UserCircle(float(rc), float(cc), float(radius))


def clicked_circle(rc: float, cc: float, neighbour_radius: float | None) -> UserCircle:
    """A click: the nearest fitted frame's radius, else a small stand-in."""
    if neighbour_radius is not None and neighbour_radius > 0:
        return UserCircle(float(rc), float(cc), float(neighbour_radius), dragged=False)
    return UserCircle(float(rc), float(cc), CLICK_FALLBACK_RADIUS_PX, dragged=False, sized=False)


def nearest_fitted_radius(radius, idx: int) -> float | None:
    """Radius of the fitted frame nearest to *idx* (not *idx* itself); earlier wins a tie."""
    if radius is None:
        return None
    values = np.asarray(radius, dtype=float)
    fitted = np.flatnonzero(np.isfinite(values) & (values > 0))
    fitted = fitted[fitted != idx]
    if fitted.size == 0:
        return None
    nearest = fitted[np.argmin(np.abs(fitted - idx))]  # argmin keeps the first (earlier) tie
    return float(values[nearest])


def is_consistent(circle: UserCircle, rc: float, cc: float, radius: float) -> bool:
    """The measured circle agrees with the user's (see the module docstring)."""
    reference = circle.radius if circle.sized else radius
    if not (np.isfinite(reference) and reference > 0):
        return False
    offset = math.hypot(rc - circle.rc, cc - circle.cc)
    return (offset <= CENTRE_TOLERANCE * reference
            and abs(radius - reference) <= RADIUS_TOLERANCE * reference)


@dataclass(frozen=True)
class GuidedVerdict:
    accepted: bool
    reason: str = ""  # why not, for the user ("" when accepted)


def judge(circle: UserCircle, fit: EdgeFrameFit) -> GuidedVerdict:
    """Accept *fit* for *circle*?  *fit* already holds the confidence floor."""
    if fit.radius <= 0:
        return GuidedVerdict(False, f"no wall found near your circle: {fit.rejected_reason}")
    if not is_consistent(circle, fit.rc, fit.cc, fit.radius):
        return GuidedVerdict(False, f"the wall found (R = {fit.radius:.1f} px) "
                                    "does not match your circle")
    return GuidedVerdict(True)


def fit_guided(raw: np.ndarray, gridx: tuple[int, int], gridy: tuple[int, int],
               job: EdgeJob, circle: UserCircle) -> tuple[EdgeFrameFit, GuidedVerdict]:
    """Measure the wall near *circle* (prior, no seed click) and judge it.

    As the tracker's re-detection does: the circle replaces the seed click,
    which would otherwise choose the region and make the prior be ignored.
    """
    params = dataclasses.replace(job.params, prior=circle.prior, seed=None)
    fit = fit_edge(raw, gridx, gridy, dataclasses.replace(job, params=params),
                   min_confidence=MIN_CONFIDENCE)
    return fit, judge(circle, fit)
