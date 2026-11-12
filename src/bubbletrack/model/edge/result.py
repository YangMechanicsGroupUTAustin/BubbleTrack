"""Public value types of the Edge engine: ``EdgeParams`` (input) and ``EdgeResult`` (output)."""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass, field

import numpy as np

_DEFINITIONS = ("steep", "outer")
_POLARITIES = ("dark", "bright")
_EDGE_CHOICES = ("outer", "inner")
_NAN = float("nan")


def _is_real(x) -> bool:
    return isinstance(x, numbers.Real) and not isinstance(x, bool)


def _finite_tuple(value, n: int, name: str) -> tuple[float, ...]:
    try:
        out = tuple(float(v) for v in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a sequence of {n} numbers, got {value!r}") from exc
    if len(out) != n or not all(math.isfinite(v) for v in out):
        raise ValueError(f"{name} must be {n} finite numbers, got {value!r}")
    return out


@dataclass(frozen=True)
class EdgeParams:
    """User-facing settings of one measurement (validated on construction)."""

    edge_definition: str = "steep"
    """"steep" = A, the 50 % crossing of the steep edge (default); "outer" = B, the outer extent."""
    polarity: str = "dark"
    """"dark" bubble on a bright background, or "bright" bubble on a dark one."""
    edge_choice: str = "outer"
    """Which ring when several edges exist along a ray: "outer" or "inner"."""
    level: float = 0.5
    """Crossing level between the local plateaus; keep 0.5 (it defines the edge)."""
    seed: tuple[float, float] | None = None
    """(row, col) click on the bubble, 1-based full-image coordinates."""
    prior: tuple[float, float, float] | None = None
    """(rc, cc, R) expected circle (e.g. from a neighbouring frame), same coordinates."""
    confidence_gate: float = 0.3

    def __post_init__(self) -> None:
        if self.edge_definition not in _DEFINITIONS:
            raise ValueError(f"edge_definition must be one of {_DEFINITIONS}, got {self.edge_definition!r}")
        if self.polarity not in _POLARITIES:
            raise ValueError(f"polarity must be one of {_POLARITIES}, got {self.polarity!r}")
        if self.edge_choice not in _EDGE_CHOICES:
            raise ValueError(f"edge_choice must be one of {_EDGE_CHOICES}, got {self.edge_choice!r}")
        if not (_is_real(self.level) and 0.0 < float(self.level) < 1.0):
            raise ValueError(f"level must be in (0, 1), got {self.level!r}")
        if not (_is_real(self.confidence_gate) and 0.0 <= float(self.confidence_gate) <= 1.0):
            raise ValueError(f"confidence_gate must be in [0, 1], got {self.confidence_gate!r}")
        if self.seed is not None:
            object.__setattr__(self, "seed", _finite_tuple(self.seed, 2, "seed"))
        if self.prior is not None:
            prior = _finite_tuple(self.prior, 3, "prior")
            if prior[2] <= 0:
                raise ValueError(f"prior radius must be positive, got {prior[2]}")
            object.__setattr__(self, "prior", prior)


def _empty_points() -> np.ndarray:
    return np.empty((0, 2))


@dataclass(frozen=True, eq=False)
class EdgeResult:
    """Measurement of one frame.  Coordinates are 1-based full-image ``[row, col]``.

    The bubble is the region enclosed by its outer iso-contour at the edge level,
    whatever its shape; ``radius`` is its area-equivalent radius ``sqrt(A / pi)``.
    """

    radius: float
    """Area-equivalent radius of the bubble region (per ``edge_definition``); NaN if nothing was found.
    A region clipped by the ROI border or excluded pixels is completed with the circle fit
    (``radius_kind == "area+circle_completion"``).  A frame below the confidence gate reports
    the circle fit instead (``radius_kind == "circle"``, flag ``circle_fallback``)."""
    rc: float
    """Centre row: the area centroid, or the circle-fit centre when the region is clipped or untrusted."""
    cc: float
    radius_sigma: float
    """Standard error of ``radius`` (outline noise plus the extrapolated part)."""
    edge_xy: np.ndarray
    """(M, 2) points of the bubble outline on a genuine edge (ROI-border closure left out)."""
    confidence: float
    flags: frozenset[str]
    r_area: float
    """Area-equivalent radius of the region (equals ``radius`` unless ``radius_kind == "circle"``)."""
    outer_radius: float
    """Definition-B radius (NaN if undefined): the outer region's area-equivalent radius with
    ``edge_definition == "outer"``, else the circle fit of the outer-extent points."""
    shoulder_width: float
    """B - A (NaN if undefined): area radii with ``edge_definition == "outer"``, else circle fits."""
    arc_coverage: float
    """Angular coverage of the circle fit's inlier edge points."""
    inlier_fraction: float
    """Fraction of the ray edge points that are circle-fit inliers."""
    mask: np.ndarray
    """The filled bubble region of the ROI (bool), for the binary display panel."""
    provenance: dict = field(default_factory=dict)
    confidence_gate: float = 0.3
    r_circle: float = _NAN
    """Radius of the robust circle fit to the ray edge points."""
    area_px: float = _NAN
    """Region area (px^2), hidden part included."""
    perimeter_px: float = _NAN
    circularity: float = _NAN
    """4 pi A / P^2 (1 for a circle)."""
    solidity: float = _NAN
    """A / convex-hull area (1 for a convex region)."""
    fragment_count: int = 0
    """Other dark regions near the main body (within ~2 R), or lobes split off it."""
    fragment_area_px: float = 0.0
    contour: np.ndarray = field(default_factory=_empty_points)
    """(N, 2) closed outline of the visible region, ROI-border closure included."""
    radius_kind: str = "area"
    """"area", "area+circle_completion" or "circle" (fallback below the confidence gate)."""
    centroid: tuple[float, float] = (_NAN, _NAN)
    """Area centroid of the full (completed) region."""

    @property
    def accepted(self) -> bool:
        """True when a positive radius was found and its confidence passes the gate."""
        return bool(np.isfinite(self.radius) and self.radius > 0 and self.confidence >= self.confidence_gate)
