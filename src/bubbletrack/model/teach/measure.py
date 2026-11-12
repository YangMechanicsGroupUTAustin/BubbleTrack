"""One frame with ML assist: what radius is reported, and where it comes from.

Precedence (``policy="combined"``, the default), per unverified frame:

1. ML proposes the bubble region; Auto measures the 50 % wall with that circle
   as its prior.  If Auto's result is accepted and lies on the ML region, it is
   reported: source ``"auto_ml"`` ("Auto (ML-guided)"), flag ``ml_guided``.
2. Otherwise ML's own outline: source ``"ml"``, flag ``ml_outline``, with ML's
   confidence; always sent to Review.
3. If ML finds no region, no result (source ``"none"``).

Manual and locked frames are never measured here (the caller skips them).
``policy="prior"`` (always Auto's guided wall) and ``"contour"`` (always ML's
outline) exist for the study.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from bubbletrack.model.edge import EdgeParams, EdgeResult, detect_edge
from bubbletrack.model.edge.engine import engine_view
from bubbletrack.model.teach.config import MeasureConfig
from bubbletrack.model.teach.model import MLModel
from bubbletrack.model.teach.region import MLRegion, ml_region

SOURCE_AUTO_ML = "auto_ml"
SOURCE_ML = "ml"
SOURCE_NONE = "none"
FLAG_GUIDED = "ml_guided"
FLAG_OUTLINE = "ml_outline"
FLAG_UNFAMILIAR = "ml_unfamiliar"

_NAN = float("nan")


@dataclass(frozen=True, eq=False)
class MLResult:
    """Outcome of one frame.  Coordinates are 1-based full-image ``(row, col)``."""

    radius: float
    rc: float
    cc: float
    confidence: float
    source: str
    """"auto_ml" (Auto's wall inside the ML region), "ml" (ML's outline) or "none"."""
    flags: frozenset[str]
    region: MLRegion | None
    """ML's own region (None when the classifier found nothing)."""
    guided: EdgeResult | None
    """Auto run with the ML circle as its prior (None when not run)."""
    confidence_gate: float = 0.3
    novelty: float = _NAN
    """Fraction of out-of-range features (see :meth:`MLModel.predict`)."""

    @property
    def accepted(self) -> bool:
        return bool(self.source != SOURCE_NONE and np.isfinite(self.radius) and self.radius > 0
                    and self.confidence >= self.confidence_gate)

    @property
    def needs_review(self) -> bool:
        """ML's own outline is always reviewed, and so is anything not accepted."""
        return self.source != SOURCE_AUTO_ML or not self.accepted


def wall_agrees(region: MLRegion, edge: EdgeResult, config: MeasureConfig = MeasureConfig()) -> bool:
    """Auto's circle lies on the ML region (same object, similar size)."""
    if not (np.isfinite(edge.radius) and edge.radius > 0):
        return False
    r = max(region.radius, 1.0)
    dr = abs(edge.radius - region.radius)
    dc = float(np.hypot(edge.rc - region.rc, edge.cc - region.cc))
    return (dr <= max(config.agree_radius[0] * r, config.agree_radius[1])
            and dc <= max(config.agree_centre[0] * r, config.agree_centre[1]))


def _none(region: MLRegion | None, guided: EdgeResult | None, gate: float) -> MLResult:
    return MLResult(_NAN, _NAN, _NAN, 0.0, SOURCE_NONE, frozenset(), region, guided, gate)


def _from_auto(edge: EdgeResult, region: MLRegion | None, flags: frozenset[str]) -> MLResult:
    return MLResult(edge.radius, edge.rc, edge.cc, edge.confidence, SOURCE_AUTO_ML,
                    frozenset(edge.flags) | flags | {FLAG_GUIDED}, region, edge, edge.confidence_gate)


def _from_region(region: MLRegion, guided: EdgeResult | None, gate: float) -> MLResult:
    flags = {FLAG_OUTLINE} | ({"clipped"} if region.clipped else set())
    return MLResult(region.radius, region.rc, region.cc, region.confidence, SOURCE_ML, frozenset(flags),
                    region, guided, gate)


def select_result(region: MLRegion | None, guided: EdgeResult | None, *,
                  config: MeasureConfig = MeasureConfig()) -> MLResult:
    """Apply the measurement policy to ML's region and Auto's guided measurement."""
    gate = config.confidence_gate
    if config.policy == "prior":
        return _none(region, guided, gate) if guided is None else _from_auto(guided, region, frozenset())
    if region is None:
        return _none(region, guided, gate)
    if config.policy == "combined" and guided is not None and guided.accepted \
            and wall_agrees(region, guided, config):
        return _from_auto(guided, region, frozenset())
    return _from_region(region, guided, gate)


def measure_frame(model: MLModel, raw: np.ndarray, *, params: EdgeParams = EdgeParams(),
                  config: MeasureConfig | None = None) -> MLResult:
    """Measure one frame with ML assist.

    *params* are the user's Auto settings (seed click, polarity, gate; a
    tracker prior if any): they pick ML's region and are passed to Auto, whose
    prior becomes the ML circle.
    """
    cfg = config or model.config.measure
    pred = model.predict(raw)
    gx, gy = model.gridx, model.gridy
    view = engine_view(raw, gx, gy, reference=model.reference, params=params) if cfg.refine else None
    region = ml_region(pred.probability, view, origin=(gx[0], gy[0]), seed=params.seed, prior=params.prior,
                       config=cfg)
    guided = None
    if region is not None and cfg.policy != "contour":
        guided = detect_edge(raw, gx, gy, reference=model.reference,
                             params=replace(params, prior=(region.rc, region.cc, region.radius)))
    res = select_result(region, guided, config=cfg)
    flags = res.flags | ({FLAG_UNFAMILIAR} if pred.novelty > cfg.novelty_warn else set())
    return replace(res, flags=flags, novelty=pred.novelty)
