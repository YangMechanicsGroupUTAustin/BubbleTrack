"""ML assist on one frame, as the app stores it (Qt-free, picklable).

The bridge between ``model.teach`` (``train_model``, ``measure_frame``) and the
app's result arrays, shared by the teaching preview (a thread) and Fit All /
Apply (a process pool whose initializer retrains the model from the examples:
training is deterministic and takes well under a second, so no model is
pickled).

Per frame (guarded policy, see ``model.teach.measure``):

* ``auto_ml`` (Auto's wall inside the ML region) and ``auto`` (plain Auto's own
  result kept) are stored as Auto results (``FitSource.AUTO``; flag
  ``ml_guided`` when guided, ``ml_disagrees`` when ML saw the bubble elsewhere);
* ``ml`` (ML's own outline) is stored as ``FitSource.ML`` with flag
  ``ml_outline`` and ML's confidence: always listed in Review;
* otherwise no result, stored exactly like plain Auto's rejection (its
  confidence, flags, metrics and guess, the dashed suggestion), so a later
  Apply can hand it back as Auto's.  An ML result clears the guess.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from bubbletrack.model import edge_pipeline
from bubbletrack.model.edge_pipeline import EdgeFrameFit, EdgeJob
from bubbletrack.model.provenance import FitSource

FLAG_GUIDED, FLAG_OUTLINE = "ml_guided", "ml_outline"
FLAG_DISAGREES, FLAG_UNFAMILIAR = "ml_disagrees", "ml_unfamiliar"
ML_FLAGS = frozenset({FLAG_GUIDED, FLAG_OUTLINE, FLAG_DISAGREES, FLAG_UNFAMILIAR})
RESULT_FLAGS = frozenset({FLAG_GUIDED, FLAG_OUTLINE})
"""Flags of a result that is not plain Auto's own."""
WORKER_THREADS = 1  # model threads inside a pool worker (results do not depend on it)
_NAN = float("nan")
_EMPTY_XY = np.empty((0, 2))


@dataclass(frozen=True, eq=False)
class MLJob:
    """Everything a worker needs to train the model (plain, picklable data)."""

    examples: dict
    """``MLExamples.to_dict()``."""
    frames: dict
    """{frame: raw pixels} of every frame with examples."""
    gridx: tuple[int, int]
    gridy: tuple[int, int]
    reference: np.ndarray | None
    n_jobs: int = WORKER_THREADS


def train_job(job: MLJob):
    """The ``MLModel`` of *job* (deterministic: every worker trains the same model)."""
    from bubbletrack.model.teach import ForestConfig, MLConfig, MLExamples, train_model

    config = MLConfig(forest=replace(ForestConfig(), n_jobs=max(1, int(job.n_jobs))))
    return train_model(MLExamples.from_dict(job.examples), job.frames.__getitem__,
                       job.gridx, job.gridy, reference=job.reference, config=config)


_WORKER: dict = {"model": None, "error": "not initialised"}


def init_worker(job: MLJob) -> None:
    """Process-pool initializer: train this worker's model once.

    A failure is kept and raised by every frame (a raising initializer would
    break the whole pool instead of reporting it).
    """
    try:
        _WORKER.update(model=train_job(job), error="")
    except Exception as exc:  # noqa: BLE001 - reported per frame
        _WORKER.update(model=None, error=f"{type(exc).__name__}: {exc}")


def worker_model():
    if _WORKER["model"] is None:
        raise RuntimeError(f"ML assist model unavailable ({_WORKER['error']})")
    return _WORKER["model"]


@dataclass(frozen=True, eq=False)
class StoredAuto:
    """A frame's stored plain-Auto result (the Fit All result), for ``auto=``.

    ``radius`` is NaN for a frame Auto rejected (its confidence is kept).
    """

    radius: float
    rc: float
    cc: float
    confidence: float
    flags: frozenset
    edge_xy: np.ndarray | None = None
    metrics: dict = field(default_factory=dict)
    suggestion: object = None
    """Auto's guess for a rejected frame (``model.suggestion.Suggestion``), if any."""

    @property
    def fitted(self) -> bool:
        return bool(np.isfinite(self.radius) and self.radius > 0)

    def as_fit(self) -> EdgeFrameFit:
        if not self.fitted:
            return EdgeFrameFit(-1.0, _NAN, _NAN, _EMPTY_XY, None, self.confidence, self.flags,
                                "no bubble found", metrics=dict(self.metrics),
                                suggestion=self.suggestion)
        xy = _EMPTY_XY if self.edge_xy is None else np.asarray(self.edge_xy, float).reshape(-1, 2)
        return EdgeFrameFit(self.radius, self.rc, self.cc, xy, None, self.confidence, self.flags, "",
                            metrics=dict(self.metrics))

    def as_result(self, gate: float):
        """An ``EdgeResult`` carrying what ML assist reads from plain Auto."""
        from bubbletrack.model.edge import EdgeResult

        r = self.radius if self.fitted else _NAN
        return EdgeResult(radius=r, rc=self.rc, cc=self.cc, radius_sigma=_NAN, edge_xy=_EMPTY_XY,
                          confidence=self.confidence, flags=self.flags, r_area=r, outer_radius=_NAN,
                          shoulder_width=_NAN, arc_coverage=_NAN, inlier_fraction=_NAN,
                          mask=np.zeros((0, 0), bool), confidence_gate=gate)


@dataclass(frozen=True, eq=False)
class MLFrameFit:
    """What is stored for one frame (``fit.radius`` -1: no result)."""

    fit: EdgeFrameFit
    source: FitSource
    ml_source: str
    """"auto_ml", "auto", "ml" or "none"."""
    novelty: float = _NAN
    result: object = None
    """The ``MLResult`` (previews only; not sent between processes)."""


def measure_ml(model, raw: np.ndarray, gridx: tuple[int, int], gridy: tuple[int, int], job: EdgeJob,
               stored: StoredAuto | None = None) -> MLFrameFit:
    """Measure one frame with ML assist and turn the outcome into a storable fit.

    *job* is the run's Auto job (settings, reference, plausible radius); the
    model must have been trained with the same ROI and reference.  *stored*
    is the frame's plain-Auto result when there is one; else Auto runs here.
    """
    from bubbletrack.model.teach import measure_frame

    if stored is None:
        res = edge_pipeline.detect(raw, gridx, gridy, job)
        plain = edge_pipeline.fit_from_result(res, job)
        auto = res if plain.radius > 0 else replace(res, radius=_NAN)
    else:
        plain, auto = stored.as_fit(), stored.as_result(job.params.confidence_gate)
    return storable(measure_frame(model, raw, params=job.params, auto=auto), plain, job)


def storable(out, plain: EdgeFrameFit, job: EdgeJob) -> MLFrameFit:
    """The fit to store for ML assist's outcome *out* (``MLResult``) and plain Auto's *plain*."""
    if out.accepted and out.source == "auto_ml":
        fit = edge_pipeline.fit_from_result(out.guided, job)
        if fit.radius > 0:
            return MLFrameFit(replace(fit, flags=frozenset(out.flags)), FitSource.AUTO, "auto_ml",
                              out.novelty, out)
    if out.accepted and out.source == "ml" and 0 < out.radius <= job.max_radius:
        return MLFrameFit(_outline_fit(out), FitSource.ML, "ml", out.novelty, out)
    unfamiliar = frozenset(out.flags) & {FLAG_UNFAMILIAR}
    if plain.radius > 0:
        flags = plain.flags | (frozenset(out.flags) & (ML_FLAGS - RESULT_FLAGS))
        return MLFrameFit(replace(plain, flags=flags), FitSource.AUTO, "auto", out.novelty, out)
    none = replace(plain, flags=plain.flags | unfamiliar,
                   rejected_reason=plain.rejected_reason or "no bubble found")
    return MLFrameFit(none, FitSource.NONE, "none", out.novelty, out)


def _outline_fit(out) -> EdgeFrameFit:
    region = out.region
    terms = region.terms or {}
    metrics = {"area_px": float(region.area)}
    metrics.update({k: float(terms[k]) for k in ("solidity", "circularity") if k in terms})
    contour = np.asarray(region.contour, dtype=np.float64).reshape(-1, 2)
    return EdgeFrameFit(float(out.radius), float(out.rc), float(out.cc), contour, region.mask,
                        float(out.confidence), frozenset(out.flags), "", metrics=metrics)


def process_frame(args: tuple):
    """Pool task (module-level, picklable): one frame with the worker's model.

    Returns the 8 fields of the Edge batch path; ``meta["source"]`` is the
    ``FitSource`` code to store.
    """
    idx, source, gridx, gridy, job, stored = args
    fit = measure_ml(worker_model(), edge_pipeline.read_raw(source), gridx, gridy, job, stored)
    f = fit.fit
    meta = {"confidence": f.confidence, "flags": f.flags, "metrics": f.metrics, "source": int(fit.source),
            "suggestion": f.suggestion if f.radius <= 0 else None}
    return (idx, f.radius, f.edge_xy, f.mask, f.rc, f.cc, 0, meta)
