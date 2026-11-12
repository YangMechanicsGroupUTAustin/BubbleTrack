"""Re-measure a hard case's frames with the current engine and the case's parameters.

This is Fit All on the bundle: the per-frame pipeline of the case's detection
mode (``frame_pipeline`` for the threshold-based modes, ``edge_pipeline`` with
the case's reference frame for Auto mode) and, when the user had it on,
sequence tracking over the stored frames.  No user input other than the
parameters is used: labels and manual fits never act as anchors.

The frames are ROI crops, so the ROI becomes the whole crop and results are
shifted back into original-frame coordinates by the ROI offset.  Auto mode
works on the ROI only, so it sees what the app saw.  The threshold-based
modes normalise with whole-frame statistics (grey range, threshold block
size, CLAHE tiles), so a crop can come out slightly differently; the report
counts how many of the app's automatic results a replay reproduces.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np

from bubbletrack.model import edge_pipeline
from bubbletrack.model.cases.schema import CaseFormatError, HardCase
from bubbletrack.model.config import validate_config
from bubbletrack.model.constants import DEFAULT_REMOVING_FACTOR
from bubbletrack.model.frame_pipeline import FrameParams, fit_rois
from bubbletrack.model.image_io import normalize_frame
from bubbletrack.model.removing_factor import compute_removing_factor

ReadFrame = Callable[[int], np.ndarray]


@dataclass(frozen=True)
class Replayed:
    """The current engine's verdict on one frame (original-frame coordinates)."""

    radius: float | None  # None: no result (rejected / nothing plausible)
    center: tuple[float, float] | None = None
    confidence: float | None = None
    tracked: bool = False  # re-measured by sequence tracking
    reason: str = ""  # why there is no result

    @property
    def accepted(self) -> bool:
        return self.radius is not None


def _clean_params(params: Mapping) -> dict:
    """The case's parameters, validated like config.json (bundles travel)."""
    clean = validate_config(dict(params))
    obj = params.get("removing_obj_radius", 0)
    clean["removing_obj_radius"] = obj if isinstance(obj, int) and not isinstance(obj, bool) \
        and 0 <= obj <= 100 else 0
    seed = params.get("edge_seed")
    ok = (isinstance(seed, (list, tuple)) and len(seed) == 2
          and all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                  for v in seed))
    clean["edge_seed"] = (float(seed[0]), float(seed[1])) if ok else None
    return clean


def _finite(value) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def _legacy_params(p: dict, mode: str, gridx, gridy, max_radius: float) -> FrameParams:
    return FrameParams(
        sensitivity=p.get("img_thr", 0.5), gridx=gridx, gridy=gridy,
        removing_factor=compute_removing_factor(
            p.get("removing_factor", DEFAULT_REMOVING_FACTOR), gridx, gridy),
        edges=tuple(p.get("bubble_cross_edges", (False, False, False, False))),
        mode=mode, removing_obj_radius=p["removing_obj_radius"],
        gaussian_sigma=p.get("gaussian_sigma", 0.0), clahe_clip=p.get("clahe_clip", 0.0),
        closing_radius=p.get("closing_radius", 0), opening_radius=p.get("opening_radius", 0),
        gradient_boundary=p.get("gradient_boundary", 0.0),
        area_max_regions=p.get("area_max_regions", 1), max_radius=max_radius,
    )


def _fit_legacy(raw: np.ndarray, params: FrameParams) -> Replayed:
    _, _, gray_roi, binary_roi = normalize_frame(
        raw, params.sensitivity, params.gridx, params.gridy,
        gaussian_sigma=params.gaussian_sigma, clahe_clip=params.clahe_clip)
    fit = fit_rois(binary_roi, gray_roi, params)
    if not fit.fitted:
        return Replayed(None, reason="no plausible bubble")
    return Replayed(fit.radius, (fit.center_row, fit.center_col))


def _from_edge(fit: edge_pipeline.EdgeFrameFit, tracked: bool = False) -> Replayed:
    confidence = _finite(fit.confidence)
    if fit.radius <= 0:
        return Replayed(None, confidence=confidence, reason=fit.rejected_reason)
    return Replayed(float(fit.radius), (float(fit.rc), float(fit.cc)), confidence, tracked)


def _edge_job(case: HardCase, p: dict, read_frame: ReadFrame) -> edge_pipeline.EdgeJob:
    ref = case.detection.reference_index
    if ref is not None and case.frame(ref) is None:
        raise CaseFormatError(f"reference frame {ref + 1} is not in the bundle")
    reference = read_frame(ref) if ref is not None else None
    seed = p["edge_seed"]
    if seed is not None:
        dr, dc = case.offset
        seed = (seed[0] - dr, seed[1] - dc)
    params = edge_pipeline.make_params(edge_definition=p.get("edge_definition", "steep"),
                                       polarity=p.get("edge_polarity", "dark"), seed=seed)
    return edge_pipeline.EdgeJob(params=params, reference=reference,
                                 max_radius=case.detection.max_radius)


def _track(order: list[int], fits: dict, images: Mapping[int, np.ndarray], gridx, gridy,
           job: edge_pipeline.EdgeJob) -> dict[int, Replayed]:
    """Sequence tracking over the stored frames, as after Fit All (no anchors)."""
    from bubbletrack.model.sequence import SOURCE_TRACKED, Observation, solve_sequence
    from bubbletrack.model.sequence_pipeline import Redetector

    nan = float("nan")
    raw = [Observation(f.radius, f.rc, f.cc, f.confidence, f.flags) if f.radius > 0
           else Observation(nan, nan, nan, f.confidence, f.flags)
           for f in (fits[i] for i in order)]
    redetector = Redetector(lambda k: images[order[k]], gridx, gridy, job)
    result = solve_sequence(raw, redetector)
    out = {i: _from_edge(fits[i]) for i in order}
    for k, solution in enumerate(result.frames):
        if solution.source == SOURCE_TRACKED:
            fit = redetector.fit_for(k, solution)
            if fit is not None:
                out[order[k]] = _from_edge(fit, tracked=True)
    return out


def _shift(result: Replayed, offset: tuple[int, int]) -> Replayed:
    if result.center is None:
        return result
    return dataclasses.replace(result, center=(result.center[0] + offset[0],
                                               result.center[1] + offset[1]))


def _crop_bounds(images: Mapping[int, np.ndarray]) -> tuple[tuple[int, int], tuple[int, int]]:
    shapes = {img.shape[:2] for img in images.values()}
    if len(shapes) != 1:
        raise CaseFormatError(f"the stored frames differ in size: {sorted(shapes)}")
    h, w = shapes.pop()
    return (1, h), (1, w)


def replay_case(case: HardCase, read_frame: ReadFrame, *,
                tracking: bool | None = None) -> dict[int, Replayed]:
    """The current engine's result for every selected and context frame.

    Parameters
    ----------
    case : HardCase
    read_frame : callable
        ``read_frame(index)`` returns the stored crop of frame *index*.
    tracking : bool, optional
        Override whether Auto mode tracks across frames (default: as the
        user had it when exporting).

    Returns
    -------
    dict
        Frame index -> :class:`Replayed`, in original-frame coordinates.
    """
    order = [f.index for f in case.measured_frames()]
    images = {i: read_frame(i) for i in order}
    if not images:
        return {}
    gridx, gridy = _crop_bounds(images)
    p = _clean_params(case.detection.params)
    mode = case.detection.mode
    if mode != "edge":
        params = _legacy_params(p, mode, gridx, gridy, case.detection.max_radius)
        results = {i: _fit_legacy(images[i], params) for i in order}
    else:
        job = _edge_job(case, p, read_frame)
        fits = {i: edge_pipeline.fit_edge(images[i], gridx, gridy, job) for i in order}
        track = case.detection.tracking if tracking is None else tracking
        results = (_track(order, fits, images, gridx, gridy, job) if track and len(order) > 1
                   else {i: _from_edge(fits[i]) for i in order})
    return {i: _shift(r, case.offset) for i, r in results.items()}
