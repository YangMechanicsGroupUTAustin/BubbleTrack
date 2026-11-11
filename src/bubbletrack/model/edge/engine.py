"""The Edge engine: single-frame bubble edge measurement (``detect_edge``).

Pipeline (ROI-local, 0-based pixel coordinates inside; 1-based full-image
coordinates in and out, i.e. pixel ``(i, j)`` has coordinate ``(i+1, j+1)``):

A  normalise by the reference frame (gain + residual surface) or, without one,
   by a robust background surface;  B  coarse mask at 50 % contrast;
E  candidate selection;  C  sub-pixel edges along ~2*pi*R rays;
D  robust circle fit;  F  outer extent, confidence, flags and provenance.
"""

from __future__ import annotations

import math
import numbers
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from bubbletrack.model.conventions import clamp_roi, roi_to_slice
from bubbletrack.model.edge.config import DEFAULT_CONFIG, EdgeConfig
from bubbletrack.model.edge.fit import CircleFit, fit_circle
from bubbletrack.model.edge.normalize import GrayFrame, Normalized, normalize_roi, saturation_mask, to_gray_frame
from bubbletrack.model.edge.quality import FitStats, confidence, confidence_terms, fit_stats, measurement_flags
from bubbletrack.model.edge.rays import RayEdges, edge_image, outer_extent, ray_edges, refine_window
from bubbletrack.model.edge.segment import CoarseMask, Selection, coarse_segment, select_candidate

ENGINE_VERSION = "edge-1.0.0"

_DEFINITIONS = ("steep", "outer")
_POLARITIES = ("dark", "bright")
_EDGE_CHOICES = ("outer", "inner")


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


@dataclass(frozen=True, eq=False)
class EdgeResult:
    """Measurement of one frame.  Coordinates are 1-based full-image ``[row, col]``."""

    radius: float
    """Radius per ``edge_definition``; NaN if nothing was found."""
    rc: float
    cc: float
    radius_sigma: float
    edge_xy: np.ndarray
    """(M, 2) inlier edge points of the reported circle."""
    confidence: float
    flags: frozenset[str]
    r_area: float
    """Area-equivalent radius of the reported edge; NaN unless the arc is complete."""
    outer_radius: float
    """Definition-B radius (NaN if undefined)."""
    shoulder_width: float
    """outer_radius - steep radius (NaN if undefined)."""
    arc_coverage: float
    inlier_fraction: float
    mask: np.ndarray
    """Coarse dark mask of the ROI (bool), for the binary display panel."""
    provenance: dict = field(default_factory=dict)
    confidence_gate: float = 0.3

    @property
    def accepted(self) -> bool:
        """True when a radius was found and its confidence passes the gate."""
        return bool(np.isfinite(self.radius) and self.confidence >= self.confidence_gate)


class _Stopwatch:
    """Per-stage wall time (ms), written straight into the provenance dict."""

    def __init__(self, laps: dict[str, float]) -> None:
        self.laps = laps
        self._start = self._last = time.perf_counter()

    def lap(self, stage: str) -> None:
        now = time.perf_counter()
        self.laps[stage] = (now - self._last) * 1e3
        self._last = now

    def stop(self) -> None:
        self.laps["total"] = (time.perf_counter() - self._start) * 1e3


def detect_edge(raw: np.ndarray, gridx: tuple[int, int], gridy: tuple[int, int], *,
                reference: np.ndarray | None = None, params: EdgeParams = EdgeParams()) -> EdgeResult:
    """Measure the bubble edge in one frame.

    Parameters
    ----------
    raw : ndarray
        Full frame, any dtype (uint8, uint16, float), 2-D or 3/4-channel.
        It is never modified.
    gridx, gridy : (min, max)
        1-based inclusive ROI row / column bounds (clamped to the image).
    reference : ndarray, optional
        Full bubble-free frame of the same size (see ``choose_reference_frame``).
        None falls back to reference-free surface normalisation (flag ``no_reference``).
    params : EdgeParams

    Returns
    -------
    EdgeResult
        ``radius`` is NaN when nothing was found; ``flags`` then names why.

    Raises
    ------
    ValueError
        For an unsupported frame, a reference of another size, or a malformed ROI.
    """
    return detect_edge_with_config(raw, gridx, gridy, reference=reference, params=params,
                                   config=DEFAULT_CONFIG)


def detect_edge_with_config(raw: np.ndarray, gridx, gridy, *, reference: np.ndarray | None,
                            params: EdgeParams, config: EdgeConfig) -> EdgeResult:
    """:func:`detect_edge` with an explicit :class:`EdgeConfig` (for tuning and tests)."""
    frame = to_gray_frame(raw)
    ref = to_gray_frame(reference) if reference is not None else None
    if ref is not None and ref.values.shape != frame.values.shape:
        raise ValueError(f"reference shape {ref.values.shape} differs from frame shape {frame.values.shape}")
    gx, gy, clamped = _parse_roi(gridx, gridy, frame.values.shape)
    origin = (gx[0], gy[0])                   # 1-based coordinate of ROI pixel (0, 0)
    flags: set[str] = set() if ref is not None else {"no_reference"}
    prov = {"engine": ENGINE_VERSION, "params": asdict(params), "roi": {"gridx": gx, "gridy": gy},
            "fallbacks": [] if ref is not None else ["no_reference: surface normalisation"],
            "timings_ms": {}, "failure": None}
    watch = _Stopwatch(prov["timings_ms"])
    if clamped:
        prov["fallbacks"].append("roi clamped to the image")
    shape = (gx[1] - gx[0] + 1, gy[1] - gy[0] + 1)
    if min(shape) < config.quality.min_roi_px:
        return _fail(np.zeros(shape, bool), flags, prov, "roi_too_small", params, watch)

    norm, sat, image, exclude = _normalize(frame, ref, roi_to_slice(gx, gy), params, config)
    prov.update(normalize=norm.method, gain=norm.gain, polarity=params.polarity)
    if sat.any():
        flags.add("saturation")
    watch.lap("normalize")
    coarse = coarse_segment(image, exclude, config.segment, config.normalize,
                            contrast_cap=1.0 if params.polarity == "dark" else np.inf)
    sel = select_candidate(coarse, norm.invalid, config.segment, config.fit,
                           seed=_local(params.seed, origin), prior=_local(params.prior, origin))
    prov.update(segment=coarse.method, smooth_sigma=round(coarse.smooth_sigma, 2), select=sel.method,
                contrast=coarse.contrast, noise=coarse.noise, n_candidates=len(sel.candidates),
                runner_up_ratio=sel.runner_up)
    watch.lap("segment_select")
    if sel.best is None:
        return _fail(coarse.mask, flags, prov, "no_candidate", params, watch)

    rng = np.random.default_rng(config.fit.rng_seed)
    edges, steep_fit = _steep_edges(edge_image(image, exclude, coarse.smooth_sigma, config.rays),
                                    norm.invalid, coarse, sel, params, config, rng)
    prov.update(edges=f"rays(choice={params.edge_choice},level={params.level:g},"
                      f"passes={1 + config.rays.refine_passes})",
                n_rays=edges.n_rays, n_edges=len(edges.points), n_clipped=edges.n_clipped,
                multi_edge_frac=edges.multi_fraction)
    if len(edges.points) < config.quality.min_edge_points:
        return _fail(coarse.mask, flags, prov, "too_few_edges", params, watch)
    if steep_fit is None:
        return _fail(coarse.mask, flags, prov, "fit_failed", params, watch)
    clipped = edges.n_clipped > 0
    steep = fit_stats(edges.points, steep_fit, config.quality, config.fit, clipped=clipped)
    watch.lap("edges_fit")
    outer = _outer(image, exclude, steep_fit, clipped, config, rng)
    watch.lap("outer")
    flags |= measurement_flags(clipped=clipped, multi_fraction=edges.multi_fraction, runner_up=sel.runner_up,
                               contrast=coarse.contrast, noise=coarse.noise, saturated=bool(sat.any()),
                               stats=steep, shoulder_width=_shoulder(steep, outer), cfg=config.quality)
    return _build(steep, outer, edges, coarse, sel, flags, prov, origin, params, config, watch)


def _parse_roi(gridx, gridy, shape: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int], bool]:
    """Integer, clamped 1-based ROI bounds and whether clamping changed them."""
    try:
        gx = (int(gridx[0]), int(gridx[1]))
        gy = (int(gridy[0]), int(gridy[1]))
    except (TypeError, ValueError, IndexError) as exc:
        raise ValueError(f"gridx/gridy must be (min, max) pairs, got {gridx!r}, {gridy!r}") from exc
    cx, cy = clamp_roi(gx, gy, shape[0], shape[1])
    return cx, cy, (cx, cy) != (tuple(sorted(gx)), tuple(sorted(gy)))


def _local(point, origin: tuple[int, int]):
    """1-based full-image (row, col[, R]) -> 0-based ROI coordinates."""
    if point is None:
        return None
    return (point[0] - origin[0], point[1] - origin[1]) + tuple(point[2:])


def _normalize(frame: GrayFrame, ref: GrayFrame | None, roi: tuple[slice, slice], params: EdgeParams,
               config: EdgeConfig) -> tuple[Normalized, np.ndarray, np.ndarray, np.ndarray]:
    """Stage A on the ROI; returns (normalized, saturated, working image, excluded pixels)."""
    values = frame.values[roi]
    sat = saturation_mask(values, frame.full_scale, config.normalize)
    dark = params.polarity == "dark"
    # a bright bubble may itself be saturated: then saturation is signal, not glare
    sat_excl = sat if dark else np.zeros_like(sat)
    ref_values, ref_bad = None, None
    if ref is not None:
        ref_values = ref.values[roi]
        ref_bad = saturation_mask(ref_values, ref.full_scale, config.normalize) | ref.non_finite[roi]
    norm = normalize_roi(values, sat_excl, config.normalize, reference=ref_values, ref_invalid=ref_bad,
                         invalid=frame.non_finite[roi])
    image = norm.image if dark else (2.0 - norm.image).astype(np.float32)
    return norm, sat, image, sat_excl | norm.invalid


def _steep_edges(e: np.ndarray, invalid: np.ndarray, coarse: CoarseMask, sel: Selection,
                 params: EdgeParams, cfg: EdgeConfig,
                 rng: np.random.Generator) -> tuple[RayEdges, CircleFit | None]:
    """Ray passes around the candidate circle, each re-centred on the previous fit.

    The reported fit is a final robust fit of the last pass's points (a fresh
    RANSAC draw), exactly as in the prototype.
    """
    circle, window = sel.best.circle, None
    edges = None
    for _ in range(1 + cfg.rays.refine_passes):
        edges = ray_edges(e, circle, coarse.contrast, coarse.noise, cfg.rays,
                          edge_choice=params.edge_choice, level=params.level,
                          window=window, invalid=invalid)
        if len(edges.points) < cfg.quality.min_edge_points:
            return edges, None
        fit = fit_circle(edges.points, rng, cfg.fit)
        if fit is None:
            break
        circle, window = fit.circle, refine_window(fit.radius, cfg.rays)
    return edges, fit_circle(edges.points, rng, cfg.fit)


def _outer(image: np.ndarray, exclude: np.ndarray, steep: CircleFit, clipped: bool, cfg: EdgeConfig,
           rng: np.random.Generator) -> FitStats | None:
    """Definition B: fit a circle to the outer-extent points (None if undefined)."""
    pts = outer_extent(image, exclude, steep.circle, cfg.rays)
    if len(pts) < cfg.rays.outer_min_points:
        return None
    fit = fit_circle(pts, rng, cfg.fit)
    if fit is None or not np.isfinite(fit.radius):
        return None
    return fit_stats(pts, fit, cfg.quality, cfg.fit, clipped=clipped)


def _shoulder(steep: FitStats, outer: FitStats | None) -> float:
    return outer.fit.radius - steep.fit.radius if outer is not None else float("nan")


def _empty_result(mask: np.ndarray, flags: set[str], prov: dict, gate: float) -> EdgeResult:
    nan = float("nan")
    return EdgeResult(radius=nan, rc=nan, cc=nan, radius_sigma=nan, edge_xy=np.empty((0, 2)),
                      confidence=0.0, flags=frozenset(flags), r_area=nan, outer_radius=nan,
                      shoulder_width=nan, arc_coverage=0.0, inlier_fraction=0.0, mask=mask,
                      provenance=prov, confidence_gate=gate)


def _fail(mask: np.ndarray, flags: set[str], prov: dict, reason: str, params: EdgeParams,
          watch: _Stopwatch) -> EdgeResult:
    prov["failure"] = reason
    watch.stop()
    return _empty_result(mask, flags | {reason}, prov, params.confidence_gate)


def _build(steep: FitStats, outer: FitStats | None, edges: RayEdges, coarse: CoarseMask, sel: Selection,
           flags: set[str], prov: dict, origin: tuple[int, int], params: EdgeParams, config: EdgeConfig,
           watch: _Stopwatch) -> EdgeResult:
    """Pick the reported definition and assemble the result (shifted to 1-based full-image coords)."""
    outer_desc = "undefined"
    if outer is not None:
        outer_desc = f"outer_extent(level={config.rays.outer_level:g}) + {outer.fit.method}"
    else:
        prov["fallbacks"].append("outer extent undefined")
    prov.update(fit=steep.fit.method, steep_radius=steep.fit.radius,
                steep_centre=(steep.fit.rc + origin[0], steep.fit.cc + origin[1]),
                rms_px=steep.rms_px, max_gap_deg=steep.max_gap_deg, outer=outer_desc)
    chosen = steep if params.edge_definition == "steep" else outer
    if chosen is None:
        return _fail(coarse.mask, flags, prov, "outer_undefined", params, watch)
    terms = confidence_terms(chosen, coarse.contrast, coarse.noise, sel.runner_up, edges.multi_fraction,
                             config.quality)
    prov["confidence_terms"] = {k: round(v, 3) for k, v in terms.items()}
    watch.stop()
    shift = np.array(origin, float)
    fit = chosen.fit
    outer_radius = outer.fit.radius if outer is not None else float("nan")
    return EdgeResult(radius=float(fit.radius), rc=float(fit.rc + shift[0]), cc=float(fit.cc + shift[1]),
                      radius_sigma=fit.radius_sigma, edge_xy=chosen.inliers + shift,
                      confidence=confidence(terms), flags=frozenset(flags), r_area=chosen.r_area,
                      outer_radius=float(outer_radius), shoulder_width=float(_shoulder(steep, outer)),
                      arc_coverage=chosen.arc_coverage, inlier_fraction=chosen.inlier_fraction,
                      mask=coarse.mask, provenance=prov, confidence_gate=params.confidence_gate)
