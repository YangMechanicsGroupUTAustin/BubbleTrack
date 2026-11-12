"""The Edge engine: single-frame bubble measurement (``detect_edge``).

Pipeline (ROI-local, 0-based pixel coordinates inside; 1-based full-image
coordinates in and out, i.e. pixel ``(i, j)`` has coordinate ``(i+1, j+1)``):

A  normalise by the reference frame (gain + residual surface) or, without one,
   by a robust background surface;  B  coarse mask at 50 % contrast;
E  candidate selection;  C  sub-pixel edges along ~2*pi*R rays;
D  robust circle fit (``r_circle``) and the outer extent;
G  the bubble as a region: outer iso-contour at the median ray-edge level,
   holes filled, thin necks split, the part hidden by the ROI border completed
   with the circle fit, fragments counted;  F  confidence, flags, provenance.
"""

from __future__ import annotations

import time
from dataclasses import asdict, replace

import numpy as np

from bubbletrack.model.conventions import clamp_roi, roi_to_slice
from bubbletrack.model.edge.config import DEFAULT_CONFIG, EdgeConfig
from bubbletrack.model.edge.fit import CircleFit, fit_circle
from bubbletrack.model.edge.normalize import GrayFrame, Normalized, normalize_roi, saturation_mask, to_gray_frame
from bubbletrack.model.edge.outline import Outline, RayPoints, background_noise, measure_outline
from bubbletrack.model.edge.quality import (
    FitStats,
    confidence,
    confidence_terms,
    fit_stats,
    is_non_circular,
    measurement_flags,
)
from bubbletrack.model.edge.rays import (
    RayEdges,
    edge_image,
    outer_extent_with_levels,
    outer_image,
    ray_edges,
    refine_window,
)
from bubbletrack.model.edge.region import own_cell
from bubbletrack.model.edge.result import EdgeParams, EdgeResult
from bubbletrack.model.edge.segment import CoarseMask, Selection, coarse_segment, select_candidate

ENGINE_VERSION = "edge-1.1.0"

__all__ = ["ENGINE_VERSION", "EdgeParams", "EdgeResult", "detect_edge", "detect_edge_with_config"]


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
    """Measure the bubble in one frame.

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
        ``radius`` is the area-equivalent radius of the bubble region (NaN when
        nothing was found; ``flags`` then names why).

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
    e = edge_image(image, exclude, coarse.smooth_sigma, config.rays)
    edges, steep_fit = _steep_edges(e, norm.invalid, coarse, sel, params, config, rng)
    prov.update(edges=f"rays(choice={params.edge_choice},level={params.level:g},"
                      f"passes={1 + config.rays.refine_passes})",
                n_rays=edges.n_rays, n_edges=len(edges.points), n_clipped=edges.n_clipped,
                multi_edge_frac=edges.multi_fraction)
    if len(edges.points) < config.quality.min_edge_points:
        return _fail(coarse.mask, flags, prov, "too_few_edges", params, watch)
    if steep_fit is None:
        return _fail(coarse.mask, flags, prov, "fit_failed", params, watch)
    steep = fit_stats(edges.points, steep_fit, config.fit)
    watch.lap("edges_fit")
    outer = _outer(image, exclude, steep_fit, config, rng)
    watch.lap("outer")
    steep_line, outer_line = _outlines(image, e, exclude, coarse, sel, edges, steep, outer, origin, params,
                                       config)
    watch.lap("region")
    if steep_line is None:
        return _fail(coarse.mask, flags, prov, "no_region", params, watch)
    _describe(prov, steep_line, outer, outer_line, origin, config)
    chosen = steep_line if params.edge_definition == "steep" else outer_line
    if chosen is None:
        return _fail(coarse.mask, flags, prov, "outer_undefined", params, watch)
    outer_radius = _outer_radius(outer, outer_line)
    shoulder = _shoulder(steep_line, outer_line, steep, outer_radius)
    region = chosen.region
    non_circular = is_non_circular(chosen.stats.fit.radius, region.radius, region.solidity, config.quality)
    flags |= measurement_flags(region=region, non_circular=non_circular, multi_fraction=edges.multi_fraction,
                               runner_up=sel.runner_up, contrast=coarse.contrast, noise=coarse.noise,
                               saturated=bool(sat.any()), shoulder_width=shoulder, cfg=config.quality)
    return _build(chosen, outer_radius, shoulder, edges, coarse, sel, flags, prov, origin, params, config, watch)


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


def _outer(image: np.ndarray, exclude: np.ndarray, steep: CircleFit, cfg: EdgeConfig,
           rng: np.random.Generator) -> tuple[FitStats, RayPoints] | None:
    """Definition B: circle fit and ray points of the outer extent (None if undefined)."""
    pts, levels, index = outer_extent_with_levels(image, exclude, steep.circle, cfg.rays)
    if len(pts) < cfg.rays.outer_min_points:
        return None
    fit = fit_circle(pts, rng, cfg.fit)
    if fit is None or not np.isfinite(fit.radius):
        return None
    return fit_stats(pts, fit, cfg.fit), RayPoints(pts, levels, index, cfg.rays.outer_rays)


def _preferred(params: EdgeParams, origin: tuple[int, int]) -> list[tuple[str, tuple]]:
    """Points that choose the main body before the candidate, in order (0-based ROI)."""
    out = []
    if params.prior is not None:
        out.append(("prior", _local(params.prior, origin)[:2]))
    if params.seed is not None:
        out.append(("seed", _local(params.seed, origin)))
    return out


def _outlines(image: np.ndarray, e: np.ndarray, exclude: np.ndarray, coarse: CoarseMask, sel: Selection,
              edges: RayEdges, steep: FitStats, outer: tuple[FitStats, RayPoints] | None,
              origin: tuple[int, int], params: EdgeParams,
              config: EdgeConfig) -> tuple[Outline | None, Outline | None]:
    """Stage G for definition A (steep) and, with ``edge_definition == "outer"``, B (outer extent)."""
    preferred = _preferred(params, origin)
    background = (~exclude) & (coarse.smooth > 1.0 - config.segment.texture_bg_level * coarse.contrast)
    min_step = max(config.rays.min_step_frac * coarse.contrast, config.rays.noise_step_k * coarse.noise)
    blur = max(config.rays.edge_blur_min, config.rays.edge_blur_frac * coarse.smooth_sigma)
    steep_rays = RayPoints(edges.points, edges.levels, edges.rays, edges.n_rays)
    steep_line = measure_outline(e, steep_rays, exclude, sel.best.interior, preferred, steep, min_step=min_step,
                                 noise=background_noise(e, background), blur=blur, config=config)
    if steep_line is None or outer is None or params.edge_definition != "outer":
        return steep_line, None
    o_img = outer_image(image, exclude, config.rays)
    outer_line = measure_outline(o_img, outer[1], exclude, sel.best.interior, preferred, outer[0],
                                 min_step=min_step,
                                 noise=background_noise(o_img, background), blur=config.rays.outer_blur_sigma,
                                 config=config, allowed=own_cell(steep_line.region))
    if outer_line is None:
        return steep_line, None
    # B sits on the toe of the edge, where no steep step is expected: the edge it is
    # derived from is A's, so A's support stands for it
    support = replace(outer_line.evidence, support=steep_line.evidence.support)
    return steep_line, replace(outer_line, evidence=support)


def _outer_radius(outer: tuple[FitStats, RayPoints] | None, outer_line: Outline | None) -> float:
    """Definition B: the outer region's area radius when measured, else its circle fit (steep mode)."""
    if outer_line is not None:
        return outer_line.region.radius
    return float(outer[0].fit.radius) if outer is not None else float("nan")


def _shoulder(steep_line: Outline, outer_line: Outline | None, steep: FitStats, outer_radius: float) -> float:
    """B - A, both areas (outer mode) or both circle fits (steep mode)."""
    if outer_line is not None:
        return outer_line.region.radius - steep_line.region.radius
    return outer_radius - steep.fit.radius


def _describe(prov: dict, steep: Outline, outer: tuple[FitStats, RayPoints] | None, outer_line: Outline | None,
              origin: tuple[int, int], config: EdgeConfig) -> None:
    fit = steep.stats.fit
    outer_desc = "undefined"
    if outer is not None:
        outer_desc = f"outer_extent(level={config.rays.outer_level:g}) + {outer[0].fit.method}"
        if outer_line is not None:
            outer_desc += f" + region(median level {outer_line.level:.3f})"
    else:
        prov["fallbacks"].append("outer extent undefined")
    reg = steep.region
    prov.update(fit=fit.method, steep_radius=reg.radius, steep_r_circle=fit.radius,
                steep_centre=(fit.rc + origin[0], fit.cc + origin[1]), rms_px=steep.stats.rms_px,
                max_gap_deg=steep.stats.max_gap_deg, outer=outer_desc,
                region={"level": round(steep.level, 4), "anchor": reg.anchor, "lobes": reg.n_lobes,
                        "clipped": reg.clipped, "hidden_fraction": round(reg.hidden_fraction, 4),
                        "support": round(steep.evidence.support, 3)})


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


def _build(line: Outline, outer_radius: float, shoulder: float, edges: RayEdges, coarse: CoarseMask,
           sel: Selection, flags: set[str], prov: dict, origin: tuple[int, int], params: EdgeParams,
           config: EdgeConfig, watch: _Stopwatch) -> EdgeResult:
    """Assemble the result of the chosen definition (shifted to 1-based full-image coords).

    A frame below the confidence gate reports the robust circle as its best guess
    (``radius_kind == "circle"``, flag ``circle_fallback``): its region could not be
    trusted, and on such frames (transparent or low-contrast bubbles, broken walls)
    the circle through the edge points is the better estimate.  ``r_area`` keeps
    the region's value.
    """
    region, stats = line.region, line.stats
    terms = confidence_terms(region, line.evidence, line.sigma_r, coarse.contrast, coarse.noise, sel.runner_up,
                             edges.multi_fraction, config.quality)
    prov["confidence_terms"] = {k: round(v, 3) for k, v in terms.items()}
    conf = confidence(terms)
    watch.stop()
    shift = np.array(origin, float)
    fallback = conf < params.confidence_gate
    rc, cc = (stats.fit.rc, stats.fit.cc) if region.clipped or fallback else region.visible.centroid
    centroid = region.centroid
    radius, sigma, kind = region.radius, line.sigma_r, "area+circle_completion" if region.completed else "area"
    if fallback:
        radius, sigma, kind = float(stats.fit.radius), stats.fit.radius_sigma, "circle"
        flags = flags | {"circle_fallback"}
    return EdgeResult(radius=radius, rc=float(rc + shift[0]), cc=float(cc + shift[1]),
                      radius_sigma=sigma, edge_xy=region.edge + shift,
                      confidence=conf, flags=frozenset(flags), r_area=region.radius,
                      outer_radius=float(outer_radius),
                      shoulder_width=float(shoulder), arc_coverage=stats.arc_coverage,
                      inlier_fraction=stats.inlier_fraction, mask=region.mask, provenance=prov,
                      confidence_gate=params.confidence_gate, r_circle=float(stats.fit.radius),
                      area_px=float(region.area), perimeter_px=float(region.perimeter),
                      circularity=region.circularity, solidity=region.solidity,
                      fragment_count=region.fragment_count, fragment_area_px=region.fragment_area,
                      contour=region.contour + shift, radius_kind=kind,
                      centroid=(float(centroid[0] + shift[0]), float(centroid[1] + shift[1])))
