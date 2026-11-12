"""Stages B and E: coarse mask, border-aware hole filling, candidates and selection.

The coarse mask only has to find *which* dark region is the bubble and roughly
where it is; the edge itself is measured later along rays.  Coordinates here
are 0-based ROI pixels ``[row, col]``.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage
from skimage import measure

from bubbletrack.model.circle_fit import circle_fit_taubin
from bubbletrack.model.edge.config import FitConfig, NormalizeConfig, SegmentConfig
from bubbletrack.model.edge.fit import fit_circle
from bubbletrack.model.edge.normalize import noise_sigma

_FILL_SWALLOW_RATIO = 1.6  # wall closing that grows a component more than this swallowed background


@dataclass(frozen=True)
class CoarseMask:
    """Output of stage B."""

    mask: np.ndarray
    """Dark pixels of the smoothed image (bool, ROI shape)."""
    smooth: np.ndarray
    """Smoothed normalised image used for the mask and the candidate contrast."""
    contrast: float
    """Frame contrast c0 = 1 - darkest smoothed level."""
    noise: float
    """Residual background texture at the working smoothing scale."""
    smooth_sigma: float
    method: str


@dataclass(frozen=True)
class Candidate:
    """One dark component, with its circle and score terms."""

    label: int
    area: float
    score: float
    circle: tuple[float, float, float]
    seed_distance: float
    circularity: float
    contrast: float
    inside_fraction: float
    """Fraction of the component inside its robust circle (the polarity test)."""
    inside_any: float = 1.0
    """The same for the better of the robust and the whole-boundary Taubin circle."""
    interior: tuple[float, float] = (float("nan"), float("nan"))
    """Deepest dark pixel of the component (distance-transform maximum): inside any shape."""
    holds_prior: bool = False
    """The prior centre lies inside the filled component."""

    def passes_polarity(self, cfg: SegmentConfig) -> bool:
        """Inside its own circle, or chosen by the user (seed click) or the tracker (prior)."""
        return self.inside_fraction >= cfg.polarity_min_inside or self.seed_distance == 0.0 or self.holds_prior


@dataclass(frozen=True)
class Selection:
    """Output of stage E."""

    best: Candidate | None
    candidates: tuple[Candidate, ...]
    runner_up: float
    """Ambiguity in [0, 1]: rival strength relative to the choice (0 = unambiguous)."""
    method: str


def coarse_segment(image: np.ndarray, exclude: np.ndarray, cfg: SegmentConfig,
                   norm_cfg: NormalizeConfig, *, contrast_cap: float = 1.0) -> CoarseMask:
    """Threshold the normalised image at ``seg_level`` of the frame contrast.

    The smoothing is the smallest scale whose residual background texture is
    at most ``texture_target`` of the contrast, so speckle does not fragment
    the mask while a clean frame keeps a sharp one.
    """
    s3 = cv2.GaussianBlur(image, (0, 0), cfg.contrast_blur_sigma)
    s3 = np.where(exclude, 1.0, s3)
    c0 = float(np.clip(1.0 - s3.min(), cfg.contrast_min, contrast_cap))
    noise = noise_sigma(image, exclude, norm_cfg)
    step = cfg.texture_subsample
    background_like = ((~exclude) & (s3 > 1.0 - cfg.texture_bg_level * c0))[::step, ::step]
    sigma, texture = cfg.smooth_fallback, noise
    if background_like.sum() >= cfg.texture_min_pixels:
        for s_try in cfg.smooth_sigmas:
            sample = cv2.GaussianBlur(image, (0, 0), s_try)[::step, ::step][background_like]
            texture = 1.4826 * float(np.median(np.abs(sample - np.median(sample))))
            if texture <= c0 * cfg.texture_target:
                sigma = s_try
                break
    smooth = cv2.GaussianBlur(image, (0, 0), sigma)
    smooth = np.where(exclude, np.maximum(smooth, 1.0), smooth)
    mask = smooth < 1.0 - cfg.seg_level * c0
    return CoarseMask(mask=mask, smooth=smooth, contrast=c0, noise=max(texture, 1e-4),
                      smooth_sigma=float(sigma), method=f"global(level={cfg.seg_level:g}c)")


def fill_holes_border_aware(crop: np.ndarray, walls: tuple[bool, bool, bool, bool]) -> np.ndarray:
    """Fill holes in a component crop; ROI edges the crop touches act as closed walls.

    *walls* is ``(top, bottom, left, right)``.  A bubble clipped by the ROI
    border is then filled although its hole is open towards the border.  If
    closing the walls would swallow background (the component is a band), the
    plain fill is returned instead.
    """
    top, bottom, left, right = walls
    pad = np.pad(crop, 1, constant_values=False)
    if top:
        pad[0, :] = True
    if bottom:
        pad[-1, :] = True
    if left:
        pad[:, 0] = True
    if right:
        pad[:, -1] = True
    filled = ndimage.binary_fill_holes(pad)[1:-1, 1:-1]
    plain = ndimage.binary_fill_holes(crop)
    if filled.sum() > _FILL_SWALLOW_RATIO * max(plain.sum(), 1):
        return plain
    return filled


def boundary_points(crop: np.ndarray, r0: int, c0: int,
                    walls: tuple[bool, bool, bool, bool]) -> np.ndarray:
    """Boundary pixels of a filled crop in ROI coordinates, ROI-border pixels excluded."""
    er = cv2.erode(crop.astype(np.uint8), np.ones((3, 3), np.uint8),
                   borderType=cv2.BORDER_CONSTANT, borderValue=0)
    b = crop & (er == 0)
    top, bottom, left, right = walls
    if top:
        b[0, :] = False
    if bottom:
        b[-1, :] = False
    if left:
        b[:, 0] = False
    if right:
        b[:, -1] = False
    r, c = np.nonzero(b)
    return np.column_stack([r + r0, c + c0]).astype(float)


def _annulus_contrast(smooth: np.ndarray, circle: tuple[float, float, float], dark: float,
                      c0: float, cfg: SegmentConfig) -> float:
    """Contrast of a candidate against its own surround (a ring just outside it)."""
    h, w = smooth.shape
    rc, cc, radius = circle
    th = np.linspace(0, 2 * np.pi, cfg.annulus_samples, endpoint=False)
    rr = np.concatenate([rc + (radius * f + cfg.annulus_offset_px) * np.sin(th) for f in cfg.annulus_factors])
    cl = np.concatenate([cc + (radius * f + cfg.annulus_offset_px) * np.cos(th) for f in cfg.annulus_factors])
    ok = (rr >= 0) & (rr <= h - 1) & (cl >= 0) & (cl <= w - 1)
    if ok.sum() < cfg.annulus_min_samples:
        return 0.0
    ring = ndimage.map_coordinates(smooth, [rr[ok], cl[ok]], order=1)
    return float(np.clip((np.median(ring) - dark) / max(c0, 1e-3), 0, 1))


def _contains(crop: np.ndarray, r0: int, c0: int, point) -> bool:
    if point is None:
        return False
    sr, sc = int(round(point[0])) - r0, int(round(point[1])) - c0
    return bool(0 <= sr < crop.shape[0] and 0 <= sc < crop.shape[1] and crop[sr, sc])


def _seed_distance(holds_seed: bool, circle: tuple[float, float, float],
                   seed: tuple[float, float] | None) -> float:
    if seed is None:
        return float("inf")
    if holds_seed:
        return 0.0
    return float(np.hypot(circle[0] - seed[0], circle[1] - seed[1]))


def _prior_factor(circle: tuple[float, float, float], area: float, holds_prior: bool,
                  prior: tuple[float, float, float] | None, cfg: SegmentConfig) -> float:
    """Proximity to the expected circle, judged by the better of the circle fit and the
    region itself (centre inside it, area-equivalent radius), so shape costs nothing."""
    if prior is None:
        return 1.0
    pr, pc, p_rad = prior
    d_centre = 0.0 if holds_prior else np.hypot(circle[0] - pr, circle[1] - pc)
    return float(np.exp(-d_centre / (cfg.prior_centre_frac * p_rad + cfg.prior_centre_px))
                 * _prior_radius_term(circle, area, prior, cfg))


def _prior_radius_term(circle: tuple[float, float, float], area: float, prior: tuple[float, float, float],
                       cfg: SegmentConfig) -> np.floating:
    """The size part of :func:`_prior_factor` (the better of the circle and the area radius)."""
    p_rad = prior[2]
    d_radius = min(abs(circle[2] - p_rad), abs(np.sqrt(area / np.pi) - p_rad))
    return np.exp(-d_radius / (cfg.prior_radius_frac * p_rad + cfg.prior_radius_px))


def _inside_fractions(crop: np.ndarray, r0: int, c0: int, pts: np.ndarray,
                      circle: tuple[float, float, float], cfg: SegmentConfig) -> tuple[float, float]:
    """Fraction of the component inside its robust circle, and inside the better of that and
    the plain Taubin circle of the whole boundary (a peanut lies inside the latter)."""
    yy, xx = np.nonzero(crop)
    yy, xx = yy + r0, xx + c0
    out = []
    for rc, cc, radius in (circle, tuple(circle_fit_taubin(pts))):
        ok = np.isfinite(radius) and radius > 0
        out.append(float(np.mean(np.hypot(yy - rc, xx - cc) <= radius + cfg.polarity_margin_px)) if ok else 0.0)
    return out[0], max(out)


def _interior(crop: np.ndarray, r0: int, c0: int) -> tuple[float, float]:
    dt = cv2.distanceTransform(np.pad(crop, 1).astype(np.uint8), cv2.DIST_L2, 3)[1:-1, 1:-1]
    i, j = np.unravel_index(int(np.argmax(dt)), dt.shape)
    return float(i + r0), float(j + c0)


def _score_component(prop, coarse: CoarseMask, near_invalid: np.ndarray | None, cfg: SegmentConfig,
                     fit_cfg: FitConfig, rng: np.random.Generator, seed, prior) -> Candidate | None:
    h, w = coarse.mask.shape
    r0, c0, r1, c1 = prop.bbox
    walls = (r0 == 0, r1 == h, c0 == 0, c1 == w)
    crop = fill_holes_border_aware(prop.image, walls)
    pts = boundary_points(crop, r0, c0, walls)
    if near_invalid is not None and len(pts):
        pts = pts[~near_invalid[pts[:, 0].astype(int), pts[:, 1].astype(int)]]
    if len(pts) < cfg.min_boundary_pts:
        return None
    fit = fit_circle(pts, rng, fit_cfg, robust=len(pts) >= cfg.robust_fit_min_pts)
    if fit is None:
        return None
    circle = fit.circle
    rc, cc, radius = circle
    if not (cfg.min_radius_px < radius < cfg.max_radius_factor * max(h, w)):
        return None
    res = np.abs(np.hypot(pts[:, 0] - rc, pts[:, 1] - cc) - radius)
    rms = float(np.sqrt(np.mean(np.minimum(res, cfg.rms_clip_frac * radius) ** 2)))
    q_circ = float(np.exp(-(rms / max(radius, 1.0)) / cfg.circularity_scale))
    holds_seed, holds_prior = _contains(crop, r0, c0, seed), _contains(crop, r0, c0, prior)
    inside, inside_any = _inside_fractions(crop, r0, c0, pts, circle, cfg)
    if inside_any < cfg.polarity_min_inside and not (holds_seed or holds_prior):
        return None        # dark region lies outside any circle of its own: aperture corner, wall
    dark = float(np.median(coarse.smooth[r0:r1, c0:c1][crop]))
    q_con = _annulus_contrast(coarse.smooth, circle, dark, coarse.contrast, cfg)
    band = (r1 - r0) > cfg.band_fraction * h or (c1 - c0) > cfg.band_fraction * w
    area = float(crop.sum())
    score = np.sqrt(area) * q_circ * q_con * (cfg.band_penalty if band else 1.0)
    score *= _prior_factor(circle, area, holds_prior, prior, cfg)
    return Candidate(label=int(prop.label), area=area, score=float(score), circle=circle,
                     seed_distance=_seed_distance(holds_seed, circle, seed),
                     circularity=q_circ, contrast=q_con, inside_fraction=inside, inside_any=inside_any,
                     interior=_interior(prop.image, r0, c0), holds_prior=holds_prior)


def _holder_runner_up(best: Candidate, cands: list[Candidate], prior: tuple[float, float, float],
                      cfg: SegmentConfig) -> float:
    """Ambiguity of a component chosen because it holds the prior centre.

    Its own score is taken without the prior's size term: the centre chose it, and a radius
    that is off (a rough circle dragged by the user, a click) must not make a rival look
    stronger, e.g. the bubble's own dark core when the radius fits the core.  Rivals keep
    their full prior factor."""
    others = [c for c in cands if c is not best]
    if not others or best.score <= 0:
        return 0.0
    free = best.score / max(float(_prior_radius_term(best.circle, best.area, prior, cfg)), 1e-300)
    return float(min(max(c.score for c in others) / free, 1.0))


def _runner_up(best: Candidate, cands: list[Candidate], by_seed: bool) -> float:
    others = [c for c in cands if c is not best]
    if not others:
        return 0.0
    if by_seed:
        if best.seed_distance == 0.0:
            return 0.0      # the click lies inside the chosen component
        nearest_other = min(c.seed_distance for c in others)
        return float(min(best.seed_distance / max(nearest_other, 1e-9), 1.0))
    rival = max(c.score for c in others)
    return float(min(rival / best.score, 1.0)) if best.score > 0 else 0.0


def select_candidate(coarse: CoarseMask, invalid: np.ndarray, cfg: SegmentConfig, fit_cfg: FitConfig,
                     *, seed: tuple[float, float] | None = None,
                     prior: tuple[float, float, float] | None = None) -> Selection:
    """Pick the bubble among the dark components of the coarse mask.

    Score = sqrt(area) x circularity x local-annulus contrast (x prior
    proximity).  Components that lie mostly outside their own circle are
    rejected (polarity test), unless they hold the seed or the prior centre:
    a clicked or tracked bubble is never rejected for its shape.  When no
    component passes, those inside the Taubin circle of their whole boundary
    are taken, so a single dark object of any shape is found.  With a
    *seed*, the component containing it, or else the one whose circle centre
    is nearest, is chosen; with a *prior*, the component containing its
    centre, or else the best score.  The ambiguity of a component chosen for
    holding the prior centre ignores the prior radius (:func:`_holder_runner_up`),
    so a rough radius costs no confidence.

    Parameters
    ----------
    coarse : CoarseMask
    invalid : ndarray of bool
        Boundary points next to these pixels are ignored.
    cfg, fit_cfg : SegmentConfig, FitConfig
    seed : (row, col), optional
        0-based ROI coordinates of a click on the bubble.
    prior : (row, col, radius), optional
        0-based ROI circle expected from a neighbouring frame.
    """
    h, w = coarse.mask.shape
    near_invalid = None
    if invalid.any():
        k = cfg.invalid_margin_px
        near_invalid = cv2.dilate(invalid.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
    labels = measure.label(coarse.mask, connectivity=2)
    props = [p for p in measure.regionprops(labels) if p.area >= cfg.min_area_px]
    props = sorted(props, key=lambda p: p.area, reverse=True)[:cfg.max_candidates * cfg.props_factor]
    rng = np.random.default_rng(cfg.rng_seed)
    scored: list[Candidate] = []
    for prop in props:
        cand = _score_component(prop, coarse, near_invalid, cfg, fit_cfg, rng, seed, prior)
        if cand is not None:
            scored.append(cand)
            if sum(c.passes_polarity(cfg) for c in scored) >= cfg.max_candidates:
                break
    # a shape outside its robust circle (a peanut) is only taken when nothing passes the test
    cands = [c for c in scored if c.passes_polarity(cfg)] or scored
    method = "seed" if seed is not None else "score"
    method += "+prior" if prior is not None else ""
    if not cands:
        return Selection(None, (), 0.0, method)
    by_seed = seed is not None
    holding = [c for c in cands if c.holds_prior]
    if by_seed:
        best = min(cands, key=lambda c: c.seed_distance)
    else:
        best = holding[0] if holding else max(cands, key=lambda c: c.score)
    if holding and not by_seed:
        return Selection(best, tuple(cands), _holder_runner_up(best, cands, prior, cfg), method)
    return Selection(best, tuple(cands), _runner_up(best, cands, by_seed), method)
