"""Stage C: radial sub-pixel edges (definition A) and the outer extent (definition B).

Definition A ("steep"): along each ray from the circle centre, the edge is the
sub-pixel crossing of the local 50 % level between the dark plateau inside and
the bright plateau outside a significant dark-to-bright step.  For the outer
edge the step must also straddle the frame's mid level, so a faint halo or an
inner ring is not taken for the bubble wall.

Definition B ("outer extent"): beyond the 50 % crossing, the first radius where
the profile recovers to 90 % of the way from the dark interior to the local
background.  B - A is the shoulder width.

Coordinates are 0-based ROI pixels ``[row, col]``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy import ndimage

from bubbletrack.model.edge.config import RayConfig

STATUS_NONE, STATUS_EDGE, STATUS_CLIPPED = 0, 1, 2


@dataclass(frozen=True)
class RayEdges:
    """Edge points found along the rays of one pass."""

    points: np.ndarray
    """(M, 2) sub-pixel edge points [row, col]."""
    status: np.ndarray
    """Per ray: STATUS_EDGE, STATUS_CLIPPED (left the valid region without an edge) or STATUS_NONE."""
    multi_fraction: float
    """Fraction of rays with two or more separate steps (halo / double ring)."""
    n_rays: int
    levels: np.ndarray = field(default_factory=lambda: np.empty(0))
    """Per edge point: the absolute crossing level between its local plateaus."""
    rays: np.ndarray = field(default_factory=lambda: np.empty(0, int))
    """Per edge point: the index of its ray (rays are evenly spaced in angle)."""

    @property
    def n_clipped(self) -> int:
        return int(np.sum(self.status == STATUS_CLIPPED))


def edge_image(image: np.ndarray, exclude: np.ndarray, smooth_sigma: float, cfg: RayConfig) -> np.ndarray:
    """Lightly smoothed normalised image sampled by the rays; excluded pixels are background."""
    e = cv2.GaussianBlur(image, (0, 0), max(cfg.edge_blur_min, cfg.edge_blur_frac * smooth_sigma))
    return np.where(exclude, np.maximum(e, 1.0), e)


def initial_window(radius: float, cfg: RayConfig) -> float:
    return max(cfg.init_window_min, cfg.init_window_frac * radius) + cfg.init_window_pad


def refine_window(radius: float, cfg: RayConfig) -> float:
    return max(cfg.refine_window_min, cfg.refine_window_frac * radius) + cfg.refine_window_pad


def _sample_rays(img: np.ndarray, rc: float, cc: float, th: np.ndarray, rs: np.ndarray,
                 invalid: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """Bilinear profiles along rays and the in-ROI / valid mask of each sample."""
    h, w = img.shape
    rr = rc + np.sin(th)[:, None] * rs[None, :]
    cl = cc + np.cos(th)[:, None] * rs[None, :]
    inside = (rr >= 0) & (rr <= h - 1) & (cl >= 0) & (cl <= w - 1)
    if invalid is not None and invalid.any():
        ri = np.clip(np.rint(rr), 0, h - 1).astype(int)
        ci = np.clip(np.rint(cl), 0, w - 1).astype(int)
        inside &= ~invalid[ri, ci]
    prof = ndimage.map_coordinates(img, [np.clip(rr, 0, h - 1).ravel(), np.clip(cl, 0, w - 1).ravel()],
                                   order=1, mode="nearest").reshape(rr.shape)
    return prof, inside


def ray_edges(e: np.ndarray, circle: tuple[float, float, float], contrast: float, noise: float,
              cfg: RayConfig, *, edge_choice: str = "outer", level: float = 0.5,
              window: float | None = None, invalid: np.ndarray | None = None) -> RayEdges:
    """Sub-pixel steep edges along about 2*pi*R rays around *circle*.

    Parameters
    ----------
    e : ndarray
        Edge image (see :func:`edge_image`).
    circle : (rc, cc, radius)
        Current circle estimate; rays span ``radius +/- window``.
    contrast, noise : float
        Frame contrast c0 and residual noise from stage B.
    cfg : RayConfig
    edge_choice : {"outer", "inner"}
        With several steps on a ray, take the outermost (seen from the
        background) or the innermost (seen from the core).
    level : float
        Crossing level between the local plateaus (0.5 defines the edge).
    window : float, optional
        Half-width of the radial search; defaults to :func:`initial_window`.
    invalid : ndarray of bool, optional
        Samples here are treated as outside the ROI.
    """
    rc, cc, r0 = circle
    n_rays = int(np.clip(round(2 * np.pi * r0), cfg.min_rays, cfg.max_rays))
    th = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    win = initial_window(r0, cfg) if window is None else window
    step = cfg.step_px
    rs = np.arange(max(0.0, r0 - win), r0 + win + step, step)
    if len(rs) < cfg.min_samples:
        rs = np.arange(rs[0], rs[0] + cfg.min_samples * step, step)
    ns = len(rs)
    prof, inside = _sample_rays(e, rc, cc, th, rs, invalid)
    # contiguous valid run per ray: [a, b)
    any_in = inside.any(1)
    a = np.argmax(inside, axis=1)
    j = np.arange(ns)[None, :]
    out_after = (j >= a[:, None]) & ~inside
    b = np.where(out_after.any(1), np.argmax(out_after, axis=1), ns)
    valid = (j >= a[:, None]) & (j < b[:, None]) & any_in[:, None]
    g = np.gradient(prof, step, axis=1)
    d1 = int(round(cfg.amplitude_half_px / step))
    plate = int(round(cfg.plateau_px / step))
    pp = np.pad(prof, ((0, 0), (d1, d1)), mode="edge")
    hi_side, lo_side = pp[:, 2 * d1:], pp[:, :-2 * d1]
    amp = hi_side - lo_side
    local_max = np.zeros_like(valid)
    local_max[:, 1:-1] = (g[:, 1:-1] > g[:, :-2]) & (g[:, 1:-1] >= g[:, 2:]) & (g[:, 1:-1] > 0)
    min_amp = max(cfg.min_step_frac * contrast, cfg.noise_step_k * noise)
    dark_seen = (j - a[:, None] >= d1) | (a[:, None] == 0)
    bright_seen = (b[:, None] - 1 - j >= d1) | (b[:, None] == ns)
    cand = local_max & valid & (amp >= min_amp) & dark_seen & bright_seen
    outer = edge_choice == "outer"
    if outer:
        mid = 1.0 - 0.5 * contrast
        cand &= (hi_side >= mid) & (lo_side <= mid)
    # candidates closer than the cluster gap belong to the same step
    gap = int(cfg.cluster_gap_px / step)
    csum = np.pad(np.cumsum(cand, axis=1), ((0, 0), (gap + 1, 0)))
    starts = cand & ((csum[:, gap:-1] - csum[:, :-gap - 1]) == 0)
    multi = float(np.sum(starts.sum(1) >= 2)) / n_rays
    has = cand.any(1)
    if outer:
        k_ref = ns - 1 - np.argmax(cand[:, ::-1], axis=1)
        win_m = (j >= (k_ref - 2 * gap)[:, None]) & (j <= k_ref[:, None])
    else:
        k_ref = np.argmax(cand, axis=1)
        win_m = (j >= k_ref[:, None]) & (j <= (k_ref + 2 * gap)[:, None])
    k_sel = np.argmax(np.where(win_m & cand, g, -np.inf), axis=1)
    status = np.zeros(n_rays, int)
    clipped = (a > 0) | (b < ns) | ~any_in
    status[~has & clipped] = STATUS_CLIPPED
    idx = np.nonzero(has)[0]
    if idx.size == 0:
        return RayEdges(np.empty((0, 2)), status, multi, n_rays)
    k = k_sel[idx]
    p = prof[idx]
    v = valid[idx]
    jj = np.arange(ns)[None, :]
    lo_win = (jj >= (k - plate)[:, None]) & (jj <= k[:, None]) & v
    hi_win = (jj >= k[:, None]) & (jj <= (k + plate)[:, None]) & v
    d_lvl = np.where(lo_win, p, np.inf).min(1)
    b_lvl = np.where(hi_win, p, -np.inf).max(1)
    lvl = d_lvl + level * (b_lvl - d_lvl)
    up = (p[:, :-1] < lvl[:, None]) & (p[:, 1:] >= lvl[:, None]) & v[:, :-1] & v[:, 1:]
    near = up & (np.abs(jj[:, :-1] - k[:, None]) <= plate)
    jc = np.where(near, np.abs(jj[:, :-1] - k[:, None]), 10 ** 6).argmin(1)
    found = near.any(1)
    rows = np.arange(len(k))
    p0, p1 = p[rows, jc], p[rows, np.minimum(jc + 1, ns - 1)]
    t = (lvl - p0) / np.maximum(p1 - p0, 1e-9)
    r_edge = rs[jc] + np.clip(t, 0, 1) * step
    r_edge, ii = r_edge[found], idx[found]
    pts = np.column_stack([rc + np.sin(th[ii]) * r_edge, cc + np.cos(th[ii]) * r_edge])
    status[ii] = STATUS_EDGE
    return RayEdges(pts, status, multi, n_rays, lvl[found], ii)


def _row_percentile(p: np.ndarray, cols: np.ndarray, q: float, min_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-row percentile (numpy's linear method) of the finite samples in columns *cols*.

    Vectorised: ``np.nanpercentile`` loops over rows in Python.  Rows with
    fewer than *min_count* finite samples give NaN.
    """
    s = np.sort(p[:, cols], axis=1)              # NaN sorts last
    count = np.isfinite(s).sum(1)
    rows = count >= min_count
    out = np.full(p.shape[0], np.nan)
    if rows.any():
        n = count[rows]
        pos = q / 100.0 * (n - 1)
        lo = np.floor(pos).astype(int)
        hi = np.minimum(lo + 1, n - 1)
        sr = s[rows]
        idx = np.arange(len(n))
        out[rows] = sr[idx, lo] + (pos - lo) * (sr[idx, hi] - sr[idx, lo])
    return out, rows


def outer_image(image: np.ndarray, exclude: np.ndarray, cfg: RayConfig) -> np.ndarray:
    """The smoothed image the outer extent is measured on; excluded pixels are background."""
    smooth = cv2.GaussianBlur(image, (0, 0), cfg.outer_blur_sigma)
    return np.where(exclude, np.maximum(smooth, 1.0), smooth)


def outer_extent(image: np.ndarray, exclude: np.ndarray, circle: tuple[float, float, float],
                 cfg: RayConfig) -> np.ndarray:
    """Definition-B edge points (see :func:`outer_extent_with_levels`)."""
    return outer_extent_with_levels(image, exclude, circle, cfg)[0]


def outer_extent_with_levels(image: np.ndarray, exclude: np.ndarray, circle: tuple[float, float, float],
                             cfg: RayConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Definition-B edge points: where each ray recovers 90 % of its step beyond the steep edge.

    Per ray, the dark level is a low percentile just inside the steep edge and
    the background a high percentile well outside it; rays that leave the
    valid region before both are observed are dropped.

    Parameters
    ----------
    image : ndarray
        Normalised image (background 1.0).
    exclude : ndarray of bool
        Pixels without usable data; a ray stops at the first one.
    circle : (rc, cc, radius)
        The steep-edge circle.
    cfg : RayConfig

    Returns
    -------
    points : ndarray of shape (M, 2)
        Outer-extent points [row, col].
    levels : ndarray of shape (M,)
        The absolute recovery level of each point's ray.
    rays : ndarray of shape (M,)
        The index of each point's ray among ``cfg.outer_rays``.
    """
    rc, cc, radius = circle
    smooth = cv2.GaussianBlur(image, (0, 0), cfg.outer_blur_sigma)
    step = cfg.outer_step_px
    r_lo = max(0.0, radius - max(cfg.outer_inner_min, cfg.outer_inner_frac * radius))
    r_hi = radius + max(cfg.outer_reach_min, cfg.outer_reach_frac * radius)
    rs = np.arange(r_lo, r_hi + step, step)
    th = np.linspace(0, 2 * np.pi, cfg.outer_rays, endpoint=False)
    prof, ok = _sample_rays(smooth, rc, cc, th, rs, exclude)
    ok = np.cumprod(ok, axis=1).astype(bool)          # contiguous from the inner start
    prof = np.where(ok, prof, np.nan)
    dark_cols = np.nonzero((rs >= radius - max(cfg.outer_dark_min, cfg.outer_dark_frac * radius))
                           & (rs <= radius - cfg.outer_dark_gap))[0]
    bg_cols = np.nonzero((rs >= radius + max(cfg.outer_bg_min, cfg.outer_bg_frac * radius)) & (rs <= r_hi))[0]
    dark, ok_d = _row_percentile(prof, dark_cols, cfg.outer_dark_pct, cfg.outer_min_dark_samples)
    bg, ok_b = _row_percentile(prof, bg_cols, cfg.outer_bg_pct, cfg.outer_min_bg_samples)
    use = ok_d & ok_b
    use[use] = (bg[use] - dark[use]) >= cfg.outer_min_contrast
    if not use.any():
        return np.empty((0, 2)), np.empty(0), np.empty(0, int)
    p, th_u, d, bgl = prof[use], th[use], dark[use], bg[use]
    l50 = (d + 0.5 * (bgl - d))[:, None]
    l90 = (d + cfg.outer_level * (bgl - d))[:, None]
    fin = np.isfinite(p)
    with np.errstate(invalid="ignore"):
        up = fin[:, :-1] & fin[:, 1:] & (p[:, :-1] < l50) & (p[:, 1:] >= l50)
    has_up = up.any(1)
    j = np.where(up, np.abs(rs[:-1] - radius)[None, :], np.inf).argmin(1)
    idx = np.arange(len(rs))[None, :]
    with np.errstate(invalid="ignore"):
        reach = fin & (p >= l90) & (idx >= j[:, None])
    has_b = reach.any(1)
    kk = np.argmax(reach, axis=1)
    rows = np.arange(len(p))
    km1 = np.maximum(kk - 1, 0)
    pk, pk1 = p[rows, kk], p[rows, km1]
    interp = rs[km1] + (l90[:, 0] - pk1) / np.maximum(pk - pk1, 1e-9) * step
    r_b = np.where(kk > j, interp, rs[kk])
    keep = has_up & has_b & np.isfinite(r_b)
    th_k, r_k = th_u[keep], r_b[keep]
    return (np.column_stack([rc + np.sin(th_k) * r_k, cc + np.cos(th_k) * r_k]), l90[keep, 0],
            np.nonzero(use)[0][keep])
