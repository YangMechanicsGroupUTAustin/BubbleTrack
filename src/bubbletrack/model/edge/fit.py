"""Stage D: robust circle fitting, covariance, arc statistics and polygon area.

Points are ``(N, 2)`` arrays of ``[row, col]`` in any consistent frame.  The
robust fit initialises from the better of a Taubin fit and a RANSAC circle,
uses Tukey-weighted IRLS only to decide who is an outlier, then reports an
unweighted geometric least-squares fit on the inliers (Tukey weights would
bias non-circular shapes towards their mid radius).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bubbletrack.model.circle_fit import circle_fit_taubin
from bubbletrack.model.edge.config import FitConfig

_TINY = 1e-12


@dataclass(frozen=True)
class CircleFit:
    """A fitted circle with its uncertainty and per-point inlier weights."""

    rc: float
    cc: float
    radius: float
    cov: np.ndarray
    """3x3 covariance of (rc, cc, radius); NaN when unavailable."""
    weights: np.ndarray
    """Per input point; > ``FitConfig.inlier_weight`` marks an inlier."""
    method: str

    @property
    def circle(self) -> tuple[float, float, float]:
        return (self.rc, self.cc, self.radius)

    @property
    def radius_sigma(self) -> float:
        var = float(self.cov[2, 2])
        return float(np.sqrt(var)) if np.isfinite(var) and var > 0 else float("nan")


def _nan_cov() -> np.ndarray:
    return np.full((3, 3), np.nan)


def ransac_circle(p: np.ndarray, rng: np.random.Generator,
                  cfg: FitConfig) -> tuple[float, float, float] | None:
    """Best 3-point circle by inlier count (vectorised RANSAC)."""
    n = len(p)
    if n < 3:
        return None
    extent = float(np.hypot(*(p.max(0) - p.min(0))))
    r_max = cfg.ransac_rmax_factor * extent + cfg.ransac_rmax_pad
    idx = rng.integers(0, n, size=(cfg.ransac_iterations, 3))
    q0, q1, q2 = p[idx[:, 0]], p[idx[:, 1]], p[idx[:, 2]]
    a11, a12 = 2 * (q1[:, 0] - q0[:, 0]), 2 * (q1[:, 1] - q0[:, 1])
    a21, a22 = 2 * (q2[:, 0] - q0[:, 0]), 2 * (q2[:, 1] - q0[:, 1])
    b1 = (q1 ** 2).sum(1) - (q0 ** 2).sum(1)
    b2 = (q2 ** 2).sum(1) - (q0 ** 2).sum(1)
    det = a11 * a22 - a12 * a21
    ok = np.abs(det) > 1e-9
    det = np.where(ok, det, 1.0)
    cr = (b1 * a22 - a12 * b2) / det
    ccol = (a11 * b2 - b1 * a21) / det
    r = np.hypot(q0[:, 0] - cr, q0[:, 1] - ccol)
    ok &= (r < r_max) & (r > 1.0)
    if not ok.any():
        return None
    cr, ccol, r = cr[ok], ccol[ok], r[ok]
    d = np.hypot(p[None, :, 0] - cr[:, None], p[None, :, 1] - ccol[:, None])
    count = (np.abs(d - r[:, None]) < cfg.ransac_tol_px).sum(1)
    i = int(np.argmax(count))
    return float(cr[i]), float(ccol[i]), float(r[i])


def irls_circle(p: np.ndarray, init: tuple[float, float, float], cfg: FitConfig, *,
                robust: bool = True,
                n_iter: int | None = None) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    """Gauss-Newton geometric circle fit, optionally Tukey-reweighted.

    Returns
    -------
    (rc, cc, radius, cov, weights)
    """
    rc, cc, radius = init
    w = np.ones(len(p))
    for _ in range(cfg.irls_iterations if n_iter is None else n_iter):
        dr, dc = p[:, 0] - rc, p[:, 1] - cc
        d = np.hypot(dr, dc) + _TINY
        res = d - radius
        if robust:
            scale = max(1.4826 * np.median(np.abs(res - np.median(res))), cfg.min_scale_px)
            u = res / (cfg.tukey_c * scale)
            w = np.where(np.abs(u) < 1, (1 - u ** 2) ** 2, 0.0)
        if w.sum() < 3:
            break
        jac = np.column_stack([-dr / d, -dc / d, -np.ones_like(d)])
        jw = jac * w[:, None]
        try:
            step = np.linalg.solve(jac.T @ jw + 1e-9 * np.eye(3), -(jw.T @ res))
        except np.linalg.LinAlgError:
            break
        rc, cc, radius = rc + step[0], cc + step[1], radius + step[2]
        if np.abs(step).max() < cfg.converge_tol:
            break
    dr, dc = p[:, 0] - rc, p[:, 1] - cc
    d = np.hypot(dr, dc) + _TINY
    res = d - radius
    jac = np.column_stack([-dr / d, -dc / d, -np.ones_like(d)])
    cov = _nan_cov()
    m = w > 0
    if m.sum() > 3:
        s2 = np.sum(w[m] * res[m] ** 2) / max(np.sum(w[m]) - 3, 1)
        try:
            cov = s2 * np.linalg.inv((jac * w[:, None]).T @ jac)
        except np.linalg.LinAlgError:
            pass
    return float(rc), float(cc), float(radius), cov, w


def taubin_cov(p: np.ndarray, rc: float, cc: float, radius: float) -> np.ndarray:
    """Covariance of an unweighted geometric fit evaluated at a given circle."""
    d = np.hypot(p[:, 0] - rc, p[:, 1] - cc) + _TINY
    res = d - radius
    jac = np.column_stack([-(p[:, 0] - rc) / d, -(p[:, 1] - cc) / d, -np.ones(len(p))])
    s2 = np.sum(res ** 2) / max(len(p) - 3, 1)
    try:
        return s2 * np.linalg.inv(jac.T @ jac)
    except np.linalg.LinAlgError:
        return _nan_cov()


def _inlier_count(p: np.ndarray, c: tuple[float, float, float], tol: float) -> int:
    return int(np.sum(np.abs(np.hypot(p[:, 0] - c[0], p[:, 1] - c[1]) - c[2]) < tol))


def fit_circle(p: np.ndarray, rng: np.random.Generator, cfg: FitConfig, *,
               robust: bool = True) -> CircleFit | None:
    """Fit a circle to edge points, robustly unless *robust* is False or points are few.

    Parameters
    ----------
    p : ndarray of shape (N, 2)
    rng : numpy Generator
        Source of the RANSAC samples (seed it for reproducible results).
    cfg : FitConfig
    robust : bool
        False forces a plain Taubin fit.

    Returns
    -------
    CircleFit or None
        None when fewer than 3 points or every fit failed.
    """
    if len(p) < 3:
        return None
    tau = tuple(float(v) for v in circle_fit_taubin(p))
    if not robust or len(p) < cfg.robust_min_pts:
        if not np.isfinite(tau[2]):
            return None
        return CircleFit(*tau, taubin_cov(p, *tau), np.ones(len(p)), "taubin")
    rs = ransac_circle(p, rng, cfg)
    cands = [c for c in (tau, rs) if c is not None and np.isfinite(c[2])]
    if not cands:
        return None
    init = max(cands, key=lambda c: _inlier_count(p, c, cfg.ransac_tol_px))
    rc, cc, radius, cov, w = irls_circle(p, init, cfg)
    jump = np.hypot(rc - init[0], cc - init[1])
    sane = (np.isfinite(radius) and radius > 0
            and jump < max(cfg.centre_jump_px, cfg.centre_jump_frac * init[2])
            and init[2] / cfg.radius_ratio_max < radius < cfg.radius_ratio_max * init[2])
    if not sane:
        if not init[2] > 0:
            return None
        res = np.abs(np.hypot(p[:, 0] - init[0], p[:, 1] - init[1]) - init[2])
        weights = (res < cfg.ransac_tol_px).astype(float)
        return CircleFit(*init, _nan_cov(), weights, "init(irls diverged)")
    inl = w > 0
    if inl.sum() >= cfg.lsq_min_inliers:
        rc2, cc2, r2, cov2, _ = irls_circle(p[inl], (rc, cc, radius), cfg, robust=False,
                                            n_iter=cfg.lsq_iterations)
        if np.isfinite(r2) and r2 > 0:
            return CircleFit(rc2, cc2, r2, cov2, inl.astype(float), "ransac/taubin+tukey+lsq")
    return CircleFit(rc, cc, radius, cov, w, "ransac/taubin+tukey")


def arc_stats(p: np.ndarray, rc: float, cc: float, nbins: int = 36) -> tuple[float, float]:
    """Angular coverage (fraction of occupied bins) and largest angular gap (deg)."""
    if len(p) == 0:
        return 0.0, 360.0
    ang = np.degrees(np.arctan2(p[:, 0] - rc, p[:, 1] - cc)) % 360
    coverage = len(np.unique((ang // (360 / nbins)).astype(int))) / nbins
    a = np.sort(ang)
    gaps = np.diff(np.concatenate([a, [a[0] + 360]]))
    return float(coverage), float(gaps.max())


def polygon_area(p: np.ndarray, rc: float, cc: float) -> float:
    """Area of the polygon through *p* ordered by angle about (rc, cc) (shoelace)."""
    if len(p) < 3:
        return 0.0
    ang = np.arctan2(p[:, 0] - rc, p[:, 1] - cc)
    order = np.argsort(ang)
    y, x = p[order, 0], p[order, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
