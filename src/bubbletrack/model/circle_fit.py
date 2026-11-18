"""Taubin circle fitting algorithm.

G. Taubin, "Estimation Of Planar Curves, Surfaces And Nonplanar Space
Curves Defined By Implicit Equations, With Applications To Edge And
Range Image Segmentation", IEEE Trans. PAMI, Vol. 13, pp 1115-1138, 1991.
"""

from __future__ import annotations

import numpy as np

from bubbletrack.model.constants import (
    TAUBIN_EPSILON,
    TAUBIN_MAX_ITER,
    TAUBIN_MIN_POINTS,
)


def circle_fit_taubin(xy: np.ndarray) -> tuple[float, float, float]:
    """Fit a circle using the Taubin algebraic method.

    Parameters
    ----------
    xy : ndarray of shape (n, 2)
        Coordinates ``[row, col]`` of boundary points.

    Returns
    -------
    (row_c, col_c, radius) : tuple of float
        Circle centre and radius.  All ``NaN`` when *n* < 3 or fitting fails.
    """
    nan3 = (np.nan, np.nan, np.nan)

    if xy.shape[0] < TAUBIN_MIN_POINTS:
        return nan3

    centroid = xy.mean(axis=0)
    xyc = xy - centroid
    zi = np.sum(xyc ** 2, axis=1)

    n = len(xy)
    mxx = np.sum(xyc[:, 0] ** 2) / n
    myy = np.sum(xyc[:, 1] ** 2) / n
    mxy = np.sum(xyc[:, 0] * xyc[:, 1]) / n
    mxz = np.sum(xyc[:, 0] * zi) / n
    myz = np.sum(xyc[:, 1] * zi) / n
    mzz = np.sum(zi ** 2) / n

    # Characteristic polynomial coefficients
    mz = mxx + myy
    cov_xy = mxx * myy - mxy * mxy
    a3 = 4.0 * mz
    a2 = -3.0 * mz * mz - mzz
    a1 = mzz * mz + 4.0 * cov_xy * mz - mxz * mxz - myz * myz - mz ** 3
    a0 = (mxz * mxz * myy + myz * myz * mxx
          - mzz * cov_xy - 2.0 * mxz * myz * mxy
          + mz * mz * cov_xy)
    a22 = 2.0 * a2
    a33 = 3.0 * a3

    # Newton's method starting at x = 0
    xnew = 0.0
    ynew = 1e20
    epsilon = TAUBIN_EPSILON
    iter_max = TAUBIN_MAX_ITER

    for _ in range(iter_max):
        yold = ynew
        ynew = a0 + xnew * (a1 + xnew * (a2 + xnew * a3))
        if abs(ynew) > abs(yold):
            xnew = 0.0
            break
        dy = a1 + xnew * (a22 + xnew * a33)
        xold = xnew
        xnew = xold - ynew / dy
        if abs((xnew - xold) / (xnew + np.finfo(float).eps)) < epsilon:
            break
        if xnew < 0.0:
            xnew = 0.0

    # Circle parameters
    det = xnew * xnew - xnew * mz + cov_xy
    if abs(det) < 1e-30:
        return nan3

    center_x = (mxz * (myy - xnew) - myz * mxy) / det / 2.0
    center_y = (myz * (mxx - xnew) - mxz * mxy) / det / 2.0

    row_c = center_x + centroid[0]
    col_c = center_y + centroid[1]
    radius = np.sqrt(center_x ** 2 + center_y ** 2 + mz)

    return (row_c, col_c, radius)
