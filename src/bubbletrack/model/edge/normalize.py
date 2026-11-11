"""Stage A: grey conversion, saturation mask, background surface, normalisation, noise.

The normalised image ``N`` has the background at 1.0 and a dark bubble below
it.  With a bubble-free reference frame, ``N = I / (gain * B_ref)`` corrected
by a smooth residual surface; without one, ``N = I / B_hat`` with ``B_hat`` a
robust polynomial surface plus a masked local correction.  All work is done in
float32 on the full-bit-depth data.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from bubbletrack.model.constants import UINT16_PACKED_DEPTHS
from bubbletrack.model.edge.config import NormalizeConfig
from bubbletrack.model.image_io import ImageLoadError, to_grayscale

_EPS = 1e-6
_SURFACE_SCALE = 1000.0  # brings a normalised image (~1.0) to camera-count magnitude


@dataclass(frozen=True)
class GrayFrame:
    """A frame as float32 grey values, with what is needed to judge saturation."""

    values: np.ndarray
    """2-D float32 grey values; non-finite inputs are replaced by the finite median."""
    full_scale: float
    """Nominal saturation level of the camera data (e.g. 65535, 4095, 255, 1.0)."""
    non_finite: np.ndarray
    """Bool mask of pixels that were NaN or inf in the input."""


@dataclass(frozen=True)
class Normalized:
    """Output of stage A for one ROI."""

    image: np.ndarray
    """Normalised float32 image, background = 1.0, invalid pixels = 1.0."""
    saturated: np.ndarray
    """Saturated blobs of the frame (bool)."""
    invalid: np.ndarray
    """Pixels without usable data: dark or saturated in the reference, or non-finite."""
    method: str
    gain: float


def to_gray_frame(img: np.ndarray) -> GrayFrame:
    """Convert any supported frame to float32 grey without losing bit depth.

    Parameters
    ----------
    img : ndarray
        2-D grey, or 3-D with 1-4 channels (BGR / BGRA as OpenCV reads them),
        of any integer or float dtype.  It is not modified.

    Returns
    -------
    GrayFrame

    Raises
    ------
    ValueError
        For an empty array, a non-numeric dtype or an unsupported layout.
    """
    if not isinstance(img, np.ndarray) or img.size == 0:
        raise ValueError("frame must be a non-empty numpy array")
    if img.ndim not in (2, 3):
        raise ValueError(f"frame must be 2-D or 3-D, got shape {img.shape}")
    if img.dtype == np.bool_ or not (np.issubdtype(img.dtype, np.integer)
                                     or np.issubdtype(img.dtype, np.floating)):
        raise ValueError(f"unsupported frame dtype {img.dtype}")
    try:
        gray = to_grayscale(img)
    except ImageLoadError as exc:
        raise ValueError(str(exc)) from exc
    values = gray.astype(np.float32)          # always a new array
    non_finite = ~np.isfinite(values)
    if non_finite.any():
        finite = values[~non_finite]
        fill = float(np.median(finite)) if finite.size else 0.0
        values[non_finite] = fill
    return GrayFrame(values=values, full_scale=full_scale(gray), non_finite=non_finite)


def full_scale(gray: np.ndarray) -> float:
    """Nominal saturation level of a grey frame, from its dtype and peak.

    uint16 frames whose peak fits in 10 or 12 bits are treated as packed
    10/12-bit camera data (same rule as ``image_io.to_threshold_uint8``).
    Float frames are [0, 1] intensities when their peak is <= 1, else counts.
    """
    if np.issubdtype(gray.dtype, np.integer):
        top = int(np.iinfo(gray.dtype).max)
        if gray.dtype == np.uint16:
            peak = int(gray.max())
            for max_value, _ in UINT16_PACKED_DEPTHS:
                if peak <= max_value:
                    return float(max_value)
        return float(top)
    finite = gray[np.isfinite(gray)]
    peak = float(finite.max()) if finite.size else 1.0
    if peak <= 1.0:
        return 1.0
    if peak <= 65535.0:
        return 65535.0
    return peak


def saturation_mask(values: np.ndarray, full: float, cfg: NormalizeConfig) -> np.ndarray:
    """Saturated blobs: pixels at the clipped top level, in blobs of >= ``sat_min_blob_px``.

    Isolated saturated grains (dense bright speckle) are not glare and are
    left out: masking them blanks large parts of a speckled frame.
    """
    top = float(values.max())
    if top < cfg.sat_peak_fraction * full:
        return np.zeros(values.shape, bool)
    hot = values >= top - full * cfg.sat_margin_fraction
    if int(hot.sum()) < cfg.sat_min_pixels:
        return np.zeros(values.shape, bool)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(hot.astype(np.uint8), connectivity=8)
    big = np.zeros(n, bool)
    big[1:] = stats[1:, cv2.CC_STAT_AREA] >= cfg.sat_min_blob_px
    return big[labels]


def _poly_terms(h: int, w: int, degree: int) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    y = (yy / max(h - 1, 1)) * 2 - 1
    x = (xx / max(w - 1, 1)) * 2 - 1
    cols = [x ** i * y ** j for i in range(degree + 1) for j in range(degree + 1 - i)]
    return np.stack([c.ravel() for c in cols], axis=1)


def estimate_background(values: np.ndarray, exclude: np.ndarray, cfg: NormalizeConfig) -> np.ndarray:
    """Smooth background surface of *values* (bubble, debris and glare excluded).

    A cubic polynomial is fitted on a shrunk image with iterative rejection of
    dark and bright outliers, then multiplied by a masked normalised-convolution
    correction that captures medium-scale illumination the cubic misses.

    Parameters
    ----------
    values : ndarray
        2-D float image.  It is not modified.
    exclude : ndarray of bool
        Pixels never used for the fit (saturated or invalid).
    cfg : NormalizeConfig

    Returns
    -------
    ndarray
        float32 surface of the same shape.
    """
    f = cfg.bg_downsample
    h, w = values.shape
    hs, ws = max(h // f, 4), max(w // f, 4)
    small = cv2.resize(values.astype(np.float32), (ws, hs), interpolation=cv2.INTER_AREA)
    ex_s = cv2.resize(exclude.astype(np.float32), (ws, hs), interpolation=cv2.INTER_AREA) > 0
    terms = _poly_terms(hs, ws, cfg.bg_poly_degree)
    y = small.ravel()
    keep = ~ex_s.ravel()
    if not keep.any():
        keep = np.ones_like(keep)
    med = float(np.median(y[keep]))
    keep &= y > cfg.bg_dark_start * med
    surface = np.full(small.shape, med, np.float32)
    k3 = np.ones((3, 3), np.uint8)
    for _ in range(cfg.bg_iterations):
        if keep.sum() < terms.shape[1] * 5:
            break
        coef, *_ = np.linalg.lstsq(terms[keep], y[keep], rcond=None)
        surface = (terms @ coef).reshape(small.shape).astype(np.float32)
        res = small / np.maximum(surface, _EPS) - 1.0
        r_in = res.ravel()[keep]
        sig = 1.4826 * float(np.median(np.abs(r_in - np.median(r_in))))
        bad = ((res < -max(cfg.bg_dark_k * sig, cfg.bg_dark_floor))
               | (res > max(cfg.bg_bright_k * sig, cfg.bg_bright_floor)) | ex_s)
        bad = cv2.dilate(bad.astype(np.uint8), k3) > 0
        keep = ~bad.ravel()
    ratio = small / np.maximum(surface, _EPS)
    wgt = keep.reshape(small.shape).astype(np.float32)
    sig_s = cfg.bg_sigma / f
    num = cv2.GaussianBlur(ratio * wgt, (0, 0), sig_s, borderType=cv2.BORDER_REFLECT)
    den = cv2.GaussianBlur(wgt, (0, 0), sig_s, borderType=cv2.BORDER_REFLECT)
    ok = den > cfg.bg_min_weight
    corr = np.where(ok, num / np.maximum(den, _EPS), 1.0).astype(np.float32)
    surface = surface * corr
    return cv2.resize(surface, (w, h), interpolation=cv2.INTER_LINEAR)


def normalize_roi(values: np.ndarray, saturated: np.ndarray, cfg: NormalizeConfig, *,
                  reference: np.ndarray | None = None,
                  ref_invalid: np.ndarray | None = None,
                  invalid: np.ndarray | None = None) -> Normalized:
    """Normalise an ROI so that its background is 1.0.

    Parameters
    ----------
    values : ndarray
        float32 ROI of the frame.  It is not modified.
    saturated : ndarray of bool
        Saturated pixels to keep out of every estimate.
    cfg : NormalizeConfig
    reference : ndarray, optional
        float32 ROI of a bubble-free reference frame.  None selects the
        reference-free surface normalisation.
    ref_invalid : ndarray of bool, optional
        Pixels of the reference without usable data (saturated, non-finite).
    invalid : ndarray of bool, optional
        Pixels of the frame without usable data (non-finite).

    Returns
    -------
    Normalized
    """
    invalid = np.zeros(values.shape, bool) if invalid is None else invalid.copy()
    if reference is None:
        surface = estimate_background(values, saturated | invalid, cfg)
        image = values / np.maximum(surface, _EPS)
        image = np.where(invalid, 1.0, image).astype(np.float32)
        method = f"surface(cubic+normconv {cfg.bg_sigma:g})" + _no_background_note(saturated | invalid)
        return Normalized(image, saturated, invalid, method, 1.0)
    blurred = cv2.GaussianBlur(reference, (0, 0), cfg.ref_blur_sigma)
    med = float(np.median(blurred))
    dead = blurred < cfg.ref_invalid_level * med
    if ref_invalid is not None:
        dead |= ref_invalid
    k = cfg.ref_invalid_dilate_px
    invalid |= cv2.dilate(dead.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
    ratio = values / np.maximum(blurred, _EPS)
    ok = ~invalid & ~saturated
    gain = float(np.median(ratio[ok])) if ok.sum() > cfg.gain_min_pixels else 1.0
    gain = gain if np.isfinite(gain) and gain > _EPS else 1.0
    image = np.where(invalid, 1.0, ratio / gain).astype(np.float32)
    residual = estimate_background(image * _SURFACE_SCALE, saturated | invalid, cfg) / _SURFACE_SCALE
    image = np.where(invalid, 1.0, image / np.maximum(residual, _EPS)).astype(np.float32)
    method = f"reference(gain={gain:.3f})+surface" + _no_background_note(saturated | invalid)
    return Normalized(image, saturated, invalid, method, gain)


def _no_background_note(excluded: np.ndarray) -> str:
    """Provenance note when no pixel of the ROI could serve as background."""
    return " [no usable background pixel]" if excluded.all() else ""


def noise_sigma(image: np.ndarray, exclude: np.ndarray, cfg: NormalizeConfig) -> float:
    """Robust per-pixel noise of a normalised image, from horizontal differences."""
    if image.shape[1] < 2:
        return _EPS
    d = np.diff(image, axis=1)
    m = (~(exclude[:, 1:] | exclude[:, :-1]) & (image[:, 1:] > cfg.noise_bright_level)
         & (image[:, :-1] > cfg.noise_bright_level))
    d = d[m] if m.sum() > cfg.noise_min_pixels else d.ravel()
    return max(float(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2)), _EPS)
