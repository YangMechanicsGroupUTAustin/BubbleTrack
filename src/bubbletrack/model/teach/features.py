"""Per-pixel features of an ROI: a fixed multiscale filter bank (no user parameters).

Channels:

* ``raw``: the grey frame divided by its robust level (95th percentile), so a
  global brightness change (flicker) does not change the features;
* ``view``: the engine's own normalised view (:func:`engine_view`): frame /
  reference when a reference exists, else the reference-free surface
  normalisation; background ~1, bubble ~0.

Each channel gives, per scale, the Gaussian-smoothed intensity, its gradient
magnitude and the two Hessian eigenvalues (``skimage.feature.multiscale_basic_features``).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from bubbletrack.model.skimage_compat import multiscale_basic_features_compat

from bubbletrack.model.conventions import clamp_roi, roi_to_slice
from bubbletrack.model.edge import EdgeParams
from bubbletrack.model.edge.engine import engine_view
from bubbletrack.model.edge.normalize import to_gray_frame
from bubbletrack.model.teach.config import FeatureConfig

_LEVEL_PERCENTILE = 95.0


@dataclass(frozen=True, eq=False)
class FeatureStack:
    values: np.ndarray
    """(h, w, F) float32 features on the (possibly 2x2-binned) ROI grid."""
    step: int
    """1 = full resolution, 2 = half resolution."""
    roi_shape: tuple[int, int]
    """Shape of the full-resolution ROI."""


def _sigmas(cfg: FeatureConfig) -> np.ndarray:
    if cfg.num_sigma == 1:
        return np.array([cfg.sigma_min])
    return np.logspace(np.log2(cfg.sigma_min), np.log2(cfg.sigma_max), num=cfg.num_sigma, base=2)


def feature_names(config: FeatureConfig = FeatureConfig()) -> tuple[str, ...]:
    """Names of the feature columns, in order."""
    kinds = ("int", "edge") + (("tex1", "tex2") if config.texture else ())
    return tuple(f"{ch}_s{s:.3g}_{k}" for ch in config.channels for s in _sigmas(config) for k in kinds)


def roi_bounds(gridx, gridy, shape: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    """Integer, clamped 1-based ROI bounds."""
    gx = (int(gridx[0]), int(gridx[1]))
    gy = (int(gridy[0]), int(gridy[1]))
    return clamp_roi(gx, gy, shape[0], shape[1])


def _raw_channel(gray: np.ndarray, roi: tuple[slice, slice]) -> np.ndarray:
    values = gray[roi]
    level = float(np.percentile(values, _LEVEL_PERCENTILE))
    return (values / max(level, 1e-6)).astype(np.float32)


def _bin2(img: np.ndarray) -> np.ndarray:
    h, w = img.shape
    return cv2.resize(img, ((w + 1) // 2, (h + 1) // 2), interpolation=cv2.INTER_AREA)


def compute_features(raw: np.ndarray, gridx, gridy, *, reference: np.ndarray | None,
                     config: FeatureConfig = FeatureConfig()) -> FeatureStack:
    """Feature stack of the ROI ``gridx`` x ``gridy`` (1-based inclusive) of *raw*.

    Raises ValueError for a reference of another size.
    """
    gray = to_gray_frame(raw).values
    gx, gy = roi_bounds(gridx, gridy, gray.shape)
    roi = roi_to_slice(gx, gy)
    planes = []
    for name in config.channels:
        if name == "raw":
            planes.append(_raw_channel(gray, roi))
        else:
            view = engine_view(raw, gx, gy, reference=reference, params=EdgeParams(polarity=config.polarity))
            planes.append(np.asarray(view, np.float32))
    roi_shape = planes[0].shape
    step = 2 if config.half_resolution else 1
    if step == 2:
        planes = [_bin2(p) for p in planes]
    feats = [multiscale_basic_features_compat(
                 p, intensity=True, edges=True, texture=config.texture,
                 sigma_min=config.sigma_min / step, sigma_max=config.sigma_max / step,
                 num_sigma=config.num_sigma, num_workers=config.num_workers)
             for p in planes]
    values = np.concatenate(feats, axis=-1).astype(np.float32, copy=False)
    return FeatureStack(values=values, step=step, roi_shape=(int(roi_shape[0]), int(roi_shape[1])))
