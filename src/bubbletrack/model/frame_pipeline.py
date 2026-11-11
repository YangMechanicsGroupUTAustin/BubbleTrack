"""Per-frame detection + measurement, as run by the GUI's Fit All.

This is the model-layer twin of ``controller/worker.py::_process_single_frame``
(load, threshold, detect with the chosen mode, measure, sanity-filter the
radius).  The CLI uses it so headless runs give the GUI's results, and
``tests/test_frame_pipeline.py`` pins the two to identical output.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from bubbletrack.model.detection import detect_bubble_with_mode, measure_detection
from bubbletrack.model.image_io import load_and_normalize


@dataclass(frozen=True)
class FrameParams:
    """Settings for fitting one frame (the GUI's Pre-tune values).

    ``removing_factor`` is the minimum blob area in pixels, i.e. the slider
    value already mapped by ``compute_removing_factor``.  ``max_radius`` is
    the largest plausible radius (the GUI uses the image's longer side).
    """

    sensitivity: float
    gridx: tuple[int, int]
    gridy: tuple[int, int]
    removing_factor: int
    edges: tuple[bool, bool, bool, bool] = (False, False, False, False)
    mode: str = "intensity"
    removing_obj_radius: int = 0
    gaussian_sigma: float = 0.0
    clahe_clip: float = 0.0
    closing_radius: int = 0
    opening_radius: int = 0
    gradient_boundary: float = 0.0
    area_max_regions: int = 1
    max_radius: float = math.inf


@dataclass(frozen=True)
class FrameFit:
    """Result for one frame; ``radius`` is -1.0 when nothing plausible was found."""

    radius: float
    center_row: float
    center_col: float
    region_count: int
    edge_xy: np.ndarray
    mask: np.ndarray

    @property
    def fitted(self) -> bool:
        return self.radius > 0


def is_plausible_radius(radius: float, max_radius: float) -> bool:
    """True if *radius* is finite, positive and at most *max_radius*."""
    return bool(np.isfinite(radius)) and 0 < radius <= max_radius


def fit_rois(
    binary_roi: np.ndarray,
    gray_roi: np.ndarray,
    params: FrameParams,
) -> FrameFit:
    """Detect and measure the bubble in already-normalised ROI images.

    Parameters
    ----------
    binary_roi : ndarray of bool
        Adaptive-threshold ROI from ``normalize_frame``.
    gray_roi : ndarray
        Normalised grey ROI in [0, 1] from ``normalize_frame``.
    params : FrameParams
        Detection settings.

    Returns
    -------
    FrameFit
        An implausible radius (non-finite, <= 0, > ``max_radius``) is
        reported as -1.0 with no edge points and a region count of 0; the
        centre and mask are kept, as in the GUI.
    """
    mask, edge_xy = detect_bubble_with_mode(
        binary_roi, gray_roi, params.mode, params.sensitivity,
        list(params.edges), params.removing_factor, params.gridx, params.gridy,
        params.removing_obj_radius,
        opening_radius=params.opening_radius,
        closing_radius=params.closing_radius,
        gaussian_sigma=params.gaussian_sigma,
        gradient_boundary=params.gradient_boundary,
        area_max_regions=params.area_max_regions,
    )
    radius, rc, cc, count = measure_detection(
        mask, edge_xy, params.mode, params.gridx, params.gridy,
    )
    if not is_plausible_radius(radius, params.max_radius):
        radius, edge_xy, count = -1.0, np.empty((0, 2)), 0
    return FrameFit(
        radius=float(radius), center_row=float(rc), center_col=float(cc),
        region_count=int(count), edge_xy=edge_xy, mask=mask,
    )


def fit_image_file(path: str, params: FrameParams) -> FrameFit:
    """Load *path* and fit it (see :func:`fit_rois`).

    Raises
    ------
    ImageLoadError
        If the file is missing or cannot be decoded.
    """
    _, _, gray_roi, binary_roi = load_and_normalize(
        path, params.sensitivity, params.gridx, params.gridy,
        gaussian_sigma=params.gaussian_sigma, clahe_clip=params.clahe_clip,
    )
    return fit_rois(binary_roi, gray_roi, params)
