"""The shared per-frame pipeline must match the GUI's Fit All worker exactly."""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from _golden_cases import CASES, render_scene
from bubbletrack.controller.worker import _process_single_frame
from bubbletrack.model.frame_pipeline import (
    FrameParams,
    fit_image_file,
    is_plausible_radius,
)
from bubbletrack.model.removing_factor import compute_removing_factor


def _worker_args(path, case, gridx, gridy, max_radius):
    pipe = case.pipeline
    return (
        0, str(path), pipe.threshold, gridx, gridy,
        list(pipe.edges), pipe.rf_slider, 0,
        pipe.sigma, pipe.clahe,
        pipe.closing, pipe.opening, pipe.mode, pipe.gradient_boundary,
        pipe.area_max_regions,
        max_radius,
    )


def _params(case, gridx, gridy, max_radius) -> FrameParams:
    pipe = case.pipeline
    return FrameParams(
        sensitivity=pipe.threshold,
        gridx=gridx,
        gridy=gridy,
        removing_factor=compute_removing_factor(pipe.rf_slider, gridx, gridy),
        edges=pipe.edges,
        mode=pipe.mode,
        gaussian_sigma=pipe.sigma,
        clahe_clip=pipe.clahe,
        closing_radius=pipe.closing,
        opening_radius=pipe.opening,
        gradient_boundary=pipe.gradient_boundary,
        area_max_regions=pipe.area_max_regions,
        max_radius=max_radius,
    )


def _same_float(a: float, b: float) -> bool:
    return (math.isnan(a) and math.isnan(b)) or a == b


@pytest.mark.parametrize("max_radius_scale", [1.0, 0.1])  # normal, and filter rejecting
@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_matches_gui_worker(case, max_radius_scale, tmp_path):
    frame = render_scene(case.scene)
    path = tmp_path / f"{case.name}.tiff"
    assert cv2.imwrite(str(path), frame)
    h, w = frame.shape[:2]
    gridx, gridy = case.pipeline.roi if case.pipeline.roi else ((1, h), (1, w))
    max_radius = float(max(h, w)) * max_radius_scale

    _, radius, edge_xy, processed, rc, cc, count = _process_single_frame(
        _worker_args(path, case, gridx, gridy, max_radius),
    )
    fit = fit_image_file(str(path), _params(case, gridx, gridy, max_radius))

    assert _same_float(fit.radius, radius)
    assert _same_float(fit.center_row, rc)
    assert _same_float(fit.center_col, cc)
    assert fit.region_count == count
    np.testing.assert_array_equal(fit.edge_xy, np.asarray(edge_xy).reshape(-1, 2))
    np.testing.assert_array_equal(fit.mask, processed)


def test_rejected_fit_reports_not_fitted(tmp_path):
    case = CASES[0]
    frame = render_scene(case.scene)
    path = tmp_path / "f.tiff"
    assert cv2.imwrite(str(path), frame)
    h, w = frame.shape
    good = fit_image_file(str(path), _params(case, (1, h), (1, w), float(max(h, w))))
    bad = fit_image_file(str(path), _params(case, (1, h), (1, w), 5.0))
    assert good.fitted and good.radius > 5.0
    assert not bad.fitted
    assert bad.radius == -1.0
    assert bad.edge_xy.shape == (0, 2)
    assert bad.region_count == 0


@pytest.mark.parametrize(("radius", "expected"), [
    (10.0, True), (100.0, True), (100.5, False), (0.0, False), (-1.0, False),
    (math.nan, False), (math.inf, False),
])
def test_is_plausible_radius(radius, expected):
    assert is_plausible_radius(radius, 100.0) is expected


def test_params_are_immutable():
    params = FrameParams(sensitivity=0.5, gridx=(1, 10), gridy=(1, 10), removing_factor=10)
    with pytest.raises(AttributeError):
        params.sensitivity = 0.6  # type: ignore[misc]
