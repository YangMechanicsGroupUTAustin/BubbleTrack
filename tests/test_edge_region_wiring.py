"""The app stores the region layer's answer: area-equivalent radius (with "Bubble may
be non-round"), shape metrics, flags and the true outline (drawn as a closed polyline)."""

from __future__ import annotations

import numpy as np
import pytest

from _edge_shapes import SHAPES, reference, render_shape
from bubbletrack.controller.edge_support import store_fit
from bubbletrack.model.edge_pipeline import EdgeJob, fit_edge, make_params
from bubbletrack.model.provenance import FitSource
from bubbletrack.model.state import AppState
from bubbletrack.ui.image_panel import is_ordered_contour

KIDNEY = next(s for s in SHAPES if s.name == "kidney")
PEANUT = next(s for s in SHAPES if s.name == "peanut")


def _stored(shape):
    image, truth = render_shape(shape)
    ref = reference()
    h, w = image.shape
    job = EdgeJob(make_params(non_round=True), ref, max_radius=float(max(h, w)))
    state = AppState().with_results_initialized(1)
    store_fit(state, 0, fit_edge(image, (1, h), (1, w), job), FitSource.AUTO)
    return state, truth


@pytest.mark.parametrize("shape", [KIDNEY, PEANUT], ids=lambda s: s.name)
def test_concave_bubble_reports_its_area_radius(shape):
    state, truth = _stored(shape)
    assert state.radius[0] == pytest.approx(truth, rel=0.01)
    assert state.fit_metrics["area_px"][0] == pytest.approx(np.pi * truth ** 2, rel=0.02)
    assert state.fit_metrics["circularity"][0] < 0.95
    assert abs(state.fit_metrics["r_circle"][0] - truth) > 0.05 * truth  # the circle is off
    assert "non_circular" in state.fit_flags[0]
    assert is_ordered_contour(state.circle_xy[0])  # outline, not a point cloud
