"""Fit's live view draws and names a frame as the finished result does: the same
circle name, the measured outline, and in Auto mode the Auto view on the right
(not a mask under the Auto view's title)."""

from __future__ import annotations

import numpy as np
import pytest

from bubbletrack.controller import edge_support
from bubbletrack.controller.display_mixin import ENGINE_VIEW_TITLE, MASK_TITLE
from bubbletrack.controller.live_view import show_live_frame
from bubbletrack.model.provenance import FitSource
from bubbletrack.model.state import update_state
from conftest import write_sequence

RADII = [0.0, 0.0, 30.0, 40.0, 50.0]  # frames 1-2: no bubble (references)


@pytest.fixture
def auto_app(gui, tmp_path):
    window, controller = gui
    controller.file_ctrl.on_folder_selected(str(write_sequence(tmp_path / "seq", RADII)))
    controller._state = update_state(controller._get_state(), detection_mode="edge")
    return window, controller


def _pixel(panel, x: int, y: int) -> int:
    return panel._pixmap_item.pixmap().toImage().pixelColor(x, y).red()


def test_auto_live_frame_looks_like_the_result(auto_app):
    window, controller = auto_app
    state = controller._get_state()
    fit = edge_support.fit_frame(state, 3, 500.0)
    assert fit.radius > 0
    edge_support.store_fit(state, 3, fit, FitSource.AUTO)
    window.binary_panel.set_title(MASK_TITLE)  # e.g. from an earlier classic view
    mask = np.ones((250, 400), dtype=bool)  # what the worker sends: never shown in Auto

    show_live_frame(state, window, 3, mask)

    assert window.original_panel.legend_texts() == ["outline (measured)", "fitted circle",
                                                    "ROI"]
    right = window.binary_panel
    assert right._title_label.text() == ENGINE_VIEW_TITLE
    assert right.legend_texts() == ["outline (measured)"]
    assert _pixel(right, 5, 5) < 250  # the grey Auto view, not the white mask


def test_auto_frame_without_a_result_draws_no_circle(auto_app):
    window, controller = auto_app
    state = controller._get_state()
    state.radius[0] = -1.0
    show_live_frame(state, window, 0, None)
    assert window.original_panel.legend_texts() == ["ROI"]
    assert window.binary_panel._title_label.text() == ENGINE_VIEW_TITLE


def test_classic_live_frame_shows_the_mask_and_a_fitted_circle(gui, make_sequence):
    window, controller = gui
    controller.file_ctrl.on_folder_selected(str(make_sequence("seq", 4)))
    state = update_state(controller._get_state(), detection_mode="intensity")
    controller._state = state
    t = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    edge = np.round(np.column_stack([125 + 30 * np.sin(t), 200 + 30 * np.cos(t)]))
    state.radius[2] = 30.0
    state.circle_fit_par[2] = [125.0, 200.0]
    state.circle_xy[2] = edge[np.lexsort((edge[:, 1], edge[:, 0]))]  # raster order: points
    state.fit_source[2] = FitSource.AUTO
    mask = np.zeros((250, 400), dtype=bool)
    mask[100:150, 175:225] = True

    show_live_frame(state, window, 2, mask)

    assert window.original_panel.legend_texts() == ["edge points", "fitted circle", "ROI"]
    assert window.binary_panel._title_label.text() == MASK_TITLE
    assert _pixel(window.binary_panel, 200, 125) == 255 and _pixel(window.binary_panel, 5, 5) == 0
