"""Auto mode shows what the engine sees (frame / reference) in the right panel,
with the measured outline, instead of a threshold mask it never uses."""

from __future__ import annotations

import numpy as np
import pytest

from _golden_cases import SceneSpec, render_scene
from bubbletrack.controller import edge_support
from bubbletrack.controller.display_mixin import (
    ENGINE_VIEW_TITLE, MASK_TITLE, display_frame,
)
from bubbletrack.model.edge.engine import engine_view
from bubbletrack.model.provenance import FitSource
from bubbletrack.model.state import update_state
from bubbletrack.ui.image_compare import CompareMode
from conftest import write_sequence

RADII = [0.0, 0.0, 30.0, 40.0, 50.0]  # 0 = no bubble (frames 1-2 are references)


def test_view_is_flat_background_and_dark_bubble():
    ref = render_scene(SceneSpec(radius=0.0, center=(125.0, 200.0), seed=1))
    frame = render_scene(SceneSpec(radius=40.0, center=(125.0, 200.0), seed=2))
    view = engine_view(frame, (1, 250), (1, 400), reference=ref)
    assert view.shape == (250, 400)
    assert np.median(view[:40, :40]) == pytest.approx(1.0, abs=0.05)   # background
    assert np.median(view[115:135, 190:210]) < 0.2                     # bubble interior


def test_view_respects_the_roi():
    frame = render_scene(SceneSpec(radius=40.0, center=(125.0, 200.0), seed=2))
    assert engine_view(frame, (51, 200), (101, 300), reference=None).shape == (150, 200)


@pytest.fixture
def auto_app(gui, tmp_path):
    window, controller = gui
    controller.file_ctrl.on_folder_selected(str(write_sequence(tmp_path / "seq", RADII)))
    controller._state = update_state(controller._get_state(), detection_mode="edge")
    return window, controller


def _show(window, controller, idx):
    state = controller._get_state()
    display_frame(state, window, idx, controller._set_state, controller._image_cache)


def test_right_panel_shows_the_engine_view_in_auto_mode(auto_app):
    window, controller = auto_app
    assert window.compare_mode is CompareMode.SIDE_BY_SIDE
    _show(window, controller, 3)
    panel = window.binary_panel
    assert panel._title_label.text() == ENGINE_VIEW_TITLE
    assert panel._overlay_items == []  # not fitted yet: no outline
    assert panel.legend_texts() == []  # and no colour key


def test_outline_is_drawn_on_the_view_once_fitted(auto_app):
    window, controller = auto_app
    state = controller._get_state()
    fit = edge_support.fit_frame(state, 3, 500.0)
    assert fit.radius > 0
    edge_support.store_fit(state, 3, fit, FitSource.AUTO)
    _show(window, controller, 3)
    assert len(window.binary_panel._overlay_items) == 1
    assert window.binary_panel.legend_texts() == ["outline (measured)"]
    assert window.original_panel.legend_texts() == ["outline (measured)", "fitted circle", "ROI"]


@pytest.mark.parametrize("non_round, circle", [(False, "fitted circle"), (True, "equal-area circle")])
def test_fit_current_frame_names_its_circle(auto_app, non_round, circle):
    """The default radius is the least-squares circle through the bubble outline; "Bubble may
    be non-round" draws the circle with the outline's area."""
    window, controller = auto_app
    controller._state = update_state(controller._get_state(), bubble_non_round=non_round)
    controller.file_ctrl.on_frame_changed(3)
    controller.pretune_ctrl.on_pretune_fit()
    assert circle in window.original_panel.legend_texts()


def test_overlay_view_names_its_mask_tint(auto_app):
    window, controller = auto_app
    window.compare_toolbar.set_mode(CompareMode.OVERLAY)
    _show(window, controller, 3)
    assert window.original_panel.legend_texts() == ["mask", "ROI"]


def test_other_modes_keep_the_mask_title(auto_app):
    window, controller = auto_app
    _show(window, controller, 3)
    controller._state = update_state(controller._get_state(), detection_mode="intensity")
    _show(window, controller, 3)
    assert window.binary_panel._title_label.text() == MASK_TITLE


@pytest.mark.parametrize("mode", [CompareMode.OVERLAY, CompareMode.WIPE])
def test_overlay_and_wipe_work_in_auto_mode(auto_app, mode):
    window, controller = auto_app
    window.compare_toolbar.set_mode(mode)
    _show(window, controller, 3)  # must not raise


def test_unreadable_frame_does_not_break_the_display(auto_app, monkeypatch):
    window, controller = auto_app

    def boom(state, idx):
        raise OSError("disk gone")
    monkeypatch.setattr(edge_support, "view_frame", boom)
    _show(window, controller, 3)  # falls back to the plain panel
    assert window.binary_panel._title_label.text() == MASK_TITLE
