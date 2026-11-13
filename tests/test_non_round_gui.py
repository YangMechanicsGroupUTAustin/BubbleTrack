""""Bubble may be non-round" in the Auto box's Advanced section, and Review's "Cut by the
image edge" row, driven through the real window offscreen."""

from __future__ import annotations

import numpy as np
import pytest

from bubbletrack.controller.review_controller import install_review
from bubbletrack.model.provenance import AREA_RADIUS_FLAG, FitSource
from bubbletrack.model.state import update_state


def _box(window):
    return window.left_panel.measure_panel.edge_box


def test_the_checkbox_sits_in_the_collapsed_advanced_section(gui):
    window, _ = gui
    box = _box(window)
    check = box._non_round_check
    assert check.text() == "Bubble may be non-round" and not box.non_round()
    assert box.advanced.isAncestorOf(check) and box.advanced._content.isHidden()
    tip = check.toolTip()
    for words in ("Off (default)", "circle", "On:", "same area", "image edge", "Fit again"):
        assert words in tip, words


def test_ticking_it_sets_the_definition_and_asks_to_fit_again(gui, make_sequence):
    window, controller = gui
    controller.file_ctrl.on_folder_selected(str(make_sequence("A", 3)))
    controller._get_state().radius[0] = 30.0  # a result measured with the circle
    _box(window)._non_round_check.setChecked(True)
    assert controller._get_state().bubble_non_round is True
    assert controller._get_state().radius[0] == 30.0  # nothing is refitted
    assert window.header.status_text() == (
        "Bubble may be non-round: on (R from the outline's area) — Fit again to apply")
    _box(window)._non_round_check.setChecked(False)
    assert controller._get_state().bubble_non_round is False
    assert window.header.status_text().startswith("Bubble may be non-round: off")


def test_without_results_there_is_nothing_to_fit_again(gui, make_sequence):
    window, controller = gui
    controller.file_ctrl.on_folder_selected(str(make_sequence("A", 3)))
    _box(window)._non_round_check.setChecked(True)
    assert "Fit again" not in window.header.status_text()


def test_the_checkbox_shows_a_loaded_setting(gui):
    window, controller = gui
    controller._state = update_state(controller._get_state(), bubble_non_round=True)
    controller._sync_ui_from_state()
    assert _box(window).non_round()
    params = window.left_panel.measure_panel.get_filter_params()
    assert params["bubble_non_round"] is True


@pytest.fixture
def review(gui, make_sequence):
    window, controller = gui
    controller.file_ctrl.on_folder_selected(str(make_sequence("A", 10)))
    return window, controller, install_review(window, controller)


def test_review_lists_a_clipped_frame_in_its_own_row(review):
    window, controller, rc = review
    state = controller._get_state()
    state.radius[:] = np.linspace(30.0, 40.0, len(state.radius))
    state.circle_fit_par[:] = [[125.0, 200.0]] * len(state.radius)
    state.fit_source[:] = FitSource.AUTO
    state.fit_confidence[:] = 0.9
    state.fit_flags[2] = frozenset({"clipped", "clipped_not_round"})
    state.fit_flags[6] = frozenset({"non_circular", AREA_RADIUS_FLAG})  # information only
    state.fit_flags[8] = frozenset({"clipped", "clipped_completed", AREA_RADIUS_FLAG})
    rc.refresh()
    assert rc.panel.texts() == ["Frame 3 · Cut by the image edge"]
    rc.panel.step(1)
    advice = rc.panel.card.advice.text()
    assert advice.startswith("Cut by the image edge here. The wall you can see is not round")
