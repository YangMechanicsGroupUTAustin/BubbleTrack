"""The review dock is installed by the app controller, and marking a frame
"no bubble" clears its scoring so it is not mistaken for a rejected detection."""

from __future__ import annotations

import numpy as np


def _load(controller, folder):
    controller.file_ctrl.on_folder_selected(str(folder))


def test_review_dock_installed(gui):
    window, controller = gui
    assert controller.review is not None
    assert window.findChild(type(controller.review.dock), "reviewDock") is not None or \
        controller.review.dock.parent() is window


def test_no_bubble_clears_confidence_flags_metrics(gui, make_sequence):
    _, controller = gui
    _load(controller, make_sequence("A", 4))
    state = controller._get_state()
    state.radius[1] = 30.0
    state.fit_confidence[1] = 0.2
    state.fit_flags[1] = frozenset({"non_circular"})
    state.fit_metrics["r_circle"][1] = 31.0
    controller._on_mark_no_bubble(1, 1)
    state = controller._get_state()
    assert np.isnan(state.fit_confidence[1])
    assert state.fit_flags[1] == frozenset()
    assert np.isnan(state.fit_metrics["r_circle"][1])
    assert all(item.frame != 1 for item in
               __import__("bubbletrack.controller.review_controller",
                          fromlist=["x"]).build_review_items(state))
