"""Regression tests for GUI state bugs found in the v3.1 bug hunt.

Each test drives the real MainWindow + AppController offscreen.
"""

from __future__ import annotations

import numpy as np
import pytest

from bubbletrack.ui.pretune_tab import PreTuneTab
from bubbletrack.ui.widgets import SliderInput, ToggleSliderInput


# -- parameter widgets -------------------------------------------------------

class TestSliderRounding:
    def test_saved_threshold_is_shown_unchanged(self, qapp):
        tab = PreTuneTab()
        for value in (0.29, 0.57, 0.58):
            tab.set_threshold(value)
            assert tab.get_threshold() == pytest.approx(value)

    def test_slider_input_rounds_instead_of_truncating(self, qapp):
        slider = SliderInput("x", 0, 100, 50, 1, 0)
        slider.set_value(0.58 * 100)  # 57.99999999999999
        assert slider.value() == 58

    def test_toggle_slider_rounds_instead_of_truncating(self, qapp):
        slider = ToggleSliderInput("x", 0, 10, 1, 0.1, 1)
        slider.set_value(0.29 * 10)  # 2.9000000000000004 -> 2.9, and 0.7*10 edge
        assert slider.value() == pytest.approx(2.9)
        slider.set_value(0.57 * 10)
        assert slider.value() == pytest.approx(5.7)


# -- post-processing parameters ----------------------------------------------

class TestPhysicalParamsReachState:
    def test_panel_edits_update_state(self, gui):
        window, controller = gui
        pp = window.left_panel.post_processing
        pp._fps_unit.setCurrentText("M")
        pp._fps_spin.setValue(5.0)
        pp._scale_spin.setValue(1.25)
        pp._fit_len_spin.setValue(15)
        state = controller._get_state()
        assert state.fps == pytest.approx(5e6)
        assert state.um2px == pytest.approx(1.25)
        assert state.rmax_fit_length == 15

    def test_unit_change_updates_state(self, gui):
        window, controller = gui
        pp = window.left_panel.post_processing
        pp._fps_spin.setValue(2.0)
        pp._fps_unit.setCurrentText("k")
        assert controller._get_state().fps == pytest.approx(2e3)


# -- folder switching / frame navigation ---------------------------------------

def _load(controller, folder):
    controller.file_ctrl.on_folder_selected(str(folder))


class TestFolderSwitch:
    def test_frame_index_resets_when_loading_shorter_folder(self, gui, make_sequence):
        window, controller = gui
        _load(controller, make_sequence("A", 40))
        window.frame_scrubber.set_value(39)
        assert controller._get_state().image_no == 39

        _load(controller, make_sequence("B", 10))
        state = controller._get_state()
        assert state.image_no == 0
        assert window.frame_scrubber.value() == 0

    def test_chart_cleared_when_loading_another_folder(self, gui, make_sequence):
        window, controller = gui
        _load(controller, make_sequence("A", 12))
        state = controller._get_state()
        state.radius[:] = 30.0
        window.radius_chart.plot_all(np.arange(1, 13), state.radius)
        assert len(window.radius_chart._frames) == 12

        _load(controller, make_sequence("B", 5))
        assert len(window.radius_chart._frames) == 0

    def test_fit_range_resets_when_loading_another_folder(self, gui, make_sequence):
        window, controller = gui
        at = window.left_panel.automatic_tab
        _load(controller, make_sequence("A", 40))
        at._start_spin.setValue(30)
        _load(controller, make_sequence("B", 10))
        assert at.get_range() == (1, 10)

    def test_programmatic_frame_change_moves_scrubber(self, gui, make_sequence):
        window, controller = gui
        _load(controller, make_sequence("A", 10))
        controller.file_ctrl.on_frame_changed(6)  # what the arrow-key shortcut calls
        assert window.frame_scrubber.value() == 6

    def test_frame_change_without_folder_is_ignored(self, gui):
        _, controller = gui
        controller.file_ctrl.on_frame_changed(-1)
        controller.file_ctrl.on_frame_changed(5)
        assert controller._get_state().image_no == 0

    def test_out_of_range_frame_is_ignored(self, gui, make_sequence):
        _, controller = gui
        _load(controller, make_sequence("A", 5))
        controller.file_ctrl.on_frame_changed(90)
        assert controller._get_state().image_no == 0


class TestFitAllRange:
    def test_start_after_end_is_rejected(self, gui, make_sequence):
        window, controller = gui
        _load(controller, make_sequence("A", 10))
        at = window.left_panel.automatic_tab
        at._end_spin.setValue(3)
        at._start_spin.setValue(8)
        controller.auto_ctrl.on_auto_fit()
        assert controller.auto_ctrl._worker is None
        assert "start" in window.header._status_label.text().lower()


# -- manual fitting --------------------------------------------------------------

def _click_circle(controller, center=(125.0, 200.0), r=30.0, n=6):
    for a in np.linspace(0, 2 * np.pi, n, endpoint=False):
        controller.manual_ctrl.on_point_clicked(center[1] + r * np.cos(a),
                                                center[0] + r * np.sin(a))


class TestManual:
    def test_clear_marks_frame_unprocessed(self, gui, make_sequence):
        _, controller = gui
        _load(controller, make_sequence("A", 5))
        state = controller._get_state()
        state.radius[0] = 40.0
        state.circle_fit_par[0] = [100.0, 100.0]
        controller.manual_ctrl.on_manual_clear()
        state = controller._get_state()
        assert state.radius[0] == -1.0
        assert np.isnan(state.circle_fit_par[0]).all()

    def test_done_without_folder_does_not_crash(self, gui):
        window, controller = gui
        controller.manual_ctrl.on_manual_select()
        _click_circle(controller)
        controller.manual_ctrl.on_manual_done()  # must not raise

    def test_points_do_not_leak_to_another_frame(self, gui, make_sequence):
        _, controller = gui
        _load(controller, make_sequence("A", 5))
        controller.manual_ctrl.on_manual_select()
        _click_circle(controller)
        controller.file_ctrl.on_frame_changed(1)
        controller.manual_ctrl.on_manual_done()
        state = controller._get_state()
        assert state.radius[1] == -1.0  # nothing written to the frame the user moved to
        assert state.radius[0] == -1.0

    def test_undo_after_done_does_not_resurrect_points(self, gui, make_sequence):
        window, controller = gui
        _load(controller, make_sequence("A", 5))
        controller.manual_ctrl.on_manual_select()
        _click_circle(controller)
        controller.manual_ctrl.on_manual_done()
        fitted = controller._get_state().radius[0]
        assert fitted > 0
        controller.manual_ctrl.undo_last_point()
        assert controller.manual_ctrl._manual_points == []
        assert controller._get_state().radius[0] == fitted


# -- single-frame refit ----------------------------------------------------------

class TestRefit:
    def test_failed_refit_keeps_previous_result(self, gui, make_sequence, monkeypatch):
        window, controller = gui
        _load(controller, make_sequence("A", 5))
        state = controller._get_state()
        state.radius[2] = 55.0
        state.circle_fit_par[2] = [120.0, 190.0]
        state.circle_xy[2] = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        from bubbletrack.controller import controller as controller_module

        monkeypatch.setattr(
            controller_module, "detect_bubble_with_mode",
            lambda *a, **k: (np.zeros((10, 10), bool), np.empty((0, 2))),
        )
        controller._on_refit_point(2)
        state = controller._get_state()
        assert state.radius[2] == 55.0
        np.testing.assert_array_equal(state.circle_fit_par[2], [120.0, 190.0])
        assert "kept" in window.header._status_label.text().lower()
