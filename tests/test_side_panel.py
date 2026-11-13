"""The Auto-first side panel: no tabs, one Fit, one Export; classic methods on request."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from PyQt6.QtWidgets import QLabel, QMessageBox, QPushButton

from bubbletrack.model import config
from bubbletrack.model.state import AppState, update_state
from bubbletrack.ui.measure_panel import MeasurePanel


def _texts(widget, kind) -> list[str]:
    return [w.text() for w in widget.findChildren(kind) if not w.isHidden()]


def _visible_to(panel, widget) -> bool:
    return widget.isVisibleTo(panel)


# -- the panel on its own ------------------------------------------------------- #

class TestMeasurePanel:
    def test_auto_only_by_default(self, qapp):
        panel = MeasurePanel()
        assert panel.method() == "edge"
        assert not _visible_to(panel, panel.method_row)
        assert _visible_to(panel, panel.edge_box)
        assert not _visible_to(panel, panel.classic_section)

    def test_no_wall_at_selector_and_bubble_colour_in_advanced(self, qapp):
        panel = MeasurePanel()
        box = panel.edge_box
        labels = [label.text() for label in box.findChildren(QLabel)]
        assert "Wall at:" not in labels
        assert box.advanced.isAncestorOf(box._polarity)
        assert box.polarity() == "dark"
        panel.set_filter_params({"detection_mode": "edge", "edge_definition": "outer",
                                 "edge_polarity": "bright"})
        params = panel.get_filter_params()
        assert params["edge_definition"] == "steep"  # Auto always uses the 50% edge
        assert params["edge_polarity"] == "bright"

    def test_classic_method_shows_its_controls(self, qapp):
        panel = MeasurePanel()
        panel.set_classic_methods_visible(True)
        panel._method_combo.setCurrentIndex(panel._method_combo.findData("gradient"))
        assert _visible_to(panel, panel.classic_section)
        assert not _visible_to(panel, panel.edge_box)
        assert not panel.classic._gradient_boundary.isHidden()  # in Advanced Filters
        assert panel.classic._area_widget.isHidden()
        assert not panel.classic._autotune_btn.isEnabled()  # Auto Tune: intensity only

    def test_method_change_is_announced(self, qapp):
        panel = MeasurePanel()
        panel.set_classic_methods_visible(True)
        seen = []
        panel.filters_changed.connect(lambda: seen.append(panel.get_filter_params()["detection_mode"]))
        panel._method_combo.setCurrentIndex(panel._method_combo.findData("intensity"))
        assert seen == ["intensity"]

    def test_preset_with_a_method_applies(self, qapp):
        """Loading a preset that names a method used to crash (missing widgets)."""
        panel = MeasurePanel()
        seen = []
        panel.filters_changed.connect(lambda: seen.append("filters"))
        panel.threshold_changed.connect(lambda v: seen.append(("thr", round(v, 2))))
        panel.apply_preset({"threshold": 0.42, "detection_mode": "area", "area_max_regions": 3,
                            "edge_flags": [True, False, False, True], "gaussian_sigma": 4})
        params = panel.get_filter_params()
        assert params["detection_mode"] == "area" and params["area_max_regions"] == 3
        assert params["gaussian_sigma"] == 4
        assert panel.get_edge_flags() == [True, False, False, True]
        assert ("thr", 0.42) in seen and seen[-1] == "filters"

    def test_preset_round_trip(self, qapp):
        a, b = MeasurePanel(), MeasurePanel()
        a.apply_preset({"threshold": 0.3, "removing_factor": 70, "detection_mode": "intensity"})
        b.apply_preset(a.preset())
        assert b.preset() == a.preset()


# -- the side panel in the window ----------------------------------------------- #

class TestDefaultWindow:
    def test_fresh_start_measures_with_auto(self, gui):
        _, controller = gui
        assert controller._get_state().detection_mode == "edge"

    def test_only_source_auto_fit_and_export(self, gui):
        window, _ = gui
        lp = window.left_panel
        shown = [b.text() for b in lp.findChildren(QPushButton) if b.isVisibleTo(lp)]
        for gone in ("Fit Current Frame", "Auto Tune", "Fit All", "Invert Mask", "Stop",
                     "Export Pixel Data", "Export Physical Data", "Generate Report",
                     "Batch Folders...", "Pre-tune", "Automatic", "Manual"):
            assert gone not in shown, gone
        assert lp.fit_controls.fit_button.text() == "Fit"
        assert lp.export_button.text() == "Export…"
        assert lp.measure_panel.edge_box.title() == "Auto (reference frame)"
        assert not lp.manual_section.isVisibleTo(lp)
        assert not any("Pre-tune" in t for t in _texts(lp, QLabel))

    def test_status_bar_mode_names_no_removed_tab(self, gui):
        window, _ = gui
        assert window.status_bar._mode_label.text() == "Mode: Measure"

    def _start(self):
        from bubbletrack.controller.controller import AppController
        from bubbletrack.ui.main_window import MainWindow

        window = MainWindow()
        return window, AppController(window)

    def test_old_saved_classic_default_starts_in_auto(self, qapp, isolated_config):
        """Older versions saved "intensity" as every config's default method."""
        config.save_config(AppState(detection_mode="intensity"))
        window, controller = self._start()
        try:
            panel = window.left_panel.measure_panel
            assert controller._get_state().detection_mode == "edge"
            assert not panel.classic_section.isVisibleTo(window.left_panel)
        finally:
            window.close()

    def test_saved_classic_method_kept_when_classic_methods_are_on(self, qapp, isolated_config):
        config.save_config(AppState(detection_mode="intensity"))
        config.set_ui_setting("show_classic_methods", True)
        window, controller = self._start()
        try:
            panel = window.left_panel.measure_panel
            assert controller._get_state().detection_mode == "intensity"
            assert panel.method_row.isVisibleTo(window.left_panel)
            assert panel.classic_section.isVisibleTo(window.left_panel)
        finally:
            window.close()

    def test_changing_the_method_redraws_the_frame(self, gui, make_sequence):
        window, controller = gui
        controller.file_ctrl.on_folder_selected(str(make_sequence("A", 4)))
        controller.settings.set("show_classic_methods", True)
        panel = window.left_panel.measure_panel
        controller._display_timer.stop()
        panel._method_combo.setCurrentIndex(panel._method_combo.findData("intensity"))
        assert controller._get_state().detection_mode == "intensity"
        assert controller._display_timer.isActive()  # titles and overlays of the new method

    def test_old_outer_wall_setting_becomes_the_50_percent_edge(self, gui):
        window, controller = gui
        controller._state = update_state(controller._get_state(), edge_definition="outer")
        controller._after_session_restore()
        assert controller._get_state().edge_definition == "steep"


class TestFit:
    def test_fit_measures_every_frame(self, gui, make_sequence, monkeypatch):
        window, controller = gui
        controller.file_ctrl.on_folder_selected(str(make_sequence("A", 12)))
        calls = []
        monkeypatch.setattr(controller.auto_ctrl, "fit_range", lambda a, b: calls.append((a, b)))
        window.left_panel.fit_controls.fit_button.click()
        assert calls == [(0, 11)]

    def test_fit_without_images_says_what_to_do(self, gui):
        window, controller = gui
        window.left_panel.fit_controls.fit_button.click()
        assert not controller.auto_ctrl.is_busy()
        assert "Load images" in window.header.status_text()

    def test_progress_and_stop_only_while_running(self, gui):
        window, controller = gui
        fc = window.left_panel.fit_controls
        assert fc.progress_row.isHidden() and fc.fit_button.isEnabled()
        fc.set_running(True)
        fc.set_progress(3, 12, "1m 0s remaining")
        assert not fc.progress_row.isHidden() and fc.stop_button.isEnabled()
        assert not fc.fit_button.isEnabled()
        assert fc.progress_text().startswith("Fitting frame 3/12")
        fc.set_running(False)
        assert fc.progress_row.isHidden() and fc.fit_button.isEnabled()

    def test_stop_reaches_the_run(self, gui):
        window, controller = gui
        stopped = []
        controller.auto_ctrl._worker = SimpleNamespace(request_stop=lambda: stopped.append("fit"))
        try:
            fc = window.left_panel.fit_controls
            fc.set_running(True)
            fc.stop_button.click()
        finally:
            controller.auto_ctrl._worker = None
        assert stopped == ["fit"]

    def test_run_lock_disables_editing_but_not_stop(self, gui):
        window, controller = gui
        lp = window.left_panel
        controller._apply_run_lock(True)
        assert not lp.measure_panel.isEnabled() and not lp.image_source.isEnabled()
        assert not lp.export_button.isEnabled()
        assert not window.clear_results_action.isEnabled()
        controller._apply_run_lock(False)
        assert lp.measure_panel.isEnabled() and lp.export_button.isEnabled()
        assert window.clear_results_action.isEnabled()


class TestClearAllResults:
    def _fitted(self, controller, make_sequence):
        controller.file_ctrl.on_folder_selected(str(make_sequence("A", 5)))
        state = controller._get_state()
        state.radius[:] = 30.0
        return state

    def test_lives_in_the_edit_menu(self, gui):
        window, _ = gui
        edit = next(a.menu() for a in window.menuBar().actions() if a.text() == "&Edit")
        assert window.clear_results_action in edit.actions()

    @pytest.mark.parametrize("answer, cleared", [
        (QMessageBox.StandardButton.No, False), (QMessageBox.StandardButton.Yes, True)])
    def test_asks_first(self, gui, make_sequence, monkeypatch, answer, cleared):
        window, controller = gui
        self._fitted(controller, make_sequence)
        asked = []
        monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: asked.append(a) or answer)
        window.clear_results_action.trigger()
        assert len(asked) == 1
        assert bool(np.all(controller._get_state().radius < 0)) is cleared

    def test_nothing_to_clear_asks_nothing(self, gui, monkeypatch):
        window, _ = gui
        monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: pytest.fail("asked"))
        window.clear_results_action.trigger()


class TestManualFixShim:
    """The Review panel still opens manual fixing through the hidden mode switch."""

    def test_manual_mode_shows_the_fix_tools_and_back_returns(self, gui):
        window, _ = gui
        lp = window.left_panel
        lp.tab_bar.select("Manual")
        assert lp.manual_section.isVisibleTo(lp) and not lp.measure_panel.isVisibleTo(lp)
        assert window.status_bar._mode_label.text() == "Mode: Manual"
        lp.manual_back_button.click()
        assert lp.tab_bar.current_name() == "Pre-tune"
        assert not lp.manual_section.isVisibleTo(lp) and lp.measure_panel.isVisibleTo(lp)
        assert window.status_bar._mode_label.text() == "Mode: Measure"
