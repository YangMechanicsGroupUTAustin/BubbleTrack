"""File > Settings: two switches, saved in the config, applied at once."""

from __future__ import annotations

import pytest
from PyQt6.QtGui import QKeySequence

from bubbletrack.event_bus import UI_SETTING_CHANGED
from bubbletrack.model import config
from bubbletrack.model.state import update_state
from bubbletrack.ui.image_compare import CompareMode
from bubbletrack.ui.settings_dialog import SETTINGS, SettingsDialog

CLASSIC, VIEWS = "show_classic_methods", "show_all_view_modes"


class TestDialog:
    def test_one_switch_per_setting_showing_the_values(self, qapp):
        dialog = SettingsDialog({CLASSIC: True, VIEWS: False})
        assert [name for name, _t, _x in SETTINGS] == [CLASSIC, VIEWS]
        assert dialog.value(CLASSIC) and not dialog.value(VIEWS)

    def test_toggling_emits_name_and_value(self, qapp):
        dialog = SettingsDialog({CLASSIC: False, VIEWS: False})
        seen = []
        dialog.setting_changed.connect(lambda name, on: seen.append((name, on)))
        dialog.switch(VIEWS).click()
        dialog.switch(VIEWS).click()
        assert seen == [(VIEWS, True), (VIEWS, False)]

    def test_set_value_is_silent(self, qapp):
        dialog = SettingsDialog({})
        seen = []
        dialog.setting_changed.connect(lambda *a: seen.append(a))
        dialog.set_value(CLASSIC, True)
        assert dialog.value(CLASSIC) and not seen


class TestMenu:
    def test_file_settings_with_ctrl_comma(self, gui):
        window, _ = gui
        action = window.settings_action
        assert action.text().replace("&", "").startswith("Settings")
        assert action.shortcut() == QKeySequence("Ctrl+,")
        file_menu = window.menuBar().actions()[0].menu()
        assert action in file_menu.actions()

    def test_action_opens_the_dialog(self, gui, monkeypatch):
        window, controller = gui
        opened = []
        monkeypatch.setattr(SettingsDialog, "exec", lambda self: opened.append(self) or 0)
        window.settings_action.trigger()
        assert len(opened) == 1 and opened[0].value(CLASSIC) is False


class TestSavedAndApplied:
    def test_change_is_saved_and_announced(self, gui):
        window, controller = gui
        events = []
        controller.bus.subscribe(UI_SETTING_CHANGED, lambda name, on: events.append((name, on)))
        dialog = controller.settings.make_dialog()
        dialog.switch(CLASSIC).click()
        assert config.ui_setting(CLASSIC) is True
        assert events == [(CLASSIC, True)]

    def test_classic_methods_appear_and_disappear_live(self, gui):
        window, controller = gui
        panel = window.left_panel.measure_panel
        assert panel.method_row.isHidden()
        controller.settings.set(CLASSIC, True)
        assert not panel.method_row.isHidden()
        assert panel.method() == "edge" and panel.classic_section.isHidden()
        controller.settings.set(CLASSIC, False)
        assert panel.method_row.isHidden()

    def test_switching_off_keeps_the_selector_while_a_classic_method_is_used(self, gui):
        window, controller = gui
        controller.settings.set(CLASSIC, True)
        controller._state = update_state(controller._get_state(), detection_mode="area")
        controller.settings.set(CLASSIC, False)
        assert not window.left_panel.measure_panel.method_row.isHidden()

    def test_view_modes_toolbar_follows_the_switch(self, gui):
        window, controller = gui
        assert window.compare_toolbar.isHidden()
        assert window.compare_mode is CompareMode.SIDE_BY_SIDE
        controller.settings.set(VIEWS, True)
        assert not window.compare_toolbar.isHidden()
        window.compare_toolbar._buttons[CompareMode.OVERLAY].click()
        assert window.compare_mode is CompareMode.OVERLAY
        controller.settings.set(VIEWS, False)
        assert window.compare_toolbar.isHidden()
        assert window.compare_mode is CompareMode.SIDE_BY_SIDE  # back to Image + Mask
        assert not window.binary_panel.isHidden()

    def test_saved_settings_apply_at_start(self, qapp, isolated_config):
        from bubbletrack.controller.controller import AppController
        from bubbletrack.ui.main_window import MainWindow

        config.set_ui_setting(VIEWS, True)
        config.set_ui_setting(CLASSIC, True)
        window = MainWindow()
        AppController(window)
        try:
            assert not window.compare_toolbar.isHidden()
            assert not window.left_panel.measure_panel.method_row.isHidden()
        finally:
            window.close()

    def test_unwritable_config_still_applies_and_warns(self, gui, monkeypatch):
        window, controller = gui
        monkeypatch.setattr(config, "set_ui_setting", lambda name, value: False)
        controller.settings.set(VIEWS, True)
        assert not window.compare_toolbar.isHidden()
        assert "not saved" in window.header.status_text()

    def test_unknown_setting_is_refused(self, gui):
        _, controller = gui
        with pytest.raises(ValueError):
            controller.settings.set("nope", True)
