"""File > Settings: saves the switches and applies them to the window at once.

``show_classic_methods``: the Method selector and the classic methods'
controls.  They are also shown whenever the current parameters use a classic
method, so a session or config from before stays editable.
``show_all_view_modes``: the image view toolbar (Image / Overlay / Wipe);
without it the view is Image + Mask.
"""

from __future__ import annotations

import logging

from bubbletrack.event_bus import UI_SETTING_CHANGED
from bubbletrack.model import config
from bubbletrack.ui.measure_panel import is_classic
from bubbletrack.ui.settings_dialog import SettingsDialog

logger = logging.getLogger(__name__)

CLASSIC_METHODS = "show_classic_methods"
ALL_VIEW_MODES = "show_all_view_modes"
_WARN = "#f59e0b"


class SettingsController:
    """Holds the current values (also when the config cannot be written)."""

    def __init__(self, window, app) -> None:
        self.w = window
        self._app = app
        self._values = {name: config.ui_setting(name) for name in config.UI_SETTINGS}
        window.settings_requested.connect(self.open_dialog)
        app.bus.subscribe(UI_SETTING_CHANGED, self._on_changed)
        self.apply_views()

    def value(self, name: str) -> bool:
        return self._values[name]

    def set(self, name: str, value: bool) -> None:
        """Save a switch, apply it, and tell the rest of the app."""
        if name not in self._values:
            raise ValueError(f"unknown UI setting {name!r}")
        self._values[name] = bool(value)
        if not config.set_ui_setting(name, bool(value)):
            self.w.header.set_status("Setting applied but not saved (config not writable)", _WARN)
        logger.info("Setting %s = %s", name, bool(value))
        self._app.bus.emit(UI_SETTING_CHANGED, name, bool(value))

    # -- applying ----------------------------------------------------------- #

    def apply_classic(self) -> None:
        """Method selector on when asked for, or when a classic method is in use."""
        mode = self._app._get_state().detection_mode
        visible = self.value(CLASSIC_METHODS) or is_classic(mode)
        self.w.left_panel.measure_panel.set_classic_methods_visible(visible)

    def apply_views(self) -> None:
        self.w.set_all_view_modes(self.value(ALL_VIEW_MODES))

    def _on_changed(self, name: str, _value: bool) -> None:
        if name == CLASSIC_METHODS:
            self.apply_classic()
        elif name == ALL_VIEW_MODES:
            self.apply_views()

    # -- the dialog ----------------------------------------------------------- #

    def make_dialog(self) -> SettingsDialog:
        dialog = SettingsDialog(self._values, parent=self.w)
        dialog.setting_changed.connect(self.set)
        return dialog

    def open_dialog(self) -> None:
        self.make_dialog().exec()
