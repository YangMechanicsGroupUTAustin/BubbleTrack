"""File > Settings: saves the switches and applies them to the window at once.

``show_classic_methods``: the Method selector and the classic methods'
controls (the side panel's ``method_row_shown`` also shows the selector while
a classic method is in use, so a session from before stays editable).
Turning it off while a classic method is selected switches to Auto; results
already measured are kept.
``show_all_view_modes``: the image view toolbar (Image / Overlay / Wipe);
without it the view is Image + Mask.
"""

from __future__ import annotations

import logging

from bubbletrack.controller.session_controller import has_results
from bubbletrack.event_bus import UI_SETTING_CHANGED
from bubbletrack.model import config
from bubbletrack.model.constants import AUTO_METHOD
from bubbletrack.ui.measure_panel import is_classic
from bubbletrack.ui.settings_dialog import SettingsDialog

logger = logging.getLogger(__name__)

CLASSIC_METHODS = "show_classic_methods"
ALL_VIEW_MODES = "show_all_view_modes"
SWITCHED_TO_AUTO = "Classic methods off: the method is now Auto"
RESULTS_KEPT = " (results so far are kept)"
_OK, _WARN = "#10b981", "#f59e0b"


class SettingsController:
    """Holds the current values (also when the config cannot be written)."""

    def __init__(self, window, app) -> None:
        self.w = window
        self._app = app
        self._values = {name: config.ui_setting(name) for name in config.UI_SETTINGS}
        window.settings_requested.connect(self.open_dialog)
        app.bus.subscribe(UI_SETTING_CHANGED, self._on_changed)
        self.apply_classic()
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
        """Pass the switch to the side panel (its ``method_row_shown`` decides)."""
        self.w.left_panel.measure_panel.set_classic_methods_visible(self.value(CLASSIC_METHODS))

    def leave_classic(self) -> None:
        """Classic methods were turned off: a classic method in use becomes Auto,
        as if chosen in the selector (state, view and tools follow; results stay)."""
        panel = self.w.left_panel.measure_panel
        previous = panel.method()
        if not is_classic(previous):
            return
        panel.set_method(AUTO_METHOD)
        logger.info("Classic methods off: method %s replaced by Auto", previous)
        kept = RESULTS_KEPT if has_results(self._app._get_state()) else ""
        self.w.header.set_status(SWITCHED_TO_AUTO + kept, _OK)

    def apply_views(self) -> None:
        self.w.set_all_view_modes(self.value(ALL_VIEW_MODES))

    def _on_changed(self, name: str, value: bool) -> None:
        if name == CLASSIC_METHODS:
            self.apply_classic()
            if not value:
                self.leave_classic()
        elif name == ALL_VIEW_MODES:
            self.apply_views()

    # -- the dialog ----------------------------------------------------------- #

    def make_dialog(self) -> SettingsDialog:
        dialog = SettingsDialog(self._values, parent=self.w)
        dialog.setting_changed.connect(self.set)
        return dialog

    def open_dialog(self) -> None:
        self.make_dialog().exec()
