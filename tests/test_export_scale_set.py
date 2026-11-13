"""Frame rate and scale must be set for the recording before a physical export.

The config carries the previous recording's values (1 Mfps, 3.2 µm/px by
default): the Export dialog marks them "Not set for this recording — check"
and physical export waits until the user typed them or pressed "These are
right"; a session restores them as set.  The status bar's scale matches.
"""

from __future__ import annotations

import numpy as np
import pytest

from bubbletrack.ui.export_dialog import BLOCKED_TEXT, NOT_SET_TEXT, ExportDialog


def _dialog(**kw) -> ExportDialog:
    values = dict(fps=1e6, um2px=3.2, fit_length=11, default_path=lambda u, f: f"/out/r.{f}")
    return ExportDialog(**{**values, **kw})


@pytest.fixture
def fitted(gui, make_sequence):
    window, controller = gui
    controller.file_ctrl.on_folder_selected(str(make_sequence("shot", 30)))
    t = np.arange(1, 31, dtype=float)
    state = controller._get_state()
    state.radius[:] = 100.0 - 0.05 * (t - 15.3) ** 2
    state.circle_fit_par[:] = 50.0
    return window, controller


class TestTheDialog:
    def test_unset_fields_are_marked_and_block_physical_export(self, qapp):
        dialog = _dialog(unset=frozenset({"fps", "scale"}), units="physical")
        assert not dialog.physical_ready()
        assert dialog._unset_label.text() == NOT_SET_TEXT and not dialog._unset_line.isHidden()
        assert "f59e0b" in dialog._fps_spin.styleSheet()
        assert not dialog.export_button.isEnabled() and dialog.status_text() == BLOCKED_TEXT

    def test_pixels_need_no_scale(self, qapp):
        dialog = _dialog(unset=frozenset({"fps", "scale"}), units="physical")
        dialog.set_units("pixel")
        assert dialog.export_button.isEnabled() and dialog.status_text() == ""

    def test_typing_a_value_sets_that_field_only(self, qapp):
        dialog = _dialog(unset=frozenset({"fps", "scale"}))
        dialog._scale_spin.setValue(1.6)
        assert dialog.unset_fields() == {"fps"} and dialog._scale_spin.styleSheet() == ""
        dialog._fps_spin.setValue(2.0)
        assert dialog.physical_ready() and dialog.export_button.isEnabled()
        assert dialog._unset_line.isHidden()

    def test_these_are_right_keeps_the_values(self, qapp):
        dialog = _dialog(unset=frozenset({"fps", "scale"}))
        seen = []
        dialog.physical_params_changed.connect(lambda: seen.append(dialog.unset_fields()))
        dialog._confirm.click()
        assert seen == [frozenset()] and dialog.get_scale() == pytest.approx(3.2)

    def test_the_report_says_not_set(self):
        from bubbletrack.model.report import build_report_parameters
        from bubbletrack.model.state import AppState

        params = build_report_parameters(AppState(), physical_set=False)
        assert params["FPS"] == params["Scale"] == "not set"
        assert build_report_parameters(AppState(), um2px=1.5)["Scale"] == "1.50 um/px"

    def test_rmax_fit_is_explained_on_screen(self, qapp):
        from PyQt6.QtWidgets import QLabel

        texts = [lbl.text() for lbl in _dialog().findChildren(QLabel)]
        assert any("largest radius" in t and "Rmax" in t for t in texts)


class TestTheRecording:
    def test_a_new_recording_starts_unset_and_says_so(self, fitted):
        window, controller = fitted
        assert not controller.export_ctrl.physical_set()
        assert window.status_bar.scale_text() == "Scale: not set"
        dialog = controller.export_ctrl.make_dialog()
        assert dialog.unset_fields() == {"fps", "scale"}

    def test_physical_export_waits_for_the_values(self, fitted, tmp_path):
        window, controller = fitted
        dialog = controller.export_ctrl.make_dialog()
        dialog.set_units("physical")
        dialog.set_path(str(tmp_path / "r.mat"))
        assert not controller.export_ctrl.export_from(dialog)
        assert not (tmp_path / "r.mat").exists()
        dialog._scale_spin.setValue(1.6)
        dialog._fps_spin.setValue(2.0)  # in M: 2 Mfps
        assert controller.export_ctrl.export_from(dialog)
        assert window.status_bar.scale_text() == "Scale: 1.600 µm/px"

    def test_set_values_are_kept_for_the_recording_and_reset_by_the_next(
            self, fitted, make_sequence):
        window, controller = fitted
        controller.export_ctrl.make_dialog().confirm_values()
        assert controller.export_ctrl.make_dialog().physical_ready()
        assert window.status_bar.scale_text() == "Scale: 3.200 µm/px"
        controller.file_ctrl.on_folder_selected(str(make_sequence("next", 5)))
        assert not controller.export_ctrl.make_dialog().physical_ready()
        assert window.status_bar.scale_text() == "Scale: not set"

    def test_a_session_restores_them_as_set(self, fitted, tmp_path):
        window, controller = fitted
        dialog = controller.export_ctrl.make_dialog()
        dialog._scale_spin.setValue(2.5)
        dialog._fps_spin.setValue(5.0)
        path = tmp_path / "s.brt"
        assert controller.save_session_file(str(path))
        assert controller.load_session_file(str(path))
        assert controller.export_ctrl.physical_set()
        assert window.status_bar.scale_text() == "Scale: 2.500 µm/px"
