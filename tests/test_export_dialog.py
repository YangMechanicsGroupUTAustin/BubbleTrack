"""The Export dialog: one Export... button replaces the Post Processing panel."""

from __future__ import annotations

import os

import numpy as np
import pytest
from PyQt6.QtWidgets import QMessageBox
from scipy.io import loadmat

from bubbletrack.controller import export_controller as ec
from bubbletrack.model import config
from bubbletrack.ui import export_dialog as ed
from bubbletrack.ui.export_dialog import ExportDialog, report_path_for, with_format_extension


def _names(units: str, fmt: str) -> str:
    return os.path.join("/out", f"shot_{units}{ed.FORMAT_EXT[fmt]}")


def _dialog(**kw) -> ExportDialog:
    values = dict(fps=5e6, um2px=2.5, fit_length=9, default_path=_names)
    values.update(kw)
    return ExportDialog(**values)


class TestPaths:
    @pytest.mark.parametrize("name, fmt, expected", [
        ("r.mat", "csv", "r.csv"),
        ("r.csv", "csv", "r.csv"),
        ("r", "xlsx", "r.xlsx"),
        ("r.v2", "mat", "r.v2.mat"),
    ])
    def test_extension_follows_the_format(self, name, fmt, expected):
        assert with_format_extension(os.path.join("a", name), fmt) == os.path.join("a", expected)

    def test_no_path_stays_empty(self):
        assert with_format_extension("", "mat") == ""

    def test_report_sits_next_to_the_export(self):
        assert report_path_for(os.path.join("d", "shot_R.mat")) == os.path.join("d", "shot_R_report.pdf")


class TestDialog:
    def test_shows_the_given_values(self, qapp):
        dialog = _dialog(fmt="csv", units="pixel")
        assert dialog.get_fps() == pytest.approx(5e6)
        assert dialog._fps_unit.currentText() == "M"
        assert dialog.get_scale() == pytest.approx(2.5)
        assert dialog.get_fit_length() == 9
        assert (dialog.get_units(), dialog.get_format()) == ("pixel", "csv")
        assert dialog.get_path() == _names("pixel", "csv")
        assert not dialog.report_enabled()

    def test_default_name_follows_units_and_format(self, qapp):
        dialog = _dialog()
        dialog.set_units("pixel")
        dialog.set_format("xlsx")
        assert dialog.get_path() == _names("pixel", "xlsx")

    def test_a_chosen_name_is_kept_and_only_its_extension_changes(self, qapp, tmp_path):
        dialog = _dialog()
        chosen = str(tmp_path / "mine.mat")
        dialog._path.setText(chosen)
        dialog._path.textEdited.emit(chosen)
        dialog.set_units("pixel")
        dialog.set_format("csv")
        assert dialog.get_path() == str(tmp_path / "mine.csv")

    def test_physical_edits_are_announced(self, qapp):
        dialog = _dialog()
        seen = []
        dialog.physical_params_changed.connect(lambda: seen.append(dialog.get_fps()))
        dialog._fps_unit.setCurrentText("k")
        dialog._scale_spin.setValue(1.5)
        assert seen[0] == pytest.approx(5e3) and len(seen) == 2
        dialog.set_physical_params(30.0, 1.0, 11)  # programmatic: silent
        assert len(seen) == 2 and dialog._fps_unit.currentText() == "Hz"


# -- the controller ------------------------------------------------------------ #

@pytest.fixture
def fitted(gui, make_sequence):
    window, controller = gui
    controller.file_ctrl.on_folder_selected(str(make_sequence("shot_7", 30)))
    state = controller._get_state()
    t = np.arange(1, 31, dtype=float)
    state.radius[:] = 100.0 - 0.05 * (t - 15.3) ** 2
    state.circle_fit_par[:] = 50.0
    return window, controller


def _export(controller, dialog) -> bool:
    return controller.export_ctrl.export_from(dialog)


class TestExportFromTheDialog:
    def test_button_and_menu_open_the_dialog(self, fitted, monkeypatch):
        window, controller = fitted
        opened = []
        monkeypatch.setattr(ExportDialog, "exec", lambda self: opened.append(self) or 0)
        window.left_panel.export_button.click()
        window.export_action.trigger()
        assert len(opened) == 2

    def test_dialog_starts_from_the_state_and_remembered_format(self, fitted, tmp_path):
        window, controller = fitted
        config.set_meta(last_export_format="xlsx", last_export_dir=str(tmp_path))
        dialog = controller.export_ctrl.make_dialog()
        state = controller._get_state()
        assert dialog.get_fps() == pytest.approx(state.fps)
        assert dialog.get_scale() == pytest.approx(state.um2px)
        assert dialog.get_format() == "xlsx"
        assert os.path.dirname(dialog.get_path()) == str(tmp_path)
        assert os.path.basename(dialog.get_path()).startswith("shot_7_RofT_data_")

    def test_physical_values_reach_the_state_even_without_exporting(self, fitted):
        window, controller = fitted
        dialog = controller.export_ctrl.make_dialog()
        dialog._fps_unit.setCurrentText("M")
        dialog._fps_spin.setValue(2.0)
        dialog._scale_spin.setValue(1.25)
        dialog._fit_len_spin.setValue(15)
        dialog.reject()
        state = controller._get_state()
        assert state.fps == pytest.approx(2e6)
        assert state.um2px == pytest.approx(1.25)
        assert state.rmax_fit_length == 15

    def test_physical_mat_with_report(self, fitted, tmp_path, monkeypatch):
        window, controller = fitted
        reports = []
        monkeypatch.setattr(ec, "generate_report", lambda path, *a, **k: reports.append(path))
        dialog = controller.export_ctrl.make_dialog()
        dialog.confirm_values()  # "These are right"
        out = tmp_path / "run.mat"
        dialog.set_path(str(out))
        dialog.set_report_enabled(True)
        assert _export(controller, dialog)
        assert "R" in loadmat(str(out))
        assert reports == [str(tmp_path / "run_report.pdf")]
        assert dialog.result() == ExportDialog.DialogCode.Accepted
        assert "run.mat" in window.header.status_text()
        assert config.get_meta("last_export_dir") == str(tmp_path)
        assert config.get_meta("last_export_format") == "mat"

    def test_pixel_csv(self, fitted, tmp_path):
        window, controller = fitted
        dialog = controller.export_ctrl.make_dialog()
        dialog.set_units("pixel")
        dialog.set_format("csv")
        dialog.set_path(str(tmp_path / "px"))
        assert _export(controller, dialog)
        assert (tmp_path / "px.csv").exists()

    def test_failure_keeps_the_dialog_open_with_the_reason(self, fitted, tmp_path):
        window, controller = fitted
        dialog = controller.export_ctrl.make_dialog()
        dialog.set_path(str(tmp_path / "missing_dir" / "r.mat"))
        dialog.set_units("pixel")
        assert not _export(controller, dialog)
        assert dialog.status_text().startswith("Error")
        assert dialog.result() != ExportDialog.DialogCode.Accepted

    def test_empty_path_is_refused(self, fitted):
        _, controller = fitted
        dialog = controller.export_ctrl.make_dialog()
        dialog.set_path("")
        assert not _export(controller, dialog)
        assert "choose" in dialog.status_text().lower()

    def test_existing_file_is_replaced_only_when_confirmed(self, fitted, tmp_path, monkeypatch):
        _, controller = fitted
        out = tmp_path / "r.mat"
        out.write_bytes(b"keep me")
        dialog = controller.export_ctrl.make_dialog()
        dialog.set_units("pixel")
        dialog.set_path(str(out))
        monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.No)
        assert not _export(controller, dialog)
        assert out.read_bytes() == b"keep me"
        monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
        assert _export(controller, dialog)
        assert out.read_bytes() != b"keep me"

    def test_no_results_says_so(self, gui, tmp_path):
        window, controller = gui
        dialog = controller.export_ctrl.make_dialog()
        dialog.set_path(str(tmp_path / "r.mat"))
        assert not _export(controller, dialog)
        assert dialog.status_text() == "No data to export"

    def test_the_units_choice_is_remembered_for_the_next_dialog(self, fitted, tmp_path):
        _, controller = fitted
        dialog = controller.export_ctrl.make_dialog()
        dialog.set_units("pixel")
        dialog.set_path(str(tmp_path / "r.mat"))
        assert _export(controller, dialog)
        assert controller.export_ctrl.make_dialog().get_units() == "pixel"
