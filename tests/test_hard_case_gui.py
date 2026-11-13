"""Export Hard Case from the real window, offscreen: menus, labels, sources."""

from __future__ import annotations

import numpy as np
import pytest
from PyQt6.QtCore import QPoint
from PyQt6.QtWidgets import QDialog, QMessageBox

from bubbletrack.controller import hard_case_controller as hcc
from bubbletrack.event_bus import RESULTS_CHANGED
from bubbletrack.model import config
from bubbletrack.model.cases.bundle import open_bundle
from bubbletrack.model.cases.schema import (
    BY_ACCEPTED, BY_DELETED, BY_MANUAL, LABEL_NO_BUBBLE, LABEL_UNLABELLED, ROLE_REFERENCE,
)
from bubbletrack.model.image_io import imread_unicode
from bubbletrack.model.provenance import FitSource
from bubbletrack.model.state import update_state
from bubbletrack.ui.hard_case_dialog import HardCaseDialog, HardCaseInput

N = 12
ROI = ((31, 220), (61, 340))


@pytest.fixture
def app(gui, make_sequence):
    window, controller = gui
    controller.file_ctrl.on_folder_selected(str(make_sequence("shot 1", N)))
    controller._state = update_state(controller._get_state(), gridx=ROI[0], gridy=ROI[1])
    state = controller._get_state()
    state.radius[:] = np.linspace(30.0, 40.0, N)
    state.circle_fit_par[:] = [[125.0, 200.0]] * N
    state.fit_source[:] = FitSource.AUTO
    state.fit_confidence[:] = 0.9
    return window, controller


@pytest.fixture
def dialogs(monkeypatch, tmp_path):
    """Accept the export dialog (with a note) and save to tmp_path/<name>."""
    seen = {}

    def fake_exec(dialog):
        seen["range"] = dialog.values().start, dialog.values().end
        seen["summary"] = dialog._summary.text()
        dialog.set_note("radius jumps at the wall")
        return QDialog.DialogCode.Accepted

    def fake_save(parent, title, default, filters):
        seen["default"] = default
        return str(tmp_path / "out" / "case"), filters  # no extension: .zip is added

    (tmp_path / "out").mkdir()
    monkeypatch.setattr(HardCaseDialog, "exec", fake_exec)
    monkeypatch.setattr(hcc.QFileDialog, "getSaveFileName", fake_save)
    seen["path"] = tmp_path / "out" / "case.zip"
    return seen


def _export(controller, start, end, path, **kw):
    request = controller.hard_cases.make_request(HardCaseInput(start, end, kw.get("note", ""),
                                                               kw.get("reporter", "")))
    return controller.hard_cases.export(request, str(path))


def test_menu_entry(app):
    window, _ = app
    file_menu = window.menuBar().actions()[0].menu()
    assert window.export_hard_case_action in file_menu.actions()
    assert window.export_hard_case_action.text() == "Export &Hard Case..."


def test_export_records_the_user_fixes_as_labels(app, tmp_path):
    window, controller = app
    state = controller._get_state()
    state.radius[4], state.fit_source[4] = 33.0, FitSource.MANUAL
    controller._on_lock_range(5, 5, locked=True)
    controller._on_delete_point(6)
    case, size = _export(controller, 4, 6, tmp_path / "c.zip", note=" wall ", reporter="Ann")
    assert size > 0 and case.selection == (4, 6)
    assert [f.index for f in case.frames] == list(range(1, 10))
    assert case.frame(4).label.by == BY_MANUAL and case.frame(4).label.radius == 33.0
    assert case.frame(5).label.by == BY_ACCEPTED
    assert (case.frame(6).label.status, case.frame(6).label.by) == (LABEL_NO_BUBBLE, BY_DELETED)
    assert case.frame(7).label.status == LABEL_UNLABELLED
    assert (case.gridx, case.gridy) == ROI and case.note == "wall"
    assert case.detection.max_radius == 400.0
    with open_bundle(tmp_path / "c.zip") as bundle:
        crop = bundle.read_frame(5)
    full = imread_unicode(state.images[5])
    assert crop.dtype == np.uint16 and np.array_equal(crop, full[30:220, 60:340])
    assert config.get_meta("hard_case_reporter") == "Ann"
    assert "Hard case saved: c.zip" in window.header._status_label.text()


def test_deleted_frames_are_forgotten_once_refitted_or_cleared(app):
    _, controller = app
    hc = controller.hard_cases
    controller._on_lock_range(3, 3, locked=True)
    controller._on_delete_range(2, 4)  # the locked frame 3 is kept
    assert hc.deleted_frames() == {2, 4}
    state = controller._get_state()
    state.radius[2] = 31.0
    controller.bus.emit(RESULTS_CHANGED)
    assert hc.deleted_frames() == {4}
    controller.auto_ctrl.on_auto_clear()
    assert hc.deleted_frames() == frozenset()


def test_file_menu_exports_the_chart_selection(app, dialogs):
    window, controller = app
    window.radius_chart.set_selection(2, 5)
    window.export_hard_case_action.trigger()
    assert dialogs["range"] == (2, 5)
    assert "4 frame(s) + 5 neighbours" in dialogs["summary"]  # 2 before (clipped), 3 after
    assert "No frame here is fixed yet" in dialogs["summary"]
    assert dialogs["default"].endswith(".zip") and "shot_1_hardcase_f3-6_" in dialogs["default"]
    with open_bundle(dialogs["path"]) as bundle:
        assert bundle.case.selection == (2, 5)
        assert bundle.case.note == "radius jumps at the wall"


def test_default_range_falls_back_to_review_row_then_shown_frame(app):
    window, controller = app
    hc = controller.hard_cases
    controller.file_ctrl.on_frame_changed(7)
    assert hc.default_range() == (7, 7)
    state = controller._get_state()
    state.fit_confidence[[3, 4]] = 0.1
    controller.review.refresh()
    controller.review.panel.step(1)
    assert hc.default_range() == (3, 4)


def test_review_row_right_click_exports_the_row(qapp, app, dialogs):
    window, controller = app
    state = controller._get_state()
    state.fit_confidence[[8, 9]] = 0.1
    controller.review.refresh()
    window.show()
    qapp.processEvents()
    panel = controller.review.panel
    rect = panel._list.visualItemRect(panel._list.item(0))
    menu = panel.context_menu_at(rect.center())
    assert [a.text() for a in menu.actions()] == ["Export as hard case..."]
    below = QPoint(rect.center().x(), rect.bottom() + 3 * rect.height())  # no row there
    assert panel.context_menu_at(below) is None
    menu.actions()[0].trigger()
    assert dialogs["range"] == (8, 9)
    with open_bundle(dialogs["path"]) as bundle:
        assert bundle.case.selection == (8, 9)


def test_large_case_asks_before_saving(app, dialogs, monkeypatch):
    window, _ = app
    monkeypatch.setattr(hcc, "SIZE_WARN_BYTES", 10)
    answers = []
    monkeypatch.setattr(hcc.QMessageBox, "question",
                        lambda *a, **k: answers.append(a[2]) or QMessageBox.StandardButton.No)
    window.export_hard_case_action.trigger()
    assert answers and "Large files are slow to send" in answers[0]
    assert not dialogs["path"].exists() and "default" not in dialogs
    monkeypatch.setattr(hcc.QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    window.export_hard_case_action.trigger()
    assert dialogs["path"].exists()


def test_refused_while_a_fit_runs(app, dialogs, monkeypatch):
    window, controller = app
    with monkeypatch.context() as m:  # undone before the window closes (its guard asks)
        m.setattr(controller.auto_ctrl, "is_busy", lambda: True)
        assert controller.hard_cases.ask_and_export(1, 2) is None
    assert "range" not in dialogs
    assert "Stop the running fit" in window.header._status_label.text()
    controller.bus.emit("run_state_changed", True)
    assert not window.export_hard_case_action.isEnabled()
    controller.bus.emit("run_state_changed", False)
    assert window.export_hard_case_action.isEnabled()


def test_auto_mode_stores_the_reference_frame_and_tracking(app, tmp_path):
    _, controller = app
    controller._state = update_state(controller._get_state(), detection_mode="edge",
                                     reference_frame=11)
    case, _ = _export(controller, 2, 3, tmp_path / "e.zip")
    assert case.detection.reference_index == 11 and case.detection.reference_explicit
    assert case.detection.tracking  # "Track across frames" is on by default
    assert case.frame(11).role == ROLE_REFERENCE


def test_video_frames_are_decoded_and_stored(gui, tmp_path):
    from test_video_fitting import write_video

    window, controller = gui
    path = write_video(tmp_path / "shot.avi", np.linspace(25, 70, 8))
    controller.file_ctrl.on_video_selected(str(path))
    case, _ = _export(controller, 3, 3, tmp_path / "v.zip")
    assert case.source["kind"] == "video" and case.source["name"] == "shot.avi"
    assert case.frame(3).source_file == ""
    reader = controller._get_state().video_reader
    with open_bundle(tmp_path / "v.zip") as bundle:
        stored = bundle.read_frame(3)
    assert stored.dtype == np.uint8 and np.array_equal(stored, reader.read_frame(3))


def test_dat_recordings_are_stored_at_16_bit(gui, tmp_path):
    from test_dat_integration import _bubble_frames
    from test_hpvx2_dat import build_dat

    window, controller = gui
    frames = _bubble_frames(np.linspace(25, 60, 6))
    path = tmp_path / "Jin_12_00_00.dat"
    path.write_bytes(build_dat(frames))
    controller.file_ctrl.on_video_selected(str(path))
    case, _ = _export(controller, 1, 2, tmp_path / "d.zip")
    assert case.source["kind"] == "dat"
    with open_bundle(tmp_path / "d.zip") as bundle:
        stored = bundle.read_frame(2)
    assert stored.dtype == np.uint16
    assert np.array_equal(stored, controller._get_state().video_reader.read_frame(2))


def test_reporter_name_is_validated(isolated_config):
    assert config.set_meta(hard_case_reporter="  Ann \n Lee ")
    assert config.get_meta("hard_case_reporter") == "Ann Lee"
    with pytest.raises(ValueError):
        config.set_meta(hard_case_reporter="x" * 81)
    with pytest.raises(ValueError):
        config.set_meta(hard_case_reporter=42)


def test_dialog_values(qapp):
    dialog = HardCaseDialog(None, n_frames=100, start=9, end=4, reporter="  Ann  Lee ",
                            describe=lambda s, e: f"{s}-{e}")
    dialog.set_note("  wall  ")
    assert dialog.values() == HardCaseInput(4, 9, "wall", "Ann Lee")
    assert dialog._summary.text() == "4-9"
    dialog._end.setValue(50)
    assert dialog._summary.text() == "9-49"
