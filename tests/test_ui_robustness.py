"""Regression tests from the 2026-11-18 bug hunt of loading, display, export and settings."""

from __future__ import annotations

import json

import numpy as np

from bubbletrack.model.report import build_report_parameters
from bubbletrack.model.state import AppState, update_state


def test_an_undecodable_video_leaves_the_recording_alone(gui, make_sequence, tmp_path):
    window, controller = gui
    folder = make_sequence("a", 6)
    controller.file_ctrl.on_folder_selected(str(folder))
    controller._get_state().radius[:] = 25.0
    import bubbletrack.controller.file_controller as fc

    class Reader:  # opens and counts frames, but frame 0 does not decode
        total_frames, fps = 10, 25.0
        closed = False

        def read_frame(self, _i):
            raise ValueError("cannot decode frame 0")

        def close(self):
            Reader.closed = True

    original = fc.open_recording
    fc.open_recording = lambda _path: Reader()
    try:
        controller.file_ctrl.on_video_selected(str(tmp_path / "clip.avi"))
    finally:
        fc.open_recording = original
    state = controller._get_state()
    assert state.folder_path == str(folder) and not state.video_path
    assert np.all(state.radius == 25.0)
    assert Reader.closed and window.header.status_text().startswith("Cannot open video")


def test_an_unreadable_frame_is_not_shown_as_the_previous_one(gui, make_sequence):
    window, controller = gui
    folder = make_sequence("a", 6)
    controller.file_ctrl.on_folder_selected(str(folder))
    controller.file_ctrl.on_frame_changed(2)
    assert controller._get_state().cur_img is not None
    bad = controller._get_state().images[3]
    with open(bad, "wb") as fh:
        fh.write(b"not an image")
    controller.file_ctrl._cache.invalidate()
    controller.file_ctrl.on_frame_changed(3)
    state = controller._get_state()
    assert state.image_no == 3 and state.cur_img is None  # nothing of frame 3 is kept as frame 4
    assert window.status_bar.frame_text().startswith("Frame 4")
    assert window.header.status_text().startswith("Error")


def test_the_report_keeps_every_digit_it_needs():
    assert build_report_parameters(AppState(), um2px=1.5)["Scale"] == "1.50 um/px"
    assert build_report_parameters(AppState(), um2px=0.065)["Scale"] == "0.065 um/px"
    assert build_report_parameters(AppState(), fps=5e5)["FPS"] == "500000 Hz"
    assert build_report_parameters(AppState(), fps=29.97002997)["FPS"] == "29.97002997 Hz"


def test_a_config_saved_with_a_bom_still_loads(isolated_config):
    from bubbletrack.model import config

    config.CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    config.CONFIG_FILE.write_bytes(b"\xef\xbb\xbf" + json.dumps({"img_thr": 0.42}).encode("utf-8"))
    assert config.load_config().get("img_thr") == 0.42


def test_nothing_measured_exports_nothing(gui, make_sequence, tmp_path):
    _window, controller = gui
    controller.file_ctrl.on_folder_selected(str(make_sequence("a", 4)))
    dialog = controller.export_ctrl.make_dialog()
    dialog.set_units("pixel")
    dialog.set_path(str(tmp_path / "r.csv"))
    assert not controller.export_ctrl.export_from(dialog)
    assert not (tmp_path / "r.csv").exists()
    assert dialog.status_text() == "No data to export"


def test_a_new_rmax_fit_length_redraws_the_chart(gui, make_sequence, monkeypatch):
    import bubbletrack.controller.export_controller as ec

    _window, controller = gui
    controller.file_ctrl.on_folder_selected(str(make_sequence("a", 30)))
    calls = []
    monkeypatch.setattr(ec, "refresh_chart", lambda state, w: calls.append(state.rmax_fit_length))
    dialog = controller.export_ctrl.make_dialog()
    dialog._fit_len_spin.setValue(controller._get_state().rmax_fit_length + 10)
    assert calls and calls[-1] == controller._get_state().rmax_fit_length


def test_leaving_a_classic_method_hides_the_selector(gui, make_sequence):
    window, controller = gui
    controller.file_ctrl.on_folder_selected(str(make_sequence("a", 4)))
    panel = window.left_panel.measure_panel
    controller._set_state(update_state(controller._get_state(), detection_mode="gradient"))
    controller.settings.apply_classic()
    assert panel.classic_methods_visible()
    panel.set_filter_params({**panel.get_filter_params(), "detection_mode": "edge"})
    panel.filters_changed.emit()
    assert not panel.classic_methods_visible()


def test_a_bad_preset_changes_nothing(gui):
    import pytest

    window, controller = gui
    panel = window.left_panel.measure_panel
    before = (panel.get_threshold(), panel.get_removing_factor_slider(), panel.method())
    for bad in ({"threshold": 0.3, "removing_factor": "x"}, {"threshold": 0.3, "detection_mode": "EDGE"},
                {"threshold": 0.3, "edge_flags": [1, 0]}):
        with pytest.raises((ValueError, TypeError)):
            panel.apply_preset(bad)
        assert (panel.get_threshold(), panel.get_removing_factor_slider(), panel.method()) == before
    panel.apply_preset({"threshold": 0.3, "edge_flags": [1, 0, 0, 1]})
    assert panel.get_threshold() == pytest.approx(0.3) and panel.get_edge_flags() == [True, False, False, True]
