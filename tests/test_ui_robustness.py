"""Regression tests from the 2026-11-18 bug hunt of loading, display, export and settings."""

from __future__ import annotations

import json
import sys

import numpy as np
import pytest

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


# -- second self-review of the overnight commits ---------------------------- #

def _make_unreadable(controller, idx):
    with open(controller._get_state().images[idx], "wb") as fh:
        fh.write(b"not an image")
    controller.file_ctrl._cache.invalidate()


def test_an_roi_drawn_on_an_unreadable_frame_is_clamped(gui, make_sequence):
    window, controller = gui
    controller.file_ctrl.on_folder_selected(str(make_sequence("a", 6)))  # 250 x 400 frames
    _make_unreadable(controller, 3)
    _make_unreadable(controller, 4)
    controller.file_ctrl.on_frame_changed(3)
    controller.file_ctrl.on_frame_changed(4)  # twice in a row: the blank keeps the frame size
    assert controller._get_state().cur_img is None
    assert window.original_panel.image_size == (250, 400)
    controller._on_roi_selected(-30, 260, -40, 420)  # a drag from the view margin
    state = controller._get_state()
    assert state.gridx == (1, 250) and state.gridy == (1, 400)


def test_a_bad_filter_value_in_a_preset_changes_nothing(gui):
    import pytest

    window, _controller = gui
    panel = window.left_panel.measure_panel
    before = panel.get_threshold()
    for bad in ("x", None, float("nan"), True):
        with pytest.raises(ValueError):
            panel.apply_preset({"threshold": 0.31, "gaussian_sigma": bad})
        assert panel.get_threshold() == pytest.approx(before)
    panel.apply_preset({"threshold": 0.31, "gaussian_sigma": 1.5, "area_max_regions": 3})
    assert panel.get_threshold() == pytest.approx(0.31)


def test_a_failed_xlsx_save_leaves_no_temporary_file(tmp_path, monkeypatch):
    import openpyxl
    import pytest

    from bubbletrack.model.export import export_excel

    def fail(self, *_a, **_k):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(openpyxl.writer.excel.ExcelWriter, "write_data", fail)
    with pytest.raises(OSError):
        export_excel(str(tmp_path / "out.xlsx"), np.array([20.0, 22.0]), np.full((2, 2), 50.0))
    assert list(tmp_path.iterdir()) == []


def test_a_target_open_elsewhere_is_named_in_the_error(tmp_path, monkeypatch):
    import pytest

    from bubbletrack.model import atomic_write

    monkeypatch.setattr(atomic_write, "_REPLACE_WAIT_S", 0.0)

    def refuse(*_a, **_k):
        raise PermissionError(13, "Access is denied")

    monkeypatch.setattr(atomic_write.os, "replace", refuse)
    with pytest.raises(PermissionError) as exc:
        atomic_write.write_text_atomic(tmp_path / "results.csv", "x")
    assert str(exc.value).startswith("results.csv is open in another program")
    assert ".tmp" not in str(exc.value)


def test_fit_current_frame_on_a_locked_frame_announces_nothing(gui, make_sequence):
    from bubbletrack.event_bus import RESULTS_CHANGED

    _window, controller = gui
    controller.file_ctrl.on_folder_selected(str(make_sequence("a", 6, r_min=20, r_max=40)))
    controller._set_state(update_state(controller._get_state(), detection_mode="edge"))
    controller.file_ctrl.on_frame_changed(4)
    controller.pretune_ctrl.on_pretune_fit()  # a stored fit, then locked
    controller._on_lock_range(4, 4, locked=True)
    controller.retrack.cancel()
    radius = float(controller._get_state().radius[4])
    events = []
    controller.bus.subscribe(RESULTS_CHANGED, lambda *a: events.append(a))
    controller.pretune_ctrl.on_pretune_fit()
    assert float(controller._get_state().radius[4]) == radius
    assert not events and not controller.retrack.timer.isActive()


def test_a_preset_is_checked_like_the_config_file(gui):
    """Huge or infinite numbers overflowed the sliders (OverflowError escaped the load)."""
    import pytest

    window, _controller = gui
    panel = window.left_panel.measure_panel
    saved = panel.preset()
    for bad in ({"threshold": float("inf")}, {"threshold": 10 ** 400}, {"removing_factor": "x"},
                {"area_max_regions": float("inf")}, {"edge_polarity": "x"}):
        with pytest.raises(ValueError):
            panel.apply_preset({**saved, **bad})
        assert panel.preset() == saved
    panel.apply_preset(saved)  # a saved preset loads back unchanged
    assert panel.preset() == saved
    panel.apply_preset({**saved, "removing_factor": 1e308})  # out of range: clamped, as in config
    assert panel.get_removing_factor_slider() == 100


def test_replacing_a_folder_is_not_called_a_file_held_open(tmp_path):
    """Windows refuses replacing a folder with the same PermissionError as a held file."""
    import pytest

    from bubbletrack.model.atomic_write import write_text_atomic

    (tmp_path / "out.csv").mkdir()
    with pytest.raises(OSError) as exc:
        write_text_atomic(tmp_path / "out.csv", "x")
    assert "open in another program" not in str(exc.value)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out.csv"]


@pytest.mark.skipif(sys.platform != "win32", reason="POSIX renames over a read-only file")
def test_a_read_only_target_is_called_read_only(tmp_path):
    import os
    import stat

    from bubbletrack.model.atomic_write import write_text_atomic

    target = tmp_path / "results.csv"
    target.write_text("old", encoding="utf-8")
    os.chmod(target, stat.S_IREAD)
    try:
        with pytest.raises(PermissionError) as exc:
            write_text_atomic(target, "new")
        assert str(exc.value) == "results.csv is read-only"
        assert target.read_text(encoding="utf-8") == "old"
    finally:
        os.chmod(target, stat.S_IREAD | stat.S_IWRITE)


def test_a_log_rollover_finishes_once_a_held_backup_is_released(tmp_path, monkeypatch):
    """A log viewer held bubbletrack.log.2: the half-done rollover (.rolling) was later
    overwritten, losing a chunk, and the retry threshold was never reset."""
    import logging

    from bubbletrack import logging_config

    log = tmp_path / "bubbletrack.log"
    handler = logging_config._SharedRotatingFileHandler(log, maxBytes=300, backupCount=2,
                                                       encoding="utf-8")
    logger = logging.getLogger("bubbletrack.test_rollover")
    logger.addHandler(handler)
    logger.propagate = False
    logger.setLevel(logging.INFO)
    real_replace, held = logging_config.os.replace, [True]

    def replace(src, dst):
        if held[0] and str(dst).endswith(".log.2"):
            raise PermissionError(32, "being used by another process")
        real_replace(src, dst)

    monkeypatch.setattr(logging_config.os, "replace", replace)
    try:
        for i in range(30):
            logger.info("record %03d %s", i, "x" * 40)
        held[0] = False
        for i in range(30, 60):
            logger.info("record %03d %s", i, "x" * 40)
        handler.flush()
        assert handler._retry_at == 0
        assert not (tmp_path / "bubbletrack.log.rolling").exists()
        newest = "".join((tmp_path / name).read_text(encoding="utf-8")
                         for name in ("bubbletrack.log.1", "bubbletrack.log"))
        assert "record 059" in newest
        sizes = [p.stat().st_size for p in tmp_path.iterdir()]
        assert max(sizes) < 3000  # no file grew far past the limit
    finally:
        logger.removeHandler(handler)
        handler.close()
