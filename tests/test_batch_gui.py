"""Batch processing through the real GUI: folders, videos, Edge mode, failures."""

from __future__ import annotations

import time

import numpy as np
import pytest
from PyQt6.QtWidgets import QFileDialog, QMessageBox

from bubbletrack.controller import batch_folder_worker as bfw
from bubbletrack.model.provenance import FitSource
from bubbletrack.model.state import update_state
from bubbletrack.ui.batch_config_dialog import BatchConfig, BatchConfigDialog
from conftest import write_sequence
from test_video_fitting import write_video

EDGE_RADII = [0.0, 0.0, 25.0, 35.0, 45.0, 55.0, 50.0, 40.0]


def _wait(qapp, predicate, timeout_s=120.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        qapp.processEvents()
        if predicate():
            return
        time.sleep(0.01)
    raise TimeoutError


@pytest.fixture
def batch_gui(qapp, gui, monkeypatch):
    """The GUI with the batch dialogs answered and the worker on threads."""
    window, controller = gui
    monkeypatch.setattr(bfw.BatchFolderWorker, "inline", True)
    monkeypatch.setattr(BatchConfigDialog, "exec",
                        lambda self: BatchConfigDialog.DialogCode.Accepted)
    # Run every source, even ones the dialog would leave unticked
    monkeypatch.setattr(BatchConfigDialog, "get_config",
                        lambda self: BatchConfig(sources=list(self._sources)))
    # What controller.py must add so "Load into Main View" can open a video
    if not controller.bus._subs.get("load_video"):
        controller.bus.subscribe("load_video", controller.file_ctrl.on_video_selected)
    yield window, controller
    controller.auto_ctrl.stop_and_wait()


def _run_batch(qapp, controller, monkeypatch, root, *, mode="intensity"):
    if mode == "edge":
        controller._state = update_state(controller._get_state(), detection_mode="edge")
    else:
        controller._state = update_state(controller._get_state(), img_thr=0.58,
                                         removing_factor=85, gaussian_sigma=3.0)
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a, **k: str(root))
    controller.auto_ctrl.on_batch_folders()
    assert controller.auto_ctrl.is_busy()
    _wait(qapp, lambda: not controller.auto_ctrl.is_busy())
    return controller.auto_ctrl._batch_store


def _browser(controller):
    return controller.auto_ctrl._batch_browser_dlg


def _select(dialog, name):
    names = [r.folder_name for r in dialog._store.results]
    dialog._folder_list.setCurrentRow(names.index(name))


def test_refused_while_another_run_is_active(gui, monkeypatch):
    window, controller = gui
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        lambda *a, **k: pytest.fail("must not ask for a folder"))
    controller.auto_ctrl._running = True
    try:
        controller.auto_ctrl.on_batch_folders()
    finally:
        controller.auto_ctrl._running = False
    assert "already running" in window.header._status_label.text()
    assert controller.auto_ctrl._batch_worker is None


def test_folders_and_videos_with_a_failure(qapp, batch_gui, monkeypatch, tmp_path):
    window, controller = batch_gui
    controller.file_ctrl.on_folder_selected(str(write_sequence(tmp_path / "tuned", [40.0])))
    root = tmp_path / "AVI8"
    write_sequence(root / "exp_a", np.linspace(25, 60, 4))
    (root / "shot_1").mkdir(parents=True)
    video = write_video(root / "shot_1" / "shot_1.avi", np.linspace(25, 70, 6))
    (root / "broken.avi").write_bytes(b"not a video")

    store = _run_batch(qapp, controller, monkeypatch, root)

    by_name = {r.folder_name: r for r in store.results}
    assert set(by_name) == {"broken.avi", "exp_a", "shot_1.avi"}
    assert by_name["exp_a"].success and by_name["shot_1.avi"].success
    assert not by_name["broken.avi"].success
    assert "1 failed" in window.header._status_label.text()

    dialog = _browser(controller)
    assert not dialog._error_label.isHidden()
    assert "broken.avi" in dialog._error_label.text()
    _select(dialog, "broken.avi")
    assert not dialog._load_btn.isEnabled()

    # Load the video result into the main view through the normal video path
    _select(dialog, "shot_1.avi")
    assert dialog._load_btn.isEnabled()
    dialog._load_btn.click()
    state = controller._get_state()
    assert state.video_path == str(video) and state.video_reader is not None
    np.testing.assert_allclose(state.radius, by_name["shot_1.avi"].radius)
    assert (state.fit_source == FitSource.AUTO).all()
    assert "Loaded shot_1.avi" in window.header._status_label.text()


def test_edge_batch_colours_by_confidence_and_restores_it(qapp, batch_gui, monkeypatch,
                                                         tmp_path):
    window, controller = batch_gui
    controller.file_ctrl.on_folder_selected(str(write_sequence(tmp_path / "tuned", EDGE_RADII)))
    root = tmp_path / "root"
    write_sequence(root / "a", EDGE_RADII)
    write_sequence(root / "b", EDGE_RADII[:6])
    (root / "v").mkdir(parents=True)
    write_video(root / "v" / "v.avi", EDGE_RADII)

    store = _run_batch(qapp, controller, monkeypatch, root, mode="edge")

    assert store.detection_mode == "edge"
    assert all(r.success for r in store.results), [r.message for r in store.results]
    dialog = _browser(controller)
    texts = [dialog._folder_list.item(i).text() for i in range(dialog._folder_list.count())]
    assert all("Avg conf" in t and "nan" not in t for t in texts), texts
    assert not dialog._error_label.text()

    _select(dialog, "a")
    dialog._load_btn.click()
    state = controller._get_state()
    result = next(r for r in store.results if r.folder_name == "a")
    np.testing.assert_allclose(state.fit_confidence, result.confidence)
    assert state.fit_flags == [frozenset(f) for f in result.flags]
    assert (state.radius[2:] > 0).all() and (state.radius[:2] == -1).all()


def test_run_error_is_shown_and_the_lock_released(qapp, batch_gui, monkeypatch, tmp_path):
    window, controller = batch_gui
    controller.file_ctrl.on_folder_selected(str(write_sequence(tmp_path / "tuned", [40.0])))
    root = tmp_path / "root"
    write_sequence(root / "a", [30.0, 40.0])
    monkeypatch.setattr(bfw, "as_source", lambda item: (_ for _ in ()).throw(OSError("gone")))

    store = _run_batch(qapp, controller, monkeypatch, root)

    assert "gone" in store.error
    assert "gone" in window.header._status_label.text()
    assert "gone" in _browser(controller)._error_label.text()
    assert not controller.auto_ctrl.is_busy()
    assert window.left_panel.automatic_tab._fit_btn.isEnabled()


def test_no_sources_found(gui, monkeypatch, tmp_path):
    _, controller = gui
    controller.file_ctrl.on_folder_selected(str(write_sequence(tmp_path / "tuned", [40.0])))
    (tmp_path / "empty_root").mkdir()
    shown = []
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        lambda *a, **k: str(tmp_path / "empty_root"))
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: shown.append(a[1]))
    controller.auto_ctrl.on_batch_folders()
    assert shown == ["No Experiments Found"]
    assert controller.auto_ctrl._batch_worker is None
