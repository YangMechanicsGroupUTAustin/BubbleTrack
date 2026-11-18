"""Failures that used to look like success (2026-11-18 silent-failure audit)."""

from __future__ import annotations

import time

import numpy as np

from bubbletrack.controller import worker as worker_module


def _wait(qapp, predicate, timeout_s=60.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        qapp.processEvents()
        if predicate():
            return
        time.sleep(0.01)
    raise TimeoutError


def _fit_all(qapp, controller):
    controller.auto_ctrl.on_auto_fit()
    _wait(qapp, lambda: not controller.auto_ctrl.is_busy())


def test_a_run_that_fails_as_a_whole_is_not_done(qapp, gui, make_sequence, monkeypatch):
    """The pool could not start: "Fitting failed" was replaced by a green "Done: 6 of 6
    frames measured" (the previous run's results), and those were tracked."""
    from bubbletrack.event_bus import RUN_FINISHED

    window, controller = gui
    monkeypatch.setattr(worker_module.BatchWorker, "inline", True)
    controller.file_ctrl.on_folder_selected(str(make_sequence("A", 6)))
    _fit_all(qapp, controller)
    first = controller._get_state().radius.copy()

    def broken(self):
        raise RuntimeError("process pool could not start")

    monkeypatch.setattr(worker_module.BatchWorker, "_task", broken)
    tracked, finished = [], []
    monkeypatch.setattr(controller.auto_ctrl, "_start_tracking", lambda: tracked.append(1))
    controller.bus.subscribe(RUN_FINISHED, lambda info: finished.append(info))
    _fit_all(qapp, controller)
    text, colour = window.header.status()
    assert text.startswith("Stopped after 0 frames") and "process pool could not start" in text
    assert colour != "#10b981"
    assert not tracked and finished[-1]["stopped"]
    np.testing.assert_array_equal(controller._get_state().radius, first)


def _write_video(path, radii):
    import cv2

    from _golden_cases import SceneSpec, render_scene

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (400, 250), False)
    for i, r in enumerate(radii):
        scene = render_scene(SceneSpec(radius=float(r), center=(125.0, 200.0), shape=(250, 400), seed=i))
        writer.write((scene // 160).astype(np.uint8))
    writer.release()


def test_a_video_session_never_restores_onto_another_shot(qapp, gui, tmp_path, monkeypatch):
    """Same path and frame count (every HPV-X2 shot has 256): A's radii appeared on B."""
    window, controller = gui
    monkeypatch.setattr(worker_module.BatchWorker, "inline", True)
    shot = tmp_path / "shot.avi"
    _write_video(shot, np.linspace(30, 60, 6))
    controller.file_ctrl.on_video_selected(str(shot), ask=False)
    _fit_all(qapp, controller)
    session = tmp_path / "a.brt"
    assert controller.session_ctrl.save(str(session))

    _write_video(shot, np.full(6, 15.0))  # another shot written over the same file
    controller.file_ctrl.on_video_selected(str(shot), ask=False)
    assert controller.session_ctrl.load(str(session))
    assert not np.any(controller._get_state().radius > 0)
    assert "results were not restored" in window.header.status()[0]


def test_a_folder_session_never_restores_onto_other_frames_with_the_same_names(
        gui, make_sequence, tmp_path):
    window, controller = gui
    folder = make_sequence("A", 6)
    controller.file_ctrl.on_folder_selected(str(folder))
    controller._get_state().radius[:] = [30.0, 31.0, 32.0, 33.0, 34.0, 35.0]
    session = tmp_path / "a.brt"
    assert controller.session_ctrl.save(str(session))

    assert controller.session_ctrl.load(str(session))  # unchanged files: restored
    assert controller._get_state().radius[5] == 35.0
    make_sequence("A", 6, r_min=10, r_max=12)  # the next shot exported into the same folder
    assert controller.session_ctrl.load(str(session))
    assert not np.any(controller._get_state().radius > 0)
    assert "results were not restored" in window.header.status()[0]


def test_a_failed_close_autosave_keeps_the_last_one_on_offer(gui, make_sequence, monkeypatch,
                                                              isolated_config):
    """The close autosave failed (disk full) but the clean exit was still recorded, so
    the next launch offered nothing, not even the periodic autosave still on disk."""
    from bubbletrack import app as app_module
    from bubbletrack.controller import session_controller as sc

    _window, controller = gui
    controller.file_ctrl.on_folder_selected(str(make_sequence("B", 4)))
    controller._get_state().radius[:] = [11.0, 12.0, 13.0, 14.0]
    session = controller.session_ctrl
    session.notify_results_changed()
    assert session.autosave_if_due()  # a periodic autosave on disk

    def disk_full(_path, _state):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(sc, "save_session", disk_full)
    assert controller._confirm_close()
    assert session.close_save_failed
    app_module.on_about_to_quit(controller._get_state(), saved=not session.close_save_failed)
    assert sc.claim_pending_recovery() is not None


def test_a_rejected_rmax_fit_is_said(tmp_path):
    """A noisy plateau gives a convex best-fit parabola: Rmax and t=0 quietly became the
    largest measured frame, with RmaxFitLength still recorded as if fitted."""
    from bubbletrack.model.export import export_rof_t_data
    from bubbletrack.model.rmax import locate_rmax

    plateau = [49, 47, 46.5, 47, 50, 47, 46.5, 47, 49]
    radius = np.array(list(np.linspace(10, 45, 8)) + plateau + list(np.linspace(45, 10, 8)))
    estimate = locate_rmax(radius, 9)
    assert not estimate.fitted
    assert any("rejected" in w for w in estimate.warnings)
    _ok, message = export_rof_t_data(str(tmp_path / "r.mat"), radius, 3.2, 1e6, 9)
    assert "rejected" in message
    clear = 50.0 - 0.1 * (np.arange(21) - 10.0) ** 2
    assert not locate_rmax(clear, 9).warnings  # a clear peak: nothing to say


def test_fit_current_on_a_kept_frame_says_what_it_keeps(gui, make_sequence):
    """It showed "R = 40.0 px" in green while the frame (and the export) kept 33.0 px."""
    from bubbletrack.model.state import update_state

    window, controller = gui
    controller.file_ctrl.on_folder_selected(str(make_sequence("a", 6, r_min=20, r_max=40)))
    for mode in ("edge", "intensity"):
        controller._set_state(update_state(controller._get_state(), detection_mode=mode, img_thr=0.58))
        controller.file_ctrl.on_frame_changed(3)
        controller.pretune_ctrl.on_pretune_fit()
        controller._on_lock_range(3, 3, locked=True)
        controller.retrack.cancel()
        controller._get_state().radius[3] = 33.0  # the kept value differs from a new fit
        controller.pretune_ctrl.on_pretune_fit()
        text, colour = window.header.status()
        assert text.startswith("Frame 4 kept R = 33.0 px") and colour != "#10b981", (mode, text)
        controller._on_lock_range(3, 3, locked=False)


def test_a_frame_whose_fit_raises_is_not_called_unreadable(qapp, gui, make_sequence, monkeypatch):
    """Any exception of the frame task was summarised as "could not be read", and the
    frame silently kept a result measured with other parameters."""
    window, controller = gui
    monkeypatch.setattr(worker_module.BatchWorker, "inline", True)
    controller.file_ctrl.on_folder_selected(str(make_sequence("A", 6)))
    _fit_all(qapp, controller)
    real = worker_module.BatchWorker._task

    def task(self):
        process, kwargs = real(self)

        def failing(args):  # the fit itself raises on frame 3 (inline: threads)
            if args[0] == 2:
                raise ValueError("reference shape (250, 400) differs from frame shape (120, 200)")
            return process(args)
        return failing, kwargs

    monkeypatch.setattr(worker_module.BatchWorker, "_task", task)
    _fit_all(qapp, controller)
    text = window.header.status()[0]
    assert "1 frame failed and kept their previous result" in text and "could not be read" not in text
