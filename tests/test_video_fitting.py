"""Video sources must be fittable, not only viewable.

Video frames are identified by placeholder ids ("video:<i>") in
``state.images``; every fitting path used to pass those ids to cv2.imread.
"""

from __future__ import annotations

import time

import cv2
import numpy as np
import pytest

from _golden_cases import SceneSpec, render_scene
from bubbletrack.controller.worker import _process_single_frame
from bubbletrack.model.provenance import FitSource
from bubbletrack.model.state import update_state


def write_video(path, radii, fps=10.0):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (400, 250))
    if not writer.isOpened():
        pytest.skip("no MJPG video writer available")
    for i, r in enumerate(radii):
        gray = render_scene(SceneSpec(radius=float(r), center=(125.0, 200.0),
                                      bit_depth=8, seed=i))
        writer.write(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
    writer.release()
    return path


@pytest.fixture
def video(tmp_path):
    return write_video(tmp_path / "shot.avi", np.linspace(25, 70, 12))


def _owner_params(controller):
    controller._state = update_state(controller._get_state(), img_thr=0.58,
                                     removing_factor=85, gaussian_sigma=3.0)


def _wait(qapp, predicate, timeout_s=90.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        qapp.processEvents()
        if predicate():
            return
        time.sleep(0.01)
    raise TimeoutError


def _status(window):
    return window.header._status_label.text()


def test_worker_accepts_raw_frames_identically(tmp_path):
    frame = render_scene(SceneSpec(radius=50.0, center=(125.0, 200.0), seed=3))
    path = tmp_path / "f.tiff"
    cv2.imwrite(str(path), frame)
    common = (0.58, (1, 250), (1, 400), (False,) * 4, 85.0, 0, 3.0, 0.0, 0, 0,
              "intensity", 0.0, 1, 400.0)
    from_path = _process_single_frame((0, str(path)) + common)
    from_array = _process_single_frame((0, frame) + common)
    assert from_path[1] == from_array[1]
    np.testing.assert_array_equal(from_path[2], from_array[2])


def test_fit_all_on_video(qapp, gui, video):
    window, controller = gui
    controller.file_ctrl.on_video_selected(str(video))
    _owner_params(controller)
    controller.auto_ctrl.on_auto_fit()
    _wait(qapp, lambda: not controller.auto_ctrl.is_busy())
    state = controller._get_state()
    assert (state.radius > 0).all(), _status(window)
    assert (state.fit_source == FitSource.AUTO).all()
    assert state.radius[-1] > state.radius[0]


def test_fit_current_frame_on_video(gui, video):
    window, controller = gui
    controller.file_ctrl.on_video_selected(str(video))
    _owner_params(controller)
    controller.file_ctrl.on_frame_changed(5)
    controller.pretune_ctrl.on_pretune_fit()
    assert controller._get_state().radius[5] > 0, _status(window)


def test_refit_point_on_video(gui, video):
    window, controller = gui
    controller.file_ctrl.on_video_selected(str(video))
    _owner_params(controller)
    controller._on_refit_point(4)
    assert controller._get_state().radius[4] > 0, _status(window)


def test_autotune_accepts_a_raw_frame():
    from bubbletrack.model.autotune import run_autotune

    frame = render_scene(SceneSpec(radius=50.0, center=(125.0, 200.0), bit_depth=8, seed=1))
    result = run_autotune(frame, (1, 250), (1, 400), (False,) * 4, gaussian_sigma=3.0,
                          max_radius=400.0)
    assert result is not None


def test_video_session_round_trip(gui, video, tmp_path):
    window, controller = gui
    controller.file_ctrl.on_video_selected(str(video))
    state = controller._get_state()
    state.radius[:] = np.arange(1, 13, dtype=float)
    path = tmp_path / "v.brt"
    assert controller.save_session_file(str(path))
    controller.file_ctrl.on_video_selected(str(video))  # reset results
    assert controller.load_session_file(str(path))
    state = controller._get_state()
    assert state.video_path == str(video)
    np.testing.assert_allclose(state.radius, np.arange(1, 13))
