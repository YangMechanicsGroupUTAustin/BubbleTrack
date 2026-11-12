"""Raw HPV-X2 .dat recordings work everywhere a video works."""

from __future__ import annotations

import time

import numpy as np
import pytest

from _golden_cases import SceneSpec, render_scene
from bubbletrack.model.frame_source import is_recording_file, open_recording
from bubbletrack.model.hpvx2_dat import DatFrameReader
from bubbletrack.model.state import update_state
from test_hpvx2_dat import build_dat


def _bubble_frames(radii):
    return np.stack([render_scene(SceneSpec(radius=float(r), center=(125.0, 200.0), seed=i))
                     for i, r in enumerate(radii)])


@pytest.fixture
def dat_file(tmp_path):
    path = tmp_path / "Jin_12_00_00.dat"
    path.write_bytes(build_dat(_bubble_frames(np.linspace(25, 60, 6))))
    return path


def test_recording_detection_and_factory(dat_file, tmp_path):
    assert is_recording_file(str(dat_file))
    assert is_recording_file("shot.AVI")
    assert not is_recording_file("frame.tiff")
    reader = open_recording(str(dat_file))
    assert isinstance(reader, DatFrameReader)
    assert reader.total_frames == 6
    reader.close()


def test_open_dat_in_main_view(gui, dat_file):
    window, controller = gui
    controller.file_ctrl.on_video_selected(str(dat_file))
    state = controller._get_state()
    assert len(state.images) == 6
    assert isinstance(state.video_reader, DatFrameReader)
    assert state.fps == pytest.approx(1e6)


def test_fit_all_on_dat(qapp, gui, dat_file):
    window, controller = gui
    controller.file_ctrl.on_video_selected(str(dat_file))
    controller._state = update_state(controller._get_state(), img_thr=0.58,
                                     removing_factor=85, gaussian_sigma=3.0)
    controller.auto_ctrl.on_auto_fit()
    deadline = time.monotonic() + 120
    while not window.left_panel.automatic_tab._fit_btn.isEnabled():
        assert time.monotonic() < deadline
        qapp.processEvents()
        time.sleep(0.01)
    state = controller._get_state()
    assert (state.radius > 0).all(), window.header._status_label.text()


def test_dropping_a_dat_file_opens_it(gui, dat_file):
    window, _ = gui
    source = window.left_panel.image_source
    seen = []
    source.video_selected.connect(seen.append)
    source._video_zone.accept_path(str(dat_file))
    assert seen == [str(dat_file)]


def test_rate_label_uses_si_prefix():
    from bubbletrack.ui.image_source import format_rate

    assert format_rate(29.97) == "29.97 fps"
    assert format_rate(50_000) == "50 kfps"
    assert format_rate(5e6) == "5 Mfps"


def test_batch_discovers_dat_files(tmp_path):
    from bubbletrack.model.batch_experiments import find_batch_sources

    shot = tmp_path / "Jin_12_00_00"
    shot.mkdir()
    (shot / "Jin_12_00_00.dat").write_bytes(build_dat(_bubble_frames([30, 40])))
    sources = find_batch_sources(str(tmp_path))
    assert [s.path.name for s in sources] == ["Jin_12_00_00.dat"]


def test_load_video_event_opens_recording(gui, dat_file):
    _, controller = gui
    controller.bus.emit("load_video", str(dat_file))
    assert controller._get_state().video_path == str(dat_file)
