"""The key-frame strip fills in after a session is loaded (not only after Fit All)."""

from __future__ import annotations

import time

import numpy as np

from bubbletrack.model.provenance import FitSource


def _settle(qapp, seconds: float = 1.0) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)


def test_strip_shows_key_frames_after_session_load(qapp, gui, make_sequence, tmp_path):
    window, controller = gui
    controller.file_ctrl.on_folder_selected(str(make_sequence("A", 12)))
    state = controller._get_state()
    state.radius[:] = np.linspace(20.0, 60.0, 12)
    state.circle_fit_par[:] = [[125.0, 200.0]] * 12
    state.fit_source[:] = FitSource.AUTO
    path = tmp_path / "s.brt"
    assert controller.save_session_file(str(path))

    controller.file_ctrl.on_folder_selected(str(make_sequence("B", 4)))  # something else first
    _settle(qapp, 0.5)
    assert controller.load_session_file(str(path))
    _settle(qapp)
    assert window.filmstrip.frames(), "the strip stays empty after loading a session"
