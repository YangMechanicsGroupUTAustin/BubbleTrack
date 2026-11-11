"""End-to-end: the real Edge engine through the GUI, worker processes included.

No fakes: a synthetic sequence with bubble-free lead-in frames (CI), and the
owner's gelatin sequence compared with the result produced by the previous
app version (local data only).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

from bubbletrack.model.provenance import FitSource
from bubbletrack.model.state import update_state
from conftest import write_sequence

TRUE_RADII = [0.0, 0.0, 25.0, 35.0, 45.0, 55.0, 50.0, 40.0]  # 0 = no bubble
REPO = Path(__file__).resolve().parents[1]
GELATIN = REPO / "test_data" / "lic" / "10_gelatin_clean_flicker"
GELATIN_REF = REPO / "test_data" / "lic" / "refs" / "10_gelatin_Rt_132046.mat"


def _wait(qapp, predicate, timeout_s=180.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        qapp.processEvents()
        if predicate():
            return
        time.sleep(0.01)
    raise TimeoutError


def _fit_all_edge(qapp, window, controller, folder):
    controller.file_ctrl.on_folder_selected(str(folder))
    controller._state = update_state(controller._get_state(), detection_mode="edge")
    controller.auto_ctrl.on_auto_fit()
    _wait(qapp, lambda: window.left_panel.automatic_tab._fit_btn.isEnabled())
    return controller._get_state()


def test_synthetic_sequence(qapp, gui, tmp_path):
    window, controller = gui
    folder = write_sequence(tmp_path / "seq", TRUE_RADII)
    state = _fit_all_edge(qapp, window, controller, folder)
    status = window.header._status_label.text()

    from bubbletrack.controller import edge_support
    assert edge_support.reference_index(state) in (0, 1), "a bubble-free frame is the reference"
    truth = np.array(TRUE_RADII)
    bubble = truth > 0
    assert (state.radius[~bubble] == -1).all(), f"no-bubble frames rejected; {status}"
    np.testing.assert_allclose(state.radius[bubble], truth[bubble], rtol=0.02, err_msg=status)
    assert (state.fit_source[bubble] == FitSource.AUTO).all()
    assert (state.fit_confidence[bubble] >= 0.3).all()


@pytest.mark.skipif(not GELATIN.is_dir(), reason="local test_data not present")
def test_gelatin_matches_previous_app_result(qapp, gui):
    window, controller = gui
    state = _fit_all_edge(qapp, window, controller, GELATIN)

    ref = np.ravel(loadmat(str(GELATIN_REF))["Radius"]).astype(float)
    both = (state.radius > 0) & (ref > 0)
    assert both.sum() > 200, f"only {both.sum()} frames fitted by both"
    diff = np.abs(state.radius[both] - ref[both])
    # Measured at the engine level: median 1.02 px (the old app's edge sits
    # between the steep edge and the outer extent).
    assert np.median(diff) < 1.5
