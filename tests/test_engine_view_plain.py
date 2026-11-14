"""The Auto view draws no extra fill or title note over dark or clipped areas of the
reference frame (owner, 2026-11-13: remove the violet "not usable" fill; 2026-11-14: no
pixel is excluded at all, so the view is the engine's image as it is); the first test
drives the real window offscreen."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from PyQt6.QtWidgets import QGraphicsPixmapItem

from bubbletrack.controller.display_mixin import ENGINE_VIEW_TITLE
from bubbletrack.model.state import update_state
from conftest import write_sequence

RADII = [0.0, 0.0, 30.0, 40.0, 50.0]  # frame 1: no bubble (the reference)


def test_dark_or_clipped_reference_areas_get_no_fill_key_or_note(gui, tmp_path):
    window, controller = gui
    folder = write_sequence(tmp_path / "seq", RADII)
    for path in sorted(folder.glob("*.tiff")):  # a dark band and glare in every frame
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        img[:90, :] = 150
        img[180:210, 20:60] = 65535
        assert cv2.imwrite(str(path), img)
    controller.file_ctrl.on_folder_selected(str(folder))
    controller._state = update_state(controller._get_state(), detection_mode="edge",
                                     reference_frame=0)
    controller.file_ctrl.on_frame_changed(3)
    panel = window.binary_panel
    assert panel._title_label.text() == ENGINE_VIEW_TITLE
    assert not getattr(panel, "_title_note", None)  # no note label after the title
    assert not [i for i in panel._overlay_items if isinstance(i, QGraphicsPixmapItem)]
    assert not any("not usable" in text for text in panel.legend_texts())


def test_auto_view_is_the_engine_image_with_no_flat_patch():
    # Owner: no grey "mask" on the image, and no pixel is excluded (edge-1.4.0): where the
    # reference frame is very dark or clipped the Auto view shows the real frame / reference
    # ratio, and it is exactly the image the engine measures
    from bubbletrack.model import edge_pipeline
    from bubbletrack.model.edge import EdgeParams
    from bubbletrack.model.edge.engine import engine_view

    rng = np.random.default_rng(0)
    ref = rng.normal(30000, 300, (120, 160))
    ref[:30, :] *= 0.03  # almost no light in the reference (the old rule excluded it)
    frame = ref * rng.uniform(0.9, 1.1, ref.shape)
    ref[90:110, 10:40] = 65535  # clipped in the reference only
    ref, frame = (np.round(a).clip(0, 65535).astype(np.uint16) for a in (ref, frame))
    roi = ((1, 120), (1, 160))
    job = edge_pipeline.EdgeJob(params=EdgeParams(), reference=ref, max_radius=160.0)

    shown = edge_pipeline.view_edge(frame, *roi, job)
    np.testing.assert_array_equal(shown, engine_view(frame, *roi, reference=ref, params=EdgeParams()))
    assert np.all(np.isfinite(shown))
    for area in (shown[:28, :], shown[92:108, 12:38]):
        assert np.std(area) > 0.01 and not np.any(area == 1.0)  # real values, not one flat fill
    assert np.median(shown[:28, :]) == pytest.approx(1.0, abs=0.05)  # the static dark band divides out
