"""The Auto view draws no extra fill or title note for the reference pixels Auto
cannot use (owner, 2026-11-13: remove the violet "not usable" fill); driven through
the real window offscreen."""

from __future__ import annotations

import cv2
import numpy as np
from PyQt6.QtWidgets import QGraphicsPixmapItem

from bubbletrack.controller.display_mixin import ENGINE_VIEW_TITLE
from bubbletrack.model.state import update_state
from conftest import write_sequence

RADII = [0.0, 0.0, 30.0, 40.0, 50.0]  # frame 1: no bubble (the reference)


def test_unusable_reference_pixels_get_no_fill_key_or_note(gui, tmp_path):
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


def test_auto_view_has_no_flat_patches_where_the_reference_is_unusable():
    # Owner: no grey "mask" on the image.  Display only: where the reference frame is too
    # dark or clipped, the Auto view shows the plain frame / reference values instead of a
    # flat background fill; everywhere else it is the engine's own image (measurement unchanged).
    from bubbletrack.model import edge_pipeline
    from bubbletrack.model.edge import EdgeParams
    from bubbletrack.model.edge.engine import engine_view, unusable_mask

    rng = np.random.default_rng(0)
    ref = np.full((120, 160), 30000, np.uint16)
    ref[:30, :] = 300  # almost no light in the reference
    ref[90:110, 10:40] = 65535  # clipped
    frame = (ref * rng.uniform(0.8, 1.2, ref.shape)).clip(0, 65535).astype(np.uint16)
    frame[:30, :] = rng.integers(200, 20000, (30, 160))  # light reaches the band now
    roi = ((1, 120), (1, 160))
    job = edge_pipeline.EdgeJob(params=EdgeParams(), reference=ref, max_radius=160.0)

    shown = edge_pipeline.view_edge(frame, *roi, job)
    engine = engine_view(frame, *roi, reference=ref, params=EdgeParams())
    blind = unusable_mask(ref, *roi)
    assert blind[:30].all()
    assert np.ptp(shown[blind]) > 0.1  # real values, not one flat fill
    assert np.array_equal(shown[~blind], engine[~blind])  # the engine's image elsewhere
    assert np.all(engine[blind] == 1.0)  # the measurement still ignores them
