"""The Auto view draws no extra fill or title note for the reference pixels Auto
cannot use (owner, 2026-11-13: remove the violet "not usable" fill); driven through
the real window offscreen."""

from __future__ import annotations

import cv2
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
