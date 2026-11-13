"""One product name and version, readable source names, proper plurals."""

from __future__ import annotations

from PyQt6.QtWidgets import QLabel

import bubbletrack
from bubbletrack.controller.file_controller import recording_kind


def test_one_product_name_and_version(qapp, gui):
    from bubbletrack.model.export_meta import build_export_metadata
    from bubbletrack.model.state import AppState
    from bubbletrack.ui.welcome_dialog import WelcomeDialog

    window, _ = gui
    assert bubbletrack.APP_TITLE == "BubbleTrack 3.1" and bubbletrack.__version__ == "3.1.0"
    assert window.windowTitle() == "BubbleTrack 3.1"
    assert window.header.findChildren(QLabel)[0].text() == "BubbleTrack"
    welcome = WelcomeDialog()
    assert welcome.windowTitle() == "Welcome to BubbleTrack 3.1"
    assert any("BubbleTrack 3.1" in lbl.text() for lbl in welcome.findChildren(QLabel))
    meta = build_export_metadata(AppState(), fps=1.0, um2px=1.0, rmax_fit_length=5)
    assert meta["app_version"] == "3.1.0"


def test_a_long_folder_name_is_elided_and_the_count_stays(qapp, gui, make_sequence):
    window, controller = gui
    name = "06_multipulse_wall_half_bubble_speckle_with_a_very_long_name"
    folder = make_sequence(name, 3)
    zone = window.left_panel.image_source._folder_zone
    zone.resize(150, 90)
    hint = zone.sizeHint().width()
    controller.file_ctrl.on_folder_selected(str(folder))
    first, second = zone.summary_text().split("\n")
    assert first.startswith("✓ 06_") and "…" in first
    assert second == "3 images"
    assert zone.sizeHint().width() == hint  # the name does not widen the zone
    assert zone._prompt_label.toolTip() == str(folder)


def test_a_dat_file_is_named_as_an_hpv_x2_recording():
    assert recording_kind("C:/data/Jin_14_42_03.dat") == "HPV-X2 recording"
    assert recording_kind("clip.AVI") == "video"


def test_counts_use_singular_and_plural(gui, make_sequence):
    window, controller = gui
    controller.file_ctrl.on_folder_selected(str(make_sequence("A", 4)))
    controller._get_state().radius[:] = 20.0
    controller._on_lock_range(1, 1, locked=True)
    assert window.header.status_text() == "Locked 1 frame"
    controller._on_lock_range(0, 2, locked=True)
    assert window.header.status_text() == "Locked 3 frames"
