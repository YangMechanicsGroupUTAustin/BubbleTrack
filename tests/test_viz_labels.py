"""File number next to the frame index: "Frame 1/91 · _011"."""

from __future__ import annotations

import pytest

from bubbletrack.ui.frame_scrubber import (
    FrameScrubber, format_frame_text, frame_file_label, frame_file_labels,
)
from bubbletrack.ui.status_bar import StatusBar


@pytest.mark.parametrize("name, label", [
    ("C:/data/Jin_17_01_56_011.tiff", "_011"),
    ("/x/frame_001.tif", "_001"),
    ("img-0042.png", "-0042"),
    ("shot 7.bmp", " 7"),
    ("IMG0012.TIF", "0012"),
    ("video:0", "#0"),
    ("video:57", "#57"),
    ("background.png", "background"),
    ("a_very_long_name_without_digits.tif", "a_very_long_n\u2026"),
])
def test_frame_file_label(name, label):
    assert frame_file_label(name) == label


def test_format_frame_text():
    assert format_frame_text(1, 91, "_011") == "Frame 1/91 \u00b7 _011"
    assert format_frame_text(3, 5) == "Frame 3/5"


def test_scrubber_label_follows_the_slider(qapp):
    scrubber = FrameScrubber()
    scrubber.set_frame_labels(frame_file_labels([f"f_{i:03d}.tif" for i in range(11, 21)]))
    scrubber.set_range(10)
    assert scrubber.label_text() == "Frame 1/10 \u00b7 _011"
    scrubber.set_value(4)
    assert scrubber.label_text() == "Frame 5/10 \u00b7 _015"
    scrubber.blockSignals(True)  # how controllers move it without a frame change
    scrubber.set_value(9)
    scrubber.blockSignals(False)
    assert scrubber.label_text() == "Frame 10/10 \u00b7 _020"


def test_status_bar_label(qapp):
    bar = StatusBar()
    bar.set_frame_labels(["_011", "_012"])
    bar.update_frame(2, 2)
    assert bar.frame_text() == "Frame 2/2 \u00b7 _012"
    bar.update_frame(5, 9)  # unknown frame: no tag rather than a wrong one
    assert bar.frame_text() == "Frame 5/9"


def test_loaded_sequence_shows_file_numbers_everywhere(gui, make_sequence):
    window, controller = gui
    controller.file_ctrl.on_folder_selected(str(make_sequence("seq", 6)))
    assert window.frame_scrubber.label_text() == "Frame 1/6 \u00b7 _001"
    assert window.status_bar.frame_text() == "Frame 1/6 \u00b7 _001"

    controller.file_ctrl.on_frame_changed(4)  # e.g. arrow key / chart click
    assert window.frame_scrubber.label_text() == "Frame 5/6 \u00b7 _005"
    assert window.status_bar.frame_text() == "Frame 5/6 \u00b7 _005"

    window.frame_scrubber.set_value(2)  # dragging the scrubber
    assert window.status_bar.frame_text() == "Frame 3/6 \u00b7 _003"
    assert window.radius_chart._frame_labels[2] == "_003"


def test_video_frames_show_the_video_frame_number(gui):
    window, controller = gui
    controller.file_ctrl._update(images=tuple(f"video:{i}" for i in range(4)))
    controller.file_ctrl._reset_views(4, ".avi")
    window.frame_scrubber.set_value(3)
    assert window.frame_scrubber.label_text() == "Frame 4/4 \u00b7 #3"
