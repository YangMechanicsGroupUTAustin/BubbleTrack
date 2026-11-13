"""A marker file keeps Fit All from writing its autosave into reference data."""

from __future__ import annotations

from bubbletrack.controller.auto_controller import (
    AUTOSAVE_NAME, NO_AUTOSAVE_MARKER, autosave_disabled,
)


def test_marker_in_a_parent_disables_the_autosave(tmp_path):
    data = tmp_path / "data" / "lic" / "rec"
    data.mkdir(parents=True)
    assert not autosave_disabled(data)
    (tmp_path / "data" / NO_AUTOSAVE_MARKER).write_text("", encoding="utf-8")
    assert autosave_disabled(data)


def test_fit_all_writes_no_autosave_under_a_marker(gui, make_sequence, tmp_path):
    window, controller = gui
    folder = make_sequence("A", 4)
    (folder.parent / NO_AUTOSAVE_MARKER).write_text("", encoding="utf-8")
    controller.file_ctrl.on_folder_selected(str(folder))
    assert controller.auto_ctrl._autosave_path() is None
    assert controller.auto_ctrl._write_autosave() is False
    assert not (folder / AUTOSAVE_NAME).exists()


def test_without_a_marker_the_autosave_goes_next_to_the_images(gui, make_sequence):
    window, controller = gui
    folder = make_sequence("B", 4)
    controller.file_ctrl.on_folder_selected(str(folder))
    assert controller.auto_ctrl._autosave_path() == folder / AUTOSAVE_NAME
