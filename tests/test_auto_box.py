"""The Auto box: one compact line per setting, tooltips, and a guarded reference.

"Use this frame" on a frame that holds a bubble asks first (the reference must
be bubble-free); the reference line says why there is none; the bubble position
reads "auto · click to set"; the ML assist line appears once it is relevant.
"""

from __future__ import annotations

import numpy as np
import pytest
from PyQt6.QtWidgets import QPushButton

from _golden_cases import SceneSpec, render_scene
from bubbletrack.controller import edge_setup
from bubbletrack.model import edge_pipeline
from bubbletrack.ui.edge_options import AUTO_POSITION_TEXT, EdgeOptionsBox


def _box(window) -> EdgeOptionsBox:
    return window.left_panel.measure_panel.edge_box


def _status(window) -> str:
    return window.header.status_text()


@pytest.fixture
def asked(gui):
    """Record 'Use it anyway?' questions; answer with ``asked.answer``."""
    _window, controller = gui

    class Asked(list):
        answer = False

    record = Asked()

    def ask(_parent, frame_no):
        record.append(frame_no)
        return record.answer

    controller.pretune_ctrl.edge_setup.ask_bubble_reference = ask
    return record


# -- the engine's bubble test ------------------------------------------------- #

def test_frame_has_bubble_uses_the_reference_search_test():
    bubble = render_scene(SceneSpec(radius=40.0, center=(125.0, 200.0), shape=(250, 400), seed=1))
    empty = render_scene(SceneSpec(radius=0.0, center=(125.0, 200.0), shape=(250, 400), seed=2))
    assert edge_pipeline.frame_has_bubble(bubble)
    assert not edge_pipeline.frame_has_bubble(empty)


# -- layout ---------------------------------------------------------------------- #

def test_every_button_has_a_tooltip_and_the_new_names(qapp):
    box = EdgeOptionsBox()
    texts = {b.text(): b.toolTip() for b in box.findChildren(QPushButton)}
    for name in ("Use this frame", "Auto", "Click bubble", "Clear", "Teach…"):
        assert texts.get(name), f"{name!r} has no tooltip"
    assert "Use current" not in texts


def test_lines_show_value_and_hide_buttons_with_nothing_to_undo(qapp):
    box = EdgeOptionsBox()
    box.show()
    assert box.seed_text() == AUTO_POSITION_TEXT
    assert box.use_frame_button.isVisible() and not box.auto_reference_button.isVisible()
    assert box.pick_button.isVisible() and not box.clear_seed_button.isVisible()
    assert not box.ml_line.isVisible()  # nothing to teach before a fit
    box.set_reference_text("Frame 11 (set)", user_set=True)
    box.set_seed_text("x 200, y 125", user_set=True)
    box.set_ml_state("Off · not taught yet", False, False, relevant=True)
    assert box.auto_reference_button.isVisible() and box.clear_seed_button.isVisible()
    assert box.ml_line.isVisible() and not box.ml_switch.isEnabled()
    assert "teach" in box.ml_switch.toolTip().lower()  # why the switch is off


def test_reference_and_position_each_fit_on_one_line(qapp):
    """Name, value and buttons share a row (no second row of buttons)."""
    box = EdgeOptionsBox()
    grid = box.layout()
    for row in (0, 1):
        assert grid.itemAtPosition(row, 0) is not None and grid.itemAtPosition(row, 2) is not None
    assert grid.itemAtPosition(0, 2).layout().indexOf(box.use_frame_button) >= 0
    assert grid.itemAtPosition(1, 2).layout().indexOf(box.pick_button) >= 0


# -- the reference --------------------------------------------------------------- #

def test_use_this_frame_on_a_bubble_frame_asks_and_cancel_keeps_auto(
        gui, make_sequence, asked, monkeypatch):
    window, controller = gui
    monkeypatch.setattr(edge_pipeline, "frame_has_bubble", lambda raw: True)
    controller.file_ctrl.on_folder_selected(str(make_sequence("A", 5)))
    controller.file_ctrl.on_frame_changed(3)
    _box(window).use_current_reference_clicked.emit()
    assert asked == [4]
    assert controller._get_state().reference_frame is None
    assert "unchanged" in _status(window)


def test_use_it_anyway_sets_the_reference_with_a_warning(gui, make_sequence, asked, monkeypatch):
    window, controller = gui
    monkeypatch.setattr(edge_pipeline, "frame_has_bubble", lambda raw: True)
    controller.file_ctrl.on_folder_selected(str(make_sequence("A", 5)))
    controller.file_ctrl.on_frame_changed(3)
    asked.answer = True
    _box(window).use_current_reference_clicked.emit()
    assert controller._get_state().reference_frame == 3
    assert _box(window).reference_text() == "Frame 4 (set)"
    assert _box(window).auto_reference_button.isVisibleTo(_box(window))
    assert "bubble" in _status(window)
    _box(window).auto_reference_clicked.emit()
    assert controller._get_state().reference_frame is None
    assert not _box(window).auto_reference_button.isVisibleTo(_box(window))


def test_bubble_free_frame_is_used_without_a_question(gui, make_sequence, asked, monkeypatch):
    window, controller = gui
    monkeypatch.setattr(edge_pipeline, "frame_has_bubble", lambda raw: False)
    controller.file_ctrl.on_folder_selected(str(make_sequence("A", 5)))
    controller.file_ctrl.on_frame_changed(1)
    _box(window).use_current_reference_clicked.emit()
    assert asked == [] and controller._get_state().reference_frame == 1
    assert _status(window) == "Reference: frame 2"


def test_the_question_names_the_frame_and_defaults_to_cancel(qapp):
    box, use = edge_setup.bubble_reference_dialog(None, 11)
    assert box.text() == ("Frame 11 seems to contain a bubble; the reference must be "
                          "bubble-free. Use it anyway?")
    assert use.text() == "Use it anyway" and box.defaultButton() is not use


@pytest.mark.parametrize(("first_has_bubble", "text"), [
    (True, edge_setup.NO_REFERENCE_NEEDED),
    (False, edge_setup.NO_REFERENCE_FOUND),
])
def test_no_reference_says_why(gui, make_sequence, monkeypatch, first_has_bubble, text):
    window, controller = gui
    monkeypatch.setattr(edge_pipeline, "_choose_reference", lambda read_raw, n: None)
    monkeypatch.setattr(edge_pipeline, "frame_has_bubble", lambda raw: first_has_bubble)
    edge_setup._first_frame_bubble.clear()
    controller.file_ctrl.on_folder_selected(str(make_sequence(f"B{int(first_has_bubble)}", 4)))
    assert _box(window).reference_text() == text


# -- the bubble position ---------------------------------------------------------------- #

def test_bubble_position_line(gui, make_sequence):
    window, controller = gui
    controller.file_ctrl.on_folder_selected(str(make_sequence("A", 3)))
    box = _box(window)
    assert box.seed_text() == AUTO_POSITION_TEXT
    box.pick_seed_clicked.emit()
    window.original_panel.point_clicked.emit(199.5, 124.5)
    assert box.seed_text() == "x 200, y 125"
    assert box.clear_seed_button.isVisibleTo(box)
    assert _status(window) == "Bubble position set"
    box.clear_seed_clicked.emit()
    assert box.seed_text() == AUTO_POSITION_TEXT and controller._get_state().edge_seed is None


# -- ML assist line -------------------------------------------------------------------- #

def test_ml_line_appears_after_a_fit(gui, make_sequence):
    window, controller = gui
    controller.file_ctrl.on_folder_selected(str(make_sequence("A", 3)))
    box = _box(window)
    assert not box.ml_line.isVisibleTo(box)
    state = controller._get_state()
    state.radius[:] = np.array([20.0, 30.0, 40.0])
    controller.ml.refresh_state_line()
    assert box.ml_line.isVisibleTo(box)
    assert box.ml_switch.text() == "Off · not taught yet" and not box.ml_switch.isEnabled()
