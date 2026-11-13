"""The Auto view shows the reference pixels Auto cannot use (too dark or saturated in the
reference frame) as a violet fill with a key line, and says so in its title when they cover
more than a tenth of the ROI; driven through the real window offscreen."""

from __future__ import annotations

import re

import cv2
import pytest
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QGraphicsPixmapItem

from bubbletrack.controller import unusable_view
from bubbletrack.controller.display_mixin import ENGINE_VIEW_TIP, ENGINE_VIEW_TITLE, MASK_TITLE
from bubbletrack.model.edge import engine
from bubbletrack.model.state import update_state
from bubbletrack.ui.overlay_legend import UNUSABLE_COLOUR, default_text
from conftest import write_sequence

RADII = [0.0, 0.0, 30.0, 40.0, 50.0]  # frame 1: no bubble (the reference)
KEY = "not usable (reference)"
GLARE = (slice(180, 210), slice(20, 60))  # a saturated block, far from the bubble


def _add_static_features(folder, band_rows: int, glare: bool) -> None:
    """A dark band along the top (no light) and a saturated block in every frame:
    parts of the setup, so the reference frame has them too."""
    for path in sorted(folder.glob("*.tiff")):
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        img[:band_rows, :] = 150
        if glare:
            img[GLARE] = 65535
        assert cv2.imwrite(str(path), img)


@pytest.fixture
def open_auto(gui, tmp_path):
    """Load a sequence (with a dark band of *band_rows* rows, and glare) in Auto mode,
    frame 1 as the reference, and show *frame*."""
    window, controller = gui

    def _open(band_rows: int = 0, glare: bool = False, frame: int = 3):
        folder = write_sequence(tmp_path / f"seq_{band_rows}_{glare}", RADII)
        if band_rows or glare:
            _add_static_features(folder, band_rows, glare)
        controller.file_ctrl.on_folder_selected(str(folder))
        controller._state = update_state(controller._get_state(), detection_mode="edge",
                                         reference_frame=0)
        controller.file_ctrl.on_frame_changed(frame)
        return window, controller

    return _open


def _fills(panel) -> list[QGraphicsPixmapItem]:
    return [item for item in panel._overlay_items if isinstance(item, QGraphicsPixmapItem)]


def _is_fill(colour: QColor) -> bool:
    """The violet fill (the pixmap premultiplies alpha: the colour may be off by a unit)."""
    want = QColor(UNUSABLE_COLOUR)
    close = all(abs(a - b) <= 2 for a, b in zip(colour.getRgb()[:3], want.getRgb()[:3]))
    return close and colour.alpha() == unusable_view.FILL_ALPHA


def _note_percent(panel) -> int:
    match = re.fullmatch(r"· (\d+) % not usable", panel.title_note())
    assert match, panel.title_note()
    return int(match.group(1))


def test_the_key_text_is_a_known_overlay_colour():
    assert default_text(UNUSABLE_COLOUR, "fill") == KEY


def test_a_large_unusable_part_is_filled_keyed_and_noted(open_auto):
    window, controller = open_auto(band_rows=50, glare=True)
    panel = window.binary_panel
    assert panel._title_label.text() == ENGINE_VIEW_TITLE
    fills = _fills(panel)
    assert len(fills) == 1
    image = fills[0].pixmap().toImage()  # pixelColor(x = column, y = row)
    band, clear, glare = image.pixelColor(200, 5), image.pixelColor(350, 150), image.pixelColor(40, 195)
    assert _is_fill(band) and _is_fill(glare)
    assert clear.alpha() == 0
    assert KEY in panel.legend_texts()
    state = controller._get_state()
    share = engine.unusable_mask(cv2.imread(state.images[0], cv2.IMREAD_UNCHANGED),
                                 state.gridx, state.gridy).mean()
    assert share > 0.2
    assert _note_percent(panel) == round(100 * share)
    tip = panel._title_label.toolTip()
    assert tip.startswith(ENGINE_VIEW_TIP) and tip == panel._title_note.toolTip()
    for words in ("too dark or saturated", "reference frame", "Use this frame", "ROI"):
        assert words in tip, words


def test_a_small_unusable_part_is_filled_and_keyed_without_a_note(open_auto):
    window, _ = open_auto(band_rows=10, glare=True)
    panel = window.binary_panel
    assert len(_fills(panel)) == 1 and KEY in panel.legend_texts()
    assert panel.title_note() == "" and panel._title_note.isHidden()
    assert panel._title_label.toolTip() == ENGINE_VIEW_TIP


def test_a_clean_reference_shows_no_fill(open_auto):
    window, _ = open_auto()
    panel = window.binary_panel
    assert _fills(panel) == [] and KEY not in panel.legend_texts()
    assert panel.title_note() == "" and panel._title_label.toolTip() == ENGINE_VIEW_TIP


def test_the_image_panel_is_unchanged(open_auto):
    window, _ = open_auto(band_rows=50, glare=True)
    left = window.original_panel
    assert _fills(left) == [] and left.legend_texts() == ["ROI"]
    assert left._title_note.isHidden()


def test_the_outline_is_drawn_over_the_fill_after_a_fit(open_auto):
    window, controller = open_auto(band_rows=50, glare=True)
    controller.pretune_ctrl.on_pretune_fit()
    panel = window.binary_panel
    assert panel.legend_texts() == ["outline (measured)", KEY]
    outline = [item for item in panel._overlay_items if not isinstance(item, QGraphicsPixmapItem)]
    assert outline and all(item.zValue() > _fills(panel)[0].zValue() for item in outline)


def test_a_refused_circle_keeps_the_fill(open_auto):
    """Fix's refused-circle view shows only the decision's overlays, but the fill describes
    the image (the wall may lie where Auto cannot see)."""
    window, controller = open_auto(band_rows=50, glare=True)
    assert controller.review.start_fix(3)
    window.original_panel.circle_drawn.emit(200.5, 125.5, 90.0, True)  # twice too big: refused
    assert controller.review.fix.strip.prompt_visible()
    right = window.binary_panel
    assert "your circle (not stored)" in right.legend_texts() and KEY in right.legend_texts()
    assert len(_fills(right)) == 1 and _note_percent(right) > 20
    assert KEY not in window.original_panel.legend_texts()


def test_the_mask_is_computed_once_while_scrubbing(open_auto, monkeypatch):
    calls = []
    real = engine.unusable_mask

    def counting(*args, **kwargs):
        calls.append(args[1:])
        return real(*args, **kwargs)

    monkeypatch.setattr(engine, "unusable_mask", counting)
    window, controller = open_auto(band_rows=50, glare=True, frame=2)
    assert calls  # computed for the frame on screen
    calls.clear()
    for idx in (3, 4, 2):
        controller.file_ctrl.on_frame_changed(idx)
        assert _fills(window.binary_panel)
    assert calls == []  # the same reference and ROI: reused
    controller._state = update_state(controller._get_state(), gridx=(1, 200), gridy=(1, 400))
    controller.file_ctrl.on_frame_changed(3)
    assert len(calls) == 1 and _note_percent(window.binary_panel) > 20


def test_a_classic_method_clears_the_note(open_auto):
    window, controller = open_auto(band_rows=50, glare=True)
    controller._state = update_state(controller._get_state(), detection_mode="intensity")
    controller.file_ctrl.on_frame_changed(3)
    panel = window.binary_panel
    assert panel._title_label.text() == MASK_TITLE and panel.title_note() == ""
    assert _fills(panel) == []
