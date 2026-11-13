"""The Auto box's reference frame and bubble position (the seed click).

A reference must be bubble-free: "Use this frame" runs the same test the
automatic choice applies to its candidates (``edge_pipeline.frame_has_bubble``)
and asks before accepting a frame that seems to hold a bubble.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from PyQt6.QtWidgets import QMessageBox

from bubbletrack.controller import edge_support
from bubbletrack.controller.base import BaseController
from bubbletrack.controller.display_mixin import current_source
from bubbletrack.model import edge_pipeline
from bubbletrack.model.state import AppState
from bubbletrack.ui.edge_options import AUTO_POSITION_TEXT

logger = logging.getLogger(__name__)

_OK, _WARN, _DIM = "#10b981", "#f59e0b", "#a1a1aa"
NO_REFERENCE_NEEDED = "none needed (bubble from frame 1)"
NO_REFERENCE_FOUND = "not found — results may be less accurate"
REFERENCE_QUESTION = ("Frame {n} seems to contain a bubble; the reference must be "
                      "bubble-free. Use it anyway?")

# Whether a source's first frame holds a bubble (read once per source).
_first_frame_bubble: dict[tuple, bool] = {}


def holds_bubble(state: AppState, idx: int) -> bool:
    """True when frame *idx* seems to contain a bubble (False if it cannot be read)."""
    try:
        return edge_pipeline.frame_has_bubble(edge_pipeline.read_raw(current_source(state, idx)))
    except Exception as exc:  # noqa: BLE001 - an unreadable frame is reported elsewhere
        logger.warning("Bubble check of frame %d failed: %s", idx + 1, exc)
        return False


def _starts_with_bubble(state: AppState) -> bool:
    key = (state.folder_path, state.video_path, len(state.images))
    if key not in _first_frame_bubble:
        _first_frame_bubble[key] = holds_bubble(state, 0)
    return _first_frame_bubble[key]


def reference_label(state: AppState) -> str:
    """'Frame 2 (auto)', 'Frame 11' (chosen: its × goes back to Auto), or why there is none."""
    if not state.images:
        return "auto"
    idx = edge_support.reference_index(state)
    if idx is None:
        return NO_REFERENCE_NEEDED if _starts_with_bubble(state) else NO_REFERENCE_FOUND
    return f"Frame {idx + 1}" + ("" if state.reference_frame is not None else " (auto)")


def position_label(state: AppState) -> str:
    """'auto · click to set', or the clicked point as x (column), y (row)."""
    if state.edge_seed is None:
        return AUTO_POSITION_TEXT
    row, col = state.edge_seed
    return f"x {col:.0f}, y {row:.0f}"


def bubble_reference_dialog(parent, frame_no: int) -> tuple[QMessageBox, object]:
    """The question box and its 'Use it anyway' button (Cancel is the default)."""
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Warning)
    box.setWindowTitle("Reference frame")
    box.setText(REFERENCE_QUESTION.format(n=frame_no))
    box.setInformativeText("Auto compares every frame with the reference: a bubble "
                           "in it hides the bubble in the frames being measured.")
    use = box.addButton("Use it anyway", QMessageBox.ButtonRole.AcceptRole)
    box.setDefaultButton(box.addButton(QMessageBox.StandardButton.Cancel))
    return box, use


def ask_bubble_reference(parent, frame_no: int) -> bool:
    """'Frame N seems to contain a bubble ... Use it anyway?'; True to use it."""
    box, use = bubble_reference_dialog(parent, frame_no)
    box.exec()
    return box.clickedButton() is use


class EdgeSetup(BaseController):
    """Reference frame and bubble position of the Auto box."""

    def __init__(self, bus, get_state, set_state, window) -> None:
        super().__init__(bus, get_state, set_state, window)
        # (parent, 1-based frame) -> use it anyway; None asks with a message box.
        self.ask_bubble_reference: Callable[[object, int], bool] | None = None
        self.picking_seed = False

    def refresh_labels(self) -> None:
        if self.state.detection_mode != "edge":
            return
        box = self.w.left_panel.measure_panel.edge_box
        try:
            box.set_reference_text(reference_label(self.state),
                                   user_set=self.state.reference_frame is not None)
        except Exception as exc:  # unreadable frames while auto-selecting
            box.set_reference_text(f"Error: {exc}", user_set=self.state.reference_frame is not None)
        box.set_seed_text(position_label(self.state), user_set=self.state.edge_seed is not None)

    # -- reference --------------------------------------------------------- #

    def use_current(self) -> None:
        """'Use this frame': the frame on screen becomes the reference (asks first
        when it seems to hold a bubble)."""
        if not self.state.images:
            return
        idx = self.state.image_no
        bubble = holds_bubble(self.state, idx)
        ask = self.ask_bubble_reference or ask_bubble_reference
        if bubble and not ask(self.w, idx + 1):
            self.w.header.set_status(f"Reference unchanged: frame {idx + 1} was not used", _DIM)
            return
        self._update(reference_frame=idx)
        self.refresh_labels()
        if bubble:
            self.w.header.set_status(f"Reference: frame {idx + 1} (it seems to hold a bubble)", _WARN)
        else:
            self.w.header.set_status(f"Reference: frame {idx + 1}", _OK)

    def use_auto(self) -> None:
        self._update(reference_frame=None)
        self.refresh_labels()

    # -- bubble position ------------------------------------------------------ #

    def pick(self) -> None:
        if not self.state.images:
            return
        self.picking_seed = True
        self.w.original_panel.set_mode("point")
        self.w.header.set_status("Click once on the bubble", _WARN)

    def on_point_clicked(self, x: float, y: float) -> None:
        """Scene click (x = col, y = row, 0-based scene); used while picking."""
        if not self.picking_seed:
            return
        self.picking_seed = False
        self.w.original_panel.set_mode("normal")
        # Scene pixel i spans [i, i+1); its 1-based index coordinate is i + 1.
        self._update(edge_seed=(y + 0.5, x + 0.5))
        self.refresh_labels()
        self.w.header.set_status("Bubble position set", _OK)

    def clear(self) -> None:
        self._update(edge_seed=None)
        self.refresh_labels()
