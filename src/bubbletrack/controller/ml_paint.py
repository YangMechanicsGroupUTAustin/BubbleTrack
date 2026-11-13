"""ML assist's "Paint examples by hand": brushes, strokes and their undo.

Folded away in the teaching panel and rarely needed.  While it is open (and
no run writes results) the image paints strokes: green on the bubble, red on
what is not, the eraser removes them; keys B / G / E pick the brush, the
mouse wheel sizes it, Ctrl+Z undoes the last stroke.  A frame with bubble and
background strokes counts as an example.  Strokes never start a training:
only Train and apply trains.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from PyQt6.QtGui import QKeySequence, QShortcut

from bubbletrack.model.ml_assist import default_brush
from bubbletrack.model.teach import BACKGROUND, BUBBLE, MLExamples, stroke_pixels
from bubbletrack.ui.ml_images import BACKGROUND_COLOUR, BUBBLE_COLOUR
from bubbletrack.ui.ml_teach_panel import TOOL_KEYS, TOOLS
from bubbletrack.ui.shortcuts import text_input_has_focus

TOOL_COLOURS = {"bubble": BUBBLE_COLOUR, "background": BACKGROUND_COLOUR, "eraser": "#e4e4e7"}
TOOL_LABEL = {"bubble": BUBBLE, "background": BACKGROUND}
BRUSH_LIMITS = (1, 80)
_OK, _WARN = "#10b981", "#f59e0b"


class HandPainting:
    """The painting tools of the teaching mode (see module doc)."""

    def __init__(self, ctrl, panel, *, wanted: Callable[[], bool],
                 changed: Callable[[], None]) -> None:
        self._c = ctrl  # MLAssistController: state, window, set_examples, status
        self.panel = panel
        self._wanted = wanted  # teaching is on, the section is open, no run
        self._changed = changed  # strokes changed (the panel's counts follow)
        self._tool = "bubble"
        self._radius: int | None = None  # brush radius (px); None: from the bubble size
        self._undo: list[MLExamples] = []
        w = ctrl.w
        self.keys: list[QShortcut] = []
        for tool in TOOLS:
            key = QShortcut(QKeySequence(TOOL_KEYS[tool]), w)
            key.activated.connect(lambda t=tool: self._on_key(t))
            key.setEnabled(False)
            self.keys.append(key)
        panel.tool_changed.connect(self.set_tool)
        panel.brush_size_changed.connect(self.set_radius)
        panel.undo_clicked.connect(self.undo_stroke)
        panel.clear_frame_clicked.connect(self.clear_frame)
        image = w.original_panel
        image.stroke_drawn.connect(self.on_stroke)
        image.brush_scrolled.connect(self.on_wheel)
        image.mode_changed.connect(self._on_image_mode)

    # -- the image's mouse and the keys ---------------------------------------- #

    def sync(self) -> None:
        """Keys and the image's paint mode only while painting is wanted."""
        paint = self._wanted()
        for key in self.keys:
            key.setEnabled(paint)
        image = self._c.w.original_panel
        if paint and image.mode == "normal":
            image.set_mode("paint")
        elif not paint and image.mode == "paint":
            image.set_mode("normal")

    def _on_image_mode(self, mode: str) -> None:
        if mode == "normal":
            self.sync()  # another tool gave the image back

    def prepare(self) -> None:
        """Teaching opens: the brush, sized to the recording's bubble."""
        if self._radius is None:
            self._radius = default_brush(self._c.state)
        self.panel.set_brush_size(self._radius)
        self.set_tool(self._tool)

    def reset(self) -> None:
        """Another recording: forget the undo and the brush size."""
        self._undo.clear()
        self._radius = None

    # -- brush ------------------------------------------------------------------ #

    def set_tool(self, tool: str) -> None:
        if tool not in TOOLS:
            return
        self._tool = tool
        self.panel.set_tool(tool)
        self._apply_brush()

    def tool(self) -> str:
        return self._tool

    def set_radius(self, radius: int) -> None:
        self._radius = int(np.clip(radius, *BRUSH_LIMITS))
        self.panel.set_brush_size(self._radius)
        self._apply_brush()

    @property
    def radius(self) -> int:
        return self._radius or default_brush(self._c.state)

    def on_wheel(self, step: int) -> None:
        if self._wanted():
            self.set_radius(self.radius + step * max(1, round(0.15 * self.radius)))

    def _apply_brush(self) -> None:
        self._c.w.original_panel.brush.set_brush(TOOL_COLOURS[self._tool], self.radius)

    def _on_key(self, tool: str) -> None:
        if self._wanted() and not text_input_has_focus():
            self.set_tool(tool)

    # -- strokes ------------------------------------------------------------------ #

    def _user_examples(self) -> MLExamples:
        ex = self._c.state.ml_examples
        return ex if isinstance(ex, MLExamples) else MLExamples()

    def on_stroke(self, points) -> None:
        """A finished stroke: scene (x, y) points on the shown frame."""
        state = self._c.state
        if not self._wanted() or state.cur_img is None:
            return
        idx = state.image_no
        pts = [(y + 0.5, x + 0.5) for x, y in points]  # scene -> 1-based (row, col)
        px = stroke_pixels(pts, self.radius, state.cur_img.shape[:2])
        (r0, r1), (c0, c1) = state.gridx, state.gridy
        px = px[(px[:, 0] >= r0) & (px[:, 0] <= r1) & (px[:, 1] >= c0) & (px[:, 1] <= c1)]
        if not len(px):
            self._c.status("Paint inside the ROI: ML assist only looks there", _WARN)
            return
        old = self._user_examples()
        new = old.erase(idx, px) if self._tool == "eraser" else old.paint(idx, TOOL_LABEL[self._tool], px)
        self._commit(old, new)

    def clear_frame(self) -> None:
        idx = self._c.state.image_no
        old = self._user_examples()
        if self._wanted() and idx in old.strokes:
            self._commit(old, old.clear(idx))

    def undo_stroke(self) -> bool:
        """Ctrl+Z while painting: the last stroke (False when there is none)."""
        if not self._wanted() or not self._undo:
            return False
        self._c.set_examples(self._undo.pop())
        self._after_change()
        self._c.status("Stroke undone", _OK)
        return True

    def _commit(self, old: MLExamples, new: MLExamples) -> None:
        if new == old:
            return
        self._undo.append(old)
        self._c.set_examples(new)
        self._after_change()

    def _after_change(self) -> None:
        self._c.redisplay()
        self._changed()
