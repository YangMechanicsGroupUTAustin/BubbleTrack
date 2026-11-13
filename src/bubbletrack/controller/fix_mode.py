"""Fix (F) in Review: fix the frame on screen, then let Review move on.

One fix at a time, on one frame, shown in the Review dock's fix strip:

* ``start(frame)``: "Drag a circle" (Auto measures the wall,
  ``CircleController``); outside the Auto engine, "Click edge points"
  (``ManualController``).  The strip switches between the two.
* A stored result (the measured wall, the user's own circle, or a circle
  through the clicked points) ends the fix and calls ``on_fixed(frame)``.
* Cancel (Esc), another frame, another source or a starting Fit All end
  it; the frame stays as it was.

Starting a fix cancels a pending or running re-track (it would redraw the
frame under the user's points); storing the fix restarts its countdown.

A fix owns the image's mouse while it runs.  Whatever else wants it (ML
assist's teaching mode) registers a start hook (``add_start_hook``): it is
called before the fix takes the image, gives the image up and returns a
note that the status line appends ("ML assist is closed ...").
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from bubbletrack.event_bus import CIRCLE_FIXED
from bubbletrack.ui.circle_tool import TOOL_CIRCLE, TOOL_POINTS
from bubbletrack.ui.fix_strip import FixStrip
from bubbletrack.ui.status_bar import MEASURE_MODE

logger = logging.getLogger(__name__)

_OK, _WARN = "#10b981", "#f59e0b"


class FixMode:
    """Runs the fix tools for one frame (see module doc)."""

    def __init__(self, app, strip: FixStrip, on_fixed: Callable[[int], None]) -> None:
        self._app = app
        self.strip = strip
        self._on_fixed = on_fixed
        self._start_hooks: list[Callable[[], str]] = []
        app.manual_ctrl.set_view(strip)
        app.circle_ctrl.set_view(strip)
        strip.points_requested.connect(lambda: self.use_tool(TOOL_POINTS))
        strip.circle_requested.connect(lambda: self.use_tool(TOOL_CIRCLE))
        strip.done_clicked.connect(self.finish_points)
        strip.cancel_clicked.connect(self.cancel)
        strip.use_circle_clicked.connect(app.circle_ctrl.use_circle_as_manual)
        bus = app.bus
        bus.subscribe(CIRCLE_FIXED, self._on_circle_fixed)
        bus.subscribe("frame_changed", self._on_frame_changed)
        bus.subscribe("source_loaded", lambda _n: self.end())
        bus.subscribe("run_state_changed", self._on_run_state)

    # -- state ------------------------------------------------------------- #

    @property
    def active(self) -> bool:
        return self.strip.frame() is not None

    @property
    def frame(self) -> int | None:
        """The frame being fixed (0-based), None while not fixing."""
        return self.strip.frame()

    @property
    def tool(self) -> str | None:
        return self.strip.tool()

    # -- starting, switching, ending ---------------------------------------- #

    def add_start_hook(self, hook: Callable[[], str]) -> None:
        """*hook()* runs before a fix takes the image: it gives the image up and
        returns a note for the status line ('' for none)."""
        self._start_hooks.append(hook)

    def start(self, frame: int) -> bool:
        """Fix *frame*, which must be on screen; False if it cannot be fixed now."""
        state = self._app._get_state()
        if not state.images or state.radius is None:
            self._status("Load images before fixing a frame", _WARN)
            return False
        if self._app.auto_ctrl.is_busy():
            self._status("Wait for the running fit to finish", _WARN)
            return False
        if state.image_no != frame:
            return False  # the caller shows the frame first
        if self.active and self.frame == frame:
            return True
        self.end()
        notes = [note for note in (hook() for hook in self._start_hooks) if note]
        # A re-track landing now would redraw the frame under the user's
        # points; storing the fix restarts its countdown.
        self._app.retrack.cancel()
        tool = TOOL_CIRCLE if self._app.circle_ctrl.available() else TOOL_POINTS
        self.strip.show_tool(frame, tool)
        self._apply(tool)
        if notes:
            header = self._app.w.header
            self._status(" — ".join([header.status_text(), *notes]), _WARN)
        logger.info("Fix started on frame %d (%s)", frame + 1, tool)
        return True

    def use_tool(self, tool: str) -> None:
        """Switch the running fix to *tool* (its points or circle prompt go)."""
        if not self.active or tool == self.tool:
            return
        if tool == TOOL_CIRCLE and not self._app.circle_ctrl.available():
            self._app.circle_ctrl.explain_unavailable()
            return
        self.strip.show_tool(self.frame, tool)
        self._apply(tool)

    def _apply(self, tool: str) -> None:
        circle, manual = self._app.circle_ctrl, self._app.manual_ctrl
        if tool == TOOL_CIRCLE:
            manual.cancel_selection()
            circle.activate()
        else:
            circle.discard_prompt()
            circle.set_active(False)
            manual.on_manual_select()

    def finish_points(self) -> None:
        """Done (Enter): fit the clicked points; a stored fit ends the fix."""
        if self.tool != TOOL_POINTS:
            return
        frame, manual = self.frame, self._app.manual_ctrl
        if manual.on_manual_done():
            self._finish(frame)
        elif not manual.selecting:  # the fit was refused: click the points again
            manual.on_manual_select(announce=False)

    def cancel(self) -> None:
        """Cancel (Esc): end the fix and leave the frame as it was."""
        frame = self.frame
        if frame is None:
            return
        self.end()
        self._status(f"Fix cancelled; frame {frame + 1} is unchanged", _OK)

    def end(self) -> None:
        """End the fix without a message (nothing is stored)."""
        if not self.active:
            return
        self.strip.end()
        circle = self._app.circle_ctrl
        circle.discard_prompt()
        circle.set_active(False)
        self._app.manual_ctrl.cancel_selection()
        self._app.w.status_bar.update_mode(MEASURE_MODE)  # the fix tools said "Manual"

    def _finish(self, frame: int) -> None:
        self.end()
        self._on_fixed(frame)

    # -- events ------------------------------------------------------------- #

    def _on_circle_fixed(self, frame: int) -> None:
        if self.frame == frame:
            self._finish(frame)

    def _on_frame_changed(self, idx: int) -> None:
        if self.active and idx != self.frame:
            self.end()  # a fix belongs to the frame it was started on

    def _on_run_state(self, running: bool) -> None:
        if running and self._app.auto_ctrl.is_busy():
            self.end()  # Fit All writes every result: no fix meanwhile

    def _status(self, text: str, colour: str) -> None:
        self._app.w.header.set_status(text, colour)
