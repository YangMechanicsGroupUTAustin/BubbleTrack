"""CircleController — "Drag a circle", the default tool of Review's Fix (F).

Press on the bubble centre, drag to the wall, release: the Auto engine
measures the wall near the drawn circle (``model.guided_fit``) and the
result is stored at once as GUIDED, or, when it does not match the circle,
the user may keep the circle itself as a manual fit.  A stored fix is the
user's decision: it is locked like an accepted frame, so Fit All keeps it
without asking.  Every stored fix can be undone with Ctrl+Z
(``model.result_history``, shared with re-tracking); after an undo the bus
gets ``RESULT_UNDONE`` with the edit, and Review shows its frame.

The tool is on while ``set_active(True)`` holds (``FixMode``), the Auto
engine is in use and no fit runs; its prompt shows in the attached view.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

from bubbletrack.controller import edge_support
from bubbletrack.controller.base import BaseController
from bubbletrack.controller.display_mixin import current_source, display_frame, refresh_chart
from bubbletrack.event_bus import CIRCLE_FIXED, EventBus
from bubbletrack.model import edge_pipeline, guided_fit
from bubbletrack.model.cache import ImageCache
from bubbletrack.model.guided_fit import UserCircle
from bubbletrack.model.provenance import FitSource
from bubbletrack.model.result_history import ResultHistory, record_edit, snapshot
from bubbletrack.ui.circle_tool import CIRCLE_UNAVAILABLE_TIP
from bubbletrack.ui.fix_strip import FixView, NoFixView
# The drawn circle (not stored) and what the engine found instead: the
# image legend's colours, so what is drawn always matches its key.
from bubbletrack.ui.overlay_legend import FOUND_COLOUR, USER_CIRCLE_COLOUR

logger = logging.getLogger(__name__)

_OK, _WARN, _ERR = "#10b981", "#f59e0b", "#ef4444"
RESULT_UNDONE = "result_undone"  # bus event (ResultEdit) after Ctrl+Z restored an edit
_MANUAL_OUTLINE_POINTS = 72  # a stored circle's outline, for both panels


@contextmanager
def _wait_cursor():
    QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
    try:
        yield
    finally:
        QApplication.restoreOverrideCursor()


def circle_outline(rc: float, cc: float, radius: float,
                   n: int = _MANUAL_OUTLINE_POINTS) -> np.ndarray:
    """(n, 2) [row, col] points on a circle, in order."""
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.column_stack([rc + radius * np.sin(angles), cc + radius * np.cos(angles)])


class CircleController(BaseController):
    """Turns a drawn circle into a stored result, and undoes it."""

    def __init__(self, bus: EventBus, get_state, set_state, window, get_max_radius,
                 cache: ImageCache | None = None, *, history: ResultHistory,
                 retrack, is_run_active) -> None:
        super().__init__(bus, get_state, set_state, window)
        self._get_max_radius = get_max_radius
        self._cache = cache
        self.history = history
        self._retrack = retrack  # RetrackController: cancelled/muted by undo
        self._is_run_active = is_run_active  # a Fit All / batch run writes results
        self._pending: tuple[int, UserCircle] | None = None  # refused circle on screen
        self._active = False  # the user chose the tool (Review's Fix)
        self._view: FixView = NoFixView()
        bus.subscribe("run_state_changed", lambda _running: self.sync_tool())
        bus.subscribe("frame_changed", self._on_frame_changed)
        bus.subscribe("source_loaded", self._on_source_loaded)

    def set_view(self, view: FixView) -> None:
        """Show the tool's availability and prompt in *view* (the fix strip)."""
        self._view = view
        self.sync_tool()

    # -- tool state ------------------------------------------------------- #

    def available(self) -> bool:
        """Drag a circle works with the Auto (reference frame) engine only."""
        return self.state.detection_mode == "edge"

    def explain_unavailable(self) -> None:
        self.w.header.set_status(CIRCLE_UNAVAILABLE_TIP.replace("\n", " "), _WARN)

    @property
    def active(self) -> bool:
        return self._active

    def set_active(self, on: bool) -> None:
        """Switch the tool on or off (off also drops a refused circle's prompt)."""
        self._active = bool(on)
        self.sync_tool()

    def activate(self) -> None:
        """Switch the tool on and say how to use it."""
        self.set_active(True)
        self.w.header.set_status("Drag from the bubble centre to its wall", _WARN)

    def sync_tool(self) -> None:
        """Image panel in circle mode while the tool is on and usable."""
        self._view.set_circle_available(self.available())
        on = self._active and self.available() and not self._is_run_active()
        panel = self.w.original_panel
        if on and panel.mode != "circle":
            panel.set_mode("circle")
            self.w.status_bar.update_mode("Manual")
        elif not on and panel.mode == "circle":
            panel.set_mode("normal")
        if not on:
            self._drop_prompt()

    # -- the gesture -------------------------------------------------------- #

    def on_circle_drawn(self, x: float, y: float, radius: float, dragged: bool) -> None:
        """Scene centre (x = col, y = row), radius and drag flag from the image panel."""
        if not self._active or not self._can_fit():
            return
        idx = self.state.image_no
        self._drop_prompt()
        circle = self._user_circle(x, y, radius, dragged, idx)
        try:
            fit, verdict = self._measure(idx, circle)
        except Exception as exc:
            logger.exception("Circle fit failed for frame %d", idx + 1)
            self.w.header.set_status(f"Circle fit failed: {exc}", _ERR)
            return
        if verdict.accepted:
            self._store_guided(idx, fit)
        else:
            self._refuse(idx, circle, fit, verdict.reason)

    def _can_fit(self) -> bool:
        if not self.state.images or self.state.radius is None:
            self.w.header.set_status("Load images before fitting", _ERR)
            return False
        if not self.available():
            self.explain_unavailable()
            return False
        if self._is_run_active():
            self.w.header.set_status("Wait for the running fit to finish", _WARN)
            return False
        return True

    def _user_circle(self, x: float, y: float, radius: float, dragged: bool,
                     idx: int) -> UserCircle:
        # Scene pixel i spans [i, i+1); its 1-based index coordinate is i + 1.
        rc, cc = y + 0.5, x + 0.5
        if dragged and radius > 0:
            return guided_fit.drawn_circle(rc, cc, radius)
        return guided_fit.clicked_circle(
            rc, cc, guided_fit.nearest_fitted_radius(self.state.radius, idx))

    def _measure(self, idx: int, circle: UserCircle):
        state = self.state
        with _wait_cursor():
            raw = edge_pipeline.read_raw(current_source(state, idx))
            job = edge_support.build_job(state, self._get_max_radius())
            return guided_fit.fit_guided(raw, state.gridx, state.gridy, job, circle)

    # -- outcomes ----------------------------------------------------------- #

    def _store_guided(self, idx: int, fit) -> None:
        before = snapshot(self.state, [idx])
        edge_support.store_fit(self.state, idx, fit, FitSource.GUIDED)
        flags = f", {', '.join(sorted(fit.flags))}" if fit.flags else ""
        self._committed(idx, before, f"Frame {idx + 1}: R = {fit.radius:.1f} px from your "
                                     f"circle (confidence {fit.confidence:.2f}{flags})")

    def use_circle_as_manual(self) -> None:
        """Prompt choice: store the refused circle itself (source MANUAL)."""
        pending, self._pending = self._pending, None
        self._view.hide_circle_prompt()
        state = self.state
        if pending is None or pending[0] != state.image_no or not pending[1].dragged:
            return
        idx, circle = pending
        before = snapshot(state, [idx])
        state.radius[idx] = circle.radius
        state.circle_fit_par[idx] = [circle.rc, circle.cc]
        state.circle_xy[idx] = circle_outline(circle.rc, circle.cc, circle.radius)
        if state.region_count is not None:
            state.region_count[idx] = 0
        if state.fit_confidence is not None:
            state.fit_confidence[idx] = np.nan  # the user's circle is not scored
        edge_support.store_meta(state, idx, {})
        if state.fit_source is not None:
            state.fit_source[idx] = FitSource.MANUAL
        self._committed(idx, before, f"Frame {idx + 1}: your circle stored as a manual fit "
                                     f"(R = {circle.radius:.1f} px)")

    def _committed(self, idx: int, before: dict, text: str) -> None:
        if self.state.fit_locked is not None:
            self.state.fit_locked[idx] = True  # the user's decision: Fit All keeps it
        self.history.push(record_edit(self.state, f"circle fix on frame {idx + 1}", before))
        self._redraw(idx)
        self.w.header.set_status(text + " — Ctrl+Z undoes", _OK)
        logger.info("Circle fix stored for frame %d: R = %.3f", idx + 1, self.state.radius[idx])
        self._results_changed()
        self.bus.emit(CIRCLE_FIXED, idx)

    def _refuse(self, idx: int, circle: UserCircle, fit, reason: str) -> None:
        self._pending = (idx, circle)
        self._redraw(idx)
        panel = self.w.original_panel
        if fit.radius > 0:
            panel.draw_fit_outline(fit.edge_xy, FOUND_COLOUR)
        if circle.dragged:  # drawn where the user drew it (scene = 1-based - 0.5)
            panel.draw_circle(circle.rc - 0.5, circle.cc - 0.5, circle.radius,
                              USER_CIRCLE_COLOUR, dashed=True)
        text = f"Frame {idx + 1}: {reason}. Nothing was stored."
        if not circle.dragged:
            text += " Drag from the centre to the wall to give its size."
        self._view.show_circle_prompt(text, offer_manual=circle.dragged)
        self.w.header.set_status(f"Frame {idx + 1}: {reason}", _WARN)

    def discard_prompt(self) -> bool:
        """Drop a refused circle and its drawing (no message); False if none."""
        if self._pending is None:
            return False
        self._drop_prompt()
        self._redraw(self.state.image_no)
        return True

    def _drop_prompt(self) -> None:
        self._pending = None
        self._view.hide_circle_prompt()

    def _redraw(self, idx: int) -> None:
        refresh_chart(self.state, self.w)
        if self.state.image_no == idx:
            display_frame(self.state, self.w, idx, self._set_state, self._cache)

    # -- undo ----------------------------------------------------------------- #

    def undo(self) -> None:
        """Ctrl+Z: undo the newest entry of the shared history (a circle fix,
        a manual point fit or clear, a re-track), restoring it exactly."""
        if self._is_run_active():
            self.w.header.set_status("Wait for the running fit to finish", _WARN)
            return
        outcome = self.history.undo(self.state)
        if not outcome.done:
            self.w.header.set_status(outcome.message, _WARN)
            return
        # A running or pending re-track was based on the edit just undone (its
        # result is applied on this thread later, so dropping it now is in time).
        self._retrack.cancel()
        self._drop_prompt()
        self._redraw(self.state.image_no)
        frames = outcome.edit.frames
        where = f"frame {frames[0] + 1}" if len(frames) == 1 else f"{len(frames)} frames"
        self.w.header.set_status(f"{outcome.message} ({where} restored)", _OK)
        with self._retrack.muted():  # an undo never triggers a new re-track
            self._results_changed()
        self.bus.emit(RESULT_UNDONE, outcome.edit)

    # -- bus events ------------------------------------------------------------ #

    def _on_frame_changed(self, _idx: int) -> None:
        self._drop_prompt()  # the refused circle belongs to the frame it was drawn on

    def _on_source_loaded(self, _n: int) -> None:
        self.history.clear()
        self._drop_prompt()
        self.sync_tool()
