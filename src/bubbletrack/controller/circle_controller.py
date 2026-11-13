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

A refused circle shows only what the decision is about, on both panels: the
user's circle (cyan dashes) and the wall Auto found near it (grey), never
the frame's stored result; the prompt names the rule that failed.  Closing
the prompt (Cancel, Keep my circle, a new drag, another frame) redraws the
frame as usual.
"""

from __future__ import annotations

import logging
import math
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

from bubbletrack.controller import edge_support
from bubbletrack.controller.base import BaseController
from bubbletrack.controller.display_mixin import current_source, display_frame, refresh_chart
from bubbletrack.event_bus import CIRCLE_FIXED, EventBus
from bubbletrack.model import edge_pipeline, guided_fit
from bubbletrack.model.cache import ImageCache
from bubbletrack.model.guided_fit import CENTRE_TOLERANCE, MIN_CONFIDENCE, UserCircle
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


@dataclass(frozen=True, eq=False)
class Refusal:
    """A drawn circle that was not stored, waiting for the user's decision."""

    frame: int
    circle: UserCircle
    found: np.ndarray | None
    """(N, 2) 1-based [row, col] outline of the wall Auto found near it, if any."""


def found_outline(fit) -> np.ndarray | None:
    """The outline Auto found for a refused circle: its wall, or its faint guess."""
    if fit.radius > 0 and len(fit.edge_xy):
        return np.asarray(fit.edge_xy, dtype=float)
    guess = getattr(fit, "suggestion", None)
    if guess is not None and len(guess.edge_xy):
        return np.asarray(guess.edge_xy, dtype=float)
    return None


def refusal_reason(circle: UserCircle, fit) -> str:
    """Why *fit* was not stored for *circle*, naming the rule (``guided_fit``)."""
    if fit.radius > 0:
        ref = circle.radius if circle.sized else fit.radius
        offset = math.hypot(fit.rc - circle.rc, fit.cc - circle.cc)
        if offset > CENTRE_TOLERANCE * ref:
            why = (f"its centre is {offset:.0f} px from yours, more than half your circle's "
                   f"radius ({CENTRE_TOLERANCE * ref:.0f} px)")
        else:
            why = (f"its radius is {fit.radius:.0f} px, yours is {ref:.0f} px: "
                   "they differ by more than half")
        return f"Auto found a wall (grey) that does not match your circle: {why}."
    if str(fit.rejected_reason).startswith("low confidence"):
        where = "here (grey)" if found_outline(fit) is not None else "near your circle"
        return (f"The wall Auto found {where} is too faint: confidence {fit.confidence:.2f}, "
                f"below the {MIN_CONFIDENCE:.2f} a drawn circle needs.")
    if fit.rejected_reason == "implausible radius":
        return "Auto found only a wall larger than any plausible bubble near your circle."
    return "Auto found no wall near your circle."


def refusal_text(frame: int, circle: UserCircle, fit, stored_radius: float) -> str:
    """The prompt: what happened and why, then what the user can do."""
    text = f"Frame {frame + 1}: {refusal_reason(circle, fit)} Nothing was stored."
    if not circle.dragged:
        return text + " Drag from the centre to the wall to give its size."
    text += " Keep your circle as this frame's result, or drag again."
    if np.isfinite(stored_radius) and stored_radius > 0:
        text += f" Keeping it replaces this frame's result (R = {stored_radius:.1f} px)."
    return text


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
        self._pending: Refusal | None = None  # refused circle on screen
        self._active = False  # the user chose the tool (Review's Fix)
        self._view: FixView = NoFixView()
        bus.subscribe("run_state_changed", lambda _running: self.sync_tool())
        bus.subscribe("frame_changed", self._on_frame_changed)
        bus.subscribe("source_loaded", self._on_source_loaded)
        # Every redraw of the refused frame shows the decision, not the result.
        window.frame_display_hooks.append(self._on_display)

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
            self.discard_prompt()  # the refused view goes with it

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
        """Prompt choice "Keep my circle": store the refused circle itself (source MANUAL)."""
        pending, self._pending = self._pending, None
        self._view.hide_circle_prompt()
        state = self.state
        if pending is None or pending.frame != state.image_no or not pending.circle.dragged:
            return
        idx, circle = pending.frame, pending.circle
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
        stored = float(self.state.radius[idx])
        self._pending = Refusal(idx, circle, found_outline(fit))
        self._redraw(idx)  # the display hook draws the refused view
        self._view.show_circle_prompt(refusal_text(idx, circle, fit, stored),
                                      offer_manual=circle.dragged)
        self.w.header.set_status(f"Frame {idx + 1}: {reason}", _WARN)

    def _on_display(self, state, idx: int) -> None:
        """Display hook: while a refused circle waits, only it and what Auto found."""
        refusal = self._pending
        if refusal is None or refusal.frame != idx:
            return
        circle = refusal.circle
        # Drawn where the user drew it (scene = 1-based - 0.5); the Auto view is the ROI.
        roi = np.array([state.gridx[0] - 1, state.gridy[0] - 1], dtype=float)
        for panel, origin in ((self.w.original_panel, np.zeros(2)), (self.w.binary_panel, roi)):
            panel.clear_overlays()
            if refusal.found is not None:
                panel.draw_fit_outline(refusal.found - origin, FOUND_COLOUR)
            if circle.dragged:
                panel.draw_circle(circle.rc - 0.5 - origin[0], circle.cc - 0.5 - origin[1],
                                  circle.radius, USER_CIRCLE_COLOUR, dashed=True)

    def discard_prompt(self) -> bool:
        """Drop a refused circle and its drawing (no message); False if none."""
        pending = self._pending is not None
        self._drop_prompt()
        if pending:
            self._redraw(self.state.image_no)
        return pending

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
