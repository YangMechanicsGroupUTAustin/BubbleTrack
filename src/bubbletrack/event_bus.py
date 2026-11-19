"""Simple event bus for decoupled communication between controllers and UI."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Every event on the bus.  Use these names, never string literals: a typo in a
# literal subscribes to an event nobody emits, silently.

# Emitted (number of frames) after a folder, video or .dat recording was loaded.
SOURCE_LOADED = "source_loaded"
# Emitted (frame index) after the frame on screen changed.
FRAME_CHANGED = "frame_changed"
# Emitted (running: bool) when a Fit All / refit run starts or ends.
RUN_STATE_CHANGED = "run_state_changed"
# Emitted (done, total) while a run measures frames.
RUN_PROGRESS = "run_progress"
# Emitted after a run, tracking included, has finished.
RUN_FINISHED = "run_finished"
# Emitted with the frames tracking wants reviewed.
TRACKING_REVIEW = "tracking_review"
# Emitted with the collapse frames tracking found (0-based), after the results it
# stored; None after a run that did not track (its results replaced the solved ones).
TRACKING_COLLAPSES = "tracking_collapses"
# Emitted (no arguments) after anything edits per-frame results in place:
# autosave, the review queue, re-tracking and ML assist listen for it.
RESULTS_CHANGED = "results_changed"
# Emitted (no arguments) after a session file restored the recording and its parameters.
SESSION_RESTORED = "session_restored"
# Emitted (ResultEdit) after Ctrl+Z restored an edit.
RESULT_UNDONE = "result_undone"

# Emitted with (name, value) after a Settings switch changed (see config.UI_SETTINGS).
UI_SETTING_CHANGED = "ui_setting_changed"
# Emitted (frame index) after the "Drag a circle" tool stored a result.
CIRCLE_FIXED = "circle_fixed"


@dataclass
class _Subscription:
    event: str
    callback: Callable
    id: int


class EventBus:
    """Publish-subscribe event bus.

    Usage::

        bus = EventBus()
        token = bus.subscribe("frame_changed", handler)
        bus.emit("frame_changed", new_idx)
        bus.unsubscribe(token)
    """

    def __init__(self) -> None:
        self._subs: dict[str, list[_Subscription]] = {}
        self._next_id = 0

    def subscribe(self, event: str, callback: Callable) -> int:
        """Subscribe to an event. Returns a token for unsubscribing."""
        token = self._next_id
        self._next_id += 1
        sub = _Subscription(event=event, callback=callback, id=token)
        self._subs.setdefault(event, []).append(sub)
        return token

    def unsubscribe(self, token: int) -> None:
        """Remove a subscription by token."""
        for event, subs in self._subs.items():
            self._subs[event] = [s for s in subs if s.id != token]

    def emit(self, event: str, *args: Any, **kwargs: Any) -> None:
        """Emit an event to all subscribers.

        If a handler raises, the exception is logged and remaining
        handlers still execute.
        """
        for sub in self._subs.get(event, []):
            try:
                sub.callback(*args, **kwargs)
            except Exception:
                logger.exception("Error in event handler for %r", event)
