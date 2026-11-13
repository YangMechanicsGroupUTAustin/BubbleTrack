"""Simple event bus for decoupled communication between controllers and UI."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Emitted (no arguments) after anything edits per-frame results in place:
# autosave, the review queue and the filmstrip listen for it.
RESULTS_CHANGED = "results_changed"

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
