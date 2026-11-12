"""Base class for domain sub-controllers."""

from __future__ import annotations

from bubbletrack.event_bus import RESULTS_CHANGED, EventBus
from bubbletrack.model.state import AppState, update_state


class BaseController:
    """Base class providing shared state access for all sub-controllers.

    Each sub-controller receives callable accessors for the shared
    ``AppState`` instance that lives on the main ``AppController``.
    """

    def __init__(self, bus: EventBus, get_state, set_state, window) -> None:
        self.bus = bus
        self._get_state = get_state
        self._set_state = set_state
        self.w = window

    @property
    def state(self) -> AppState:
        return self._get_state()

    @state.setter
    def state(self, new_state: AppState) -> None:
        self._set_state(new_state)

    def _update(self, **kwargs) -> None:
        """Convenience: update state with new field values."""
        self.state = update_state(self.state, **kwargs)

    def _results_changed(self) -> None:
        """Announce an in-place edit of the per-frame results."""
        self.bus.emit(RESULTS_CHANGED)
