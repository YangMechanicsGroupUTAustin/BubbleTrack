"""Debounced autosave of the full session to ~/.bubbletrack/recovery/autosave.brt.

The SessionController is driven with light fakes and an injectable clock;
the config directory is isolated, so nothing touches the real home folder.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest

from bubbletrack.controller import session_controller as sc
from bubbletrack.event_bus import EventBus
from bubbletrack.model import config
from bubbletrack.model.session import load_session
from bubbletrack.model.state import AppState


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def saves(monkeypatch):
    """Count real session writes."""
    calls: list[str] = []
    real = sc.save_session

    def _spy(path, state):
        calls.append(str(path))
        real(path, state)

    monkeypatch.setattr(sc, "save_session", _spy)
    return calls


def _make(qapp, *, with_results: bool = True, busy: list | None = None):
    state = AppState(images=("a.tif", "b.tif", "c.tif")).with_results_initialized(3)
    if with_results:
        state.radius[:] = [10.0, 12.5, -1.0]
    holder = {"state": state}
    busy = busy if busy is not None else [False]
    clock = _Clock()
    bus = EventBus()
    ctrl = sc.SessionController(
        bus, lambda: holder["state"], lambda s: holder.update(state=s),
        SimpleNamespace(header=SimpleNamespace(set_status=lambda *a: None)),
        file_ctrl=None,
        is_busy=lambda: busy[0],
        locate_folder=lambda _original: "",
        after_restore=lambda: None,
        clock=clock,
    )
    return ctrl, clock, bus


@pytest.fixture
def make(qapp, isolated_config):
    created = []

    def _factory(**kwargs):
        ctrl, clock, bus = _make(qapp, **kwargs)
        created.append(ctrl)
        return ctrl, clock, bus

    yield _factory
    for ctrl in created:
        ctrl.stop_autosave()


class TestAutosave:
    def test_writes_the_full_session_after_the_debounce(self, make, saves):
        ctrl, _, _ = make()
        ctrl.notify_results_changed()
        timer = ctrl._autosave_timer
        assert timer.isActive()
        assert timer.interval() == int(sc.AUTOSAVE_DEBOUNCE_S * 1000)

        path = ctrl.autosave_if_due()
        assert path == str(sc.autosave_path())
        assert sc.autosave_path().parent == config.CONFIG_DIR / "recovery"
        assert load_session(path)["radius"][1] == 12.5
        assert saves == [path + ".tmp"]  # written next to it, then renamed
        assert not os.path.exists(path + ".tmp")

    def test_at_most_once_per_interval(self, make, saves):
        ctrl, clock, _ = make()
        ctrl.notify_results_changed()
        ctrl.autosave_if_due()                 # first save at t = 1000
        clock.now += 10
        for _ in range(5):                     # a burst of edits
            ctrl.notify_results_changed()
        remaining = ctrl._autosave_timer.interval() / 1000
        assert remaining == pytest.approx(sc.AUTOSAVE_MIN_INTERVAL_S - 10, abs=0.01)

        assert ctrl.autosave_if_due() is None  # fired early: re-armed, no write
        assert len(saves) == 1
        assert ctrl._autosave_timer.isActive()
        clock.now += sc.AUTOSAVE_MIN_INTERVAL_S
        assert ctrl.autosave_if_due() is not None
        assert len(saves) == 2

    def test_nothing_to_save_without_changes_or_results(self, make, saves):
        ctrl, _, _ = make()
        assert ctrl.autosave_if_due() is None   # never notified
        empty, _, _ = make(with_results=False)
        empty.notify_results_changed()
        assert empty.autosave_if_due() is None
        assert saves == []

    def test_waits_for_a_run_to_finish(self, make, saves):
        busy = [True]
        ctrl, _, bus = make(busy=busy)
        ctrl.notify_results_changed()
        assert ctrl.autosave_if_due() is None
        assert saves == []

        busy[0] = False
        ctrl.stop_autosave()
        bus.emit("run_state_changed", True)
        assert not ctrl._autosave_timer.isActive()
        bus.emit("run_state_changed", False)  # the run ended: results changed
        assert ctrl._autosave_timer.isActive()
        assert ctrl.autosave_if_due() is not None

    def test_results_changed_bus_event_schedules_a_save(self, make):
        ctrl, _, bus = make()
        bus.emit(sc.RESULTS_CHANGED_EVENT)
        assert ctrl._autosave_timer.isActive()
        assert ctrl.autosave_if_due() is not None

    def test_write_failure_is_logged_not_raised(self, make, monkeypatch, caplog):
        ctrl, _, _ = make()

        def _fail(_path, _state):
            raise OSError("disk full")

        monkeypatch.setattr(sc, "save_session", _fail)
        ctrl.notify_results_changed()
        assert ctrl.autosave_if_due() is None
        assert "disk full" in caplog.text

    def test_config_dir_is_read_at_call_time(self, make, monkeypatch, tmp_path):
        ctrl, _, _ = make()
        moved = tmp_path / "elsewhere"
        monkeypatch.setattr(config, "CONFIG_DIR", moved)
        ctrl.notify_results_changed()
        assert ctrl.autosave_if_due() == str(moved / "recovery" / sc.AUTOSAVE_NAME)


class TestCloseSave:
    def test_close_writes_the_autosave_and_cancels_the_timer(self, make, saves):
        ctrl, _, _ = make()
        ctrl.notify_results_changed()
        path = ctrl.save_last_session()
        assert path == str(sc.autosave_path())
        assert not ctrl._autosave_timer.isActive()
        assert ctrl.autosave_if_due() is None  # nothing left pending
        assert len(saves) == 1

    def test_close_without_results_writes_nothing(self, make, saves):
        ctrl, _, _ = make(with_results=False)
        assert ctrl.save_last_session() is None
        assert saves == []


class TestPendingRecovery:
    def _autosave(self, mtime: float) -> None:
        path = sc.autosave_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
        os.utime(path, (mtime, mtime))

    def test_nothing_pending_without_an_autosave(self, isolated_config):
        assert sc.claim_pending_recovery() is None

    def test_autosave_after_the_last_clean_exit_is_claimed(self, isolated_config):
        config.record_clean_exit(1_000_000.0)
        self._autosave(1_000_100.0)
        claimed = sc.claim_pending_recovery()
        assert claimed == sc.autosave_path().with_name(sc.PREVIOUS_SESSION_NAME)
        assert claimed.read_text(encoding="utf-8") == "{}"
        # Moved aside, so this session's autosaves cannot overwrite it.
        assert not sc.autosave_path().exists()

    def test_autosave_from_a_clean_exit_is_not_offered(self, isolated_config):
        self._autosave(1_000_000.0)
        config.record_clean_exit(1_000_000.5)
        assert sc.claim_pending_recovery() is None
        assert sc.autosave_path().exists()

    def test_clock_slack_does_not_trigger_an_offer(self, isolated_config):
        config.record_clean_exit(1_000_000.0)
        self._autosave(1_000_000.0 + sc.RECOVERY_MTIME_SLACK_S / 2)
        assert sc.claim_pending_recovery() is None

    def test_no_clean_exit_on_record_counts_as_a_crash(self, isolated_config):
        self._autosave(1_000_000.0)
        assert sc.claim_pending_recovery() is not None


def test_array_helper_requires_results():
    assert not sc.has_results(AppState())
    state = AppState().with_results_initialized(2)
    assert not sc.has_results(state)
    state.radius[0] = np.float64(3.0)
    assert sc.has_results(state)
