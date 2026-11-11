"""Tests for the last-resort uncaught-exception handler."""

from __future__ import annotations

import sys
import threading

import numpy as np

from bubbletrack.crash_handler import CrashHandler
from bubbletrack.model.session import load_session
from bubbletrack.model.state import AppState


def _state_with_results() -> AppState:
    state = AppState(folder_path="C:/data/shot").with_results_initialized(3)
    state.radius[:] = [10.0, 20.0, -1.0]
    return state


def _raise_and_capture(exc: BaseException):
    try:
        raise exc
    except BaseException:
        return sys.exc_info()


class _Clock:
    def __init__(self, t: float = 1000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t


def _make(tmp_path, state=None, clock=None, notify=None, fallback=None):
    calls: list[tuple[str, str | None]] = []
    handler = CrashHandler(
        state_provider=(lambda: state),
        notify=notify or (lambda summary, path: calls.append((summary, path))),
        recovery_dir=tmp_path / "recovery",
        clock=clock or _Clock(),
        fallback=fallback or (lambda *a: None),
    )
    return handler, calls


class TestHandling:
    def test_notifies_with_summary_and_saves_recovery(self, tmp_path):
        handler, calls = _make(tmp_path, state=_state_with_results())
        handler(*_raise_and_capture(ValueError("boom")))
        assert len(calls) == 1
        summary, path = calls[0]
        assert "ValueError" in summary and "boom" in summary
        assert path is not None
        restored = load_session(path)
        np.testing.assert_array_equal(restored["radius"], [10.0, 20.0, -1.0])
        assert restored["folder_path"] == "C:/data/shot"

    def test_no_recovery_file_without_results(self, tmp_path):
        handler, calls = _make(tmp_path, state=AppState())
        handler(*_raise_and_capture(RuntimeError("x")))
        assert calls[0][1] is None
        assert not (tmp_path / "recovery").exists()

    def test_recovery_saves_are_rate_limited(self, tmp_path):
        clock = _Clock()
        handler, calls = _make(tmp_path, state=_state_with_results(), clock=clock)
        handler(*_raise_and_capture(RuntimeError("first")))
        clock.t += 5.0
        handler(*_raise_and_capture(RuntimeError("second")))
        clock.t += 60.0
        handler(*_raise_and_capture(RuntimeError("third")))
        saved = sorted((tmp_path / "recovery").glob("*.brt"))
        assert len(saved) == 2
        assert calls[1][1] == calls[0][1]  # second error points at the recent file

    def test_broken_state_provider_still_notifies(self, tmp_path):
        calls = []

        def broken():
            raise RuntimeError("state unavailable")

        handler = CrashHandler(
            state_provider=broken,
            notify=lambda s, p: calls.append((s, p)),
            recovery_dir=tmp_path / "recovery",
        )
        handler(*_raise_and_capture(KeyError("k")))
        assert len(calls) == 1 and calls[0][1] is None

    def test_broken_notify_does_not_raise(self, tmp_path):
        def notify(summary, path):
            raise RuntimeError("dialog failed")

        handler, _ = _make(tmp_path, state=_state_with_results(), notify=notify)
        handler(*_raise_and_capture(ValueError("boom")))  # must not raise

    def test_reentrant_errors_are_not_re_notified(self, tmp_path):
        calls = []
        handler = None

        def notify(summary, path):
            calls.append(summary)
            handler(*_raise_and_capture(RuntimeError("inside notify")))

        handler = CrashHandler(
            state_provider=lambda: None,
            notify=notify,
            recovery_dir=tmp_path / "recovery",
        )
        handler(*_raise_and_capture(ValueError("outer")))
        assert len(calls) == 1

    def test_exception_with_broken_str_still_notifies(self, tmp_path):
        class Nasty(Exception):
            def __str__(self):
                raise RuntimeError("cannot format")

        handler, calls = _make(tmp_path, state=_state_with_results())
        handler(*_raise_and_capture(Nasty()))  # must not raise
        assert len(calls) == 1
        assert "Nasty" in calls[0][0]

    def test_keyboard_interrupt_goes_to_fallback(self, tmp_path):
        seen = []
        handler, calls = _make(tmp_path, fallback=lambda *a: seen.append(a[0]))
        handler(*_raise_and_capture(KeyboardInterrupt()))
        assert seen == [KeyboardInterrupt]
        assert calls == []


class TestInstall:
    def test_install_and_uninstall_restore_hooks(self, tmp_path):
        before_sys, before_thread = sys.excepthook, threading.excepthook
        handler, _ = _make(tmp_path)
        handler.install()
        try:
            assert sys.excepthook is not before_sys
            assert threading.excepthook is not before_thread
        finally:
            handler.uninstall()
        assert sys.excepthook is before_sys
        assert threading.excepthook is before_thread

    def test_thread_exceptions_are_routed(self, tmp_path):
        handler, calls = _make(tmp_path)
        handler.install()
        try:
            t = threading.Thread(target=lambda: 1 / 0)
            t.start()
            t.join()
        finally:
            handler.uninstall()
        assert len(calls) == 1
        assert "ZeroDivisionError" in calls[0][0]
