"""Resume after an unclean exit: the startup restore offer and the clean-exit record.

The prompt is injected (or the real non-blocking box is clicked), so no
test ever waits for a user.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication, QMessageBox, QWidget

from bubbletrack import app as app_module
from bubbletrack.controller import session_controller as sc
from bubbletrack.model import config
from bubbletrack.model.session import load_session
from bubbletrack.model.state import AppState


class _SessionCtrl:
    def __init__(self) -> None:
        self.loaded: list[str] = []

    def load(self, path: str) -> bool:
        self.loaded.append(path)
        return True


def _pending_autosave() -> None:
    """An autosave written after the last clean exit (i.e. a crash)."""
    config.record_clean_exit(time.time() - 3600)
    path = sc.autosave_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}", encoding="utf-8")


def _wait(predicate, timeout_s: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        QApplication.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    return False


class TestOffer:
    def test_restore_loads_the_claimed_session(self, isolated_config):
        _pending_autosave()
        ctrl, asked = _SessionCtrl(), []

        def prompt(_parent, path, on_restore):
            asked.append(path)
            on_restore()

        offered = app_module.offer_session_restore(None, ctrl, prompt=prompt)
        assert offered == asked[0]
        assert ctrl.loaded == [str(offered)]
        assert offered.name == sc.PREVIOUS_SESSION_NAME

    def test_discard_loads_nothing_but_keeps_the_file(self, isolated_config):
        _pending_autosave()
        ctrl = _SessionCtrl()
        offered = app_module.offer_session_restore(None, ctrl, prompt=lambda *a: None)
        assert ctrl.loaded == []
        assert offered.exists()

    def test_no_prompt_after_a_clean_exit(self, isolated_config):
        path = sc.autosave_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
        config.record_clean_exit()
        asked = []
        assert app_module.offer_session_restore(
            None, _SessionCtrl(), prompt=lambda *a: asked.append(a)) is None
        assert asked == []


class TestRealPrompt:
    @pytest.mark.parametrize("button, restored", [("Restore", True), ("Discard", False)])
    def test_box_is_non_blocking_and_routes_the_choice(self, qapp, tmp_path, button, restored):
        parent = QWidget()
        calls = []
        try:
            app_module.ask_restore_session(parent, tmp_path / "previous_session.brt",
                                           lambda: calls.append(True))
            box = next(b for b in parent.findChildren(QMessageBox) if b.isVisible())
            assert "previous_session.brt" in box.text()
            next(b for b in box.buttons() if b.text() == button).click()
            if restored:
                assert _wait(lambda: calls == [True])
            else:
                _wait(lambda: False, timeout_s=0.1)
                assert calls == []
        finally:
            parent.close()


class TestCleanExit:
    def test_quit_saves_parameters_and_records_the_exit(self, isolated_config, monkeypatch):
        monkeypatch.setattr(config.time, "time", lambda: 4242.0)
        app_module.on_about_to_quit(AppState(img_thr=0.33))
        assert config.get_meta("last_clean_exit") == 4242.0
        assert config.load_config()["img_thr"] == 0.33


def test_crash_then_restore_end_to_end(gui, make_sequence, isolated_config):
    """Autosave during work, no clean exit, then the offer restores the results."""
    window, controller = gui
    folder = make_sequence("A", 4)
    controller.file_ctrl.on_folder_selected(str(folder))
    controller._get_state().radius[:] = [11.0, 12.0, 13.0, 14.0]
    session = controller.session_ctrl
    session.notify_results_changed()
    autosave = session.autosave_if_due()
    assert autosave and os.path.exists(autosave)

    # A clean exit stamped before the autosave = the app crashed afterwards.
    config.record_clean_exit(os.path.getmtime(autosave) - 3600)
    controller._get_state().radius[:] = -1.0  # "a new launch" with empty results

    offered = app_module.offer_session_restore(
        window, session, prompt=lambda _p, _path, on_restore: on_restore())
    assert offered is not None
    np.testing.assert_allclose(controller._get_state().radius, [11.0, 12.0, 13.0, 14.0])
    assert load_session(str(offered))["radius"][0] == 11.0
