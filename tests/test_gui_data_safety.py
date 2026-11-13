"""Regression tests: GUI paths that used to lose or corrupt results."""

from __future__ import annotations

import json
import time

import numpy as np
import pytest

from bubbletrack.model.provenance import FitSource
from bubbletrack.model.session import load_session, save_session
from bubbletrack.model.state import update_state


def _load(controller, folder) -> None:
    controller.file_ctrl.on_folder_selected(str(folder))


def _status(window) -> str:
    return window.header._status_label.text()


def _wait(qapp, predicate, timeout_s: float = 90.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        qapp.processEvents()
        if predicate():
            return
        time.sleep(0.01)
    raise TimeoutError("condition not reached")


def _run_fit_all(qapp, window, controller) -> None:
    controller.auto_ctrl.on_auto_fit()
    _wait(qapp, lambda: not controller.auto_ctrl.is_busy())


def _use_owner_params(controller) -> None:
    """The owner's everyday classic parameters; app defaults (thr 0.50) hit the
    known noise-blob regime on clean synthetic frames."""
    controller._state = update_state(controller._get_state(), detection_mode="intensity",
                                     img_thr=0.58, removing_factor=85, gaussian_sigma=3.0)


def _click_circle(controller, center=(125.0, 200.0), r=30.0, n=8):
    for a in np.linspace(0, 2 * np.pi, n, endpoint=False):
        controller.manual_ctrl.on_point_clicked(center[1] + r * np.cos(a),
                                                center[0] + r * np.sin(a))


# -- sessions ---------------------------------------------------------------------

class TestSessionRestore:
    def _fitted_state(self, controller):
        state = controller._get_state()
        n = len(state.images)
        state.radius[:] = np.linspace(20, 40, n)
        state.circle_fit_par[:] = [[120.0, 190.0]] * n
        for i in range(n):
            state.circle_xy[i] = np.array([[1.0, 2.0], [3.0, 4.5], [6.0, 7.0]]) + i
        state.fit_source[:] = FitSource.AUTO
        state.fit_source[2] = FitSource.MANUAL
        controller._state = update_state(state, gridx=(10, 200), gridy=(20, 350),
                                         image_no=3, img_thr=0.58)

    def test_round_trip_restores_results_roi_frame_and_sources(
            self, gui, make_sequence, tmp_path):
        window, controller = gui
        _load(controller, make_sequence("A", 8))
        self._fitted_state(controller)
        path = tmp_path / "s.brt"
        controller.save_session_file(str(path))

        _load(controller, make_sequence("B", 5))  # something else in between
        assert controller.load_session_file(str(path))

        state = controller._get_state()
        np.testing.assert_allclose(state.radius, np.linspace(20, 40, 8))
        np.testing.assert_allclose(state.circle_fit_par[0], [120.0, 190.0])
        np.testing.assert_allclose(state.circle_xy[5], np.array([[6, 7], [8, 9.5], [11, 12]]))
        assert state.fit_source[2] == FitSource.MANUAL
        assert (state.gridx, state.gridy) == ((10, 200), (20, 350))
        assert state.image_no == 3
        assert window.frame_scrubber.value() == 3
        assert window.left_panel.measure_panel.get_threshold() == pytest.approx(0.58)
        assert len(window.radius_chart._frames) == 8

    def test_frame_count_mismatch_keeps_params_not_results(
            self, gui, make_sequence, tmp_path):
        window, controller = gui
        folder = make_sequence("A", 8)
        _load(controller, folder)
        self._fitted_state(controller)
        path = tmp_path / "s.brt"
        controller.save_session_file(str(path))
        for extra in sorted(folder.glob("*.tiff"))[6:]:
            extra.unlink()

        assert controller.load_session_file(str(path))
        state = controller._get_state()
        assert len(state.images) == 6
        assert (state.radius == -1).all()
        assert state.img_thr == pytest.approx(0.58)
        assert "frame count" in _status(window).lower()

    def test_missing_folder_is_reported_and_current_state_kept(
            self, gui, make_sequence, tmp_path, monkeypatch):
        window, controller = gui
        _load(controller, make_sequence("A", 4))
        session = {"version": "3.0", "folder_path": str(tmp_path / "gone"),
                   "img_thr": 0.7, "radius": [1.0, 2.0]}
        path = tmp_path / "s.brt"
        path.write_text(json.dumps(session), encoding="utf-8")
        monkeypatch.setattr(controller, "_locate_moved_folder", lambda original: "")

        assert not controller.load_session_file(str(path))
        assert "not found" in _status(window).lower()
        assert controller._get_state().folder_path.endswith("A")

    def test_unreadable_session_file_is_reported(self, gui, tmp_path):
        window, controller = gui
        bad = tmp_path / "bad.brt"
        bad.write_bytes(b"\xff\xfe\x00 not json")
        assert not controller.load_session_file(str(bad))
        assert not controller.load_session_file(str(tmp_path))  # a directory
        assert "failed" in _status(window).lower()


class TestSessionFile:
    def test_file_is_strict_json_without_nan(self, tmp_path):
        from bubbletrack.model.state import AppState

        state = AppState(folder_path="x").with_results_initialized(3)
        state.radius[1] = 12.5
        path = tmp_path / "s.brt"
        save_session(str(path), state)

        def reject(token):
            raise ValueError(f"non-standard JSON token {token}")

        json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)
        data = load_session(str(path))
        assert np.isnan(data["circle_fit_par"][0]).all()
        assert data["radius"][1] == 12.5

    @pytest.mark.parametrize("version, ok", [("3.0", True), ("10.0", True),
                                             ("3.1", True), ("2.9", False), (3.0, True)])
    def test_version_compared_numerically(self, tmp_path, version, ok):
        path = tmp_path / "s.brt"
        path.write_text(json.dumps({"version": version}), encoding="utf-8")
        if ok:
            load_session(str(path))
        else:
            with pytest.raises(ValueError):
                load_session(str(path))


# -- Fit All run safety --------------------------------------------------------------

class TestRunSafety:
    def test_controls_locked_while_fitting(self, qapp, gui, make_sequence):
        window, controller = gui
        _load(controller, make_sequence("A", 30))
        controller.auto_ctrl.on_auto_fit()
        lp = window.left_panel
        assert not lp.image_source.isEnabled()
        assert not lp.measure_panel.isEnabled()
        assert not lp.export_button.isEnabled()
        assert not window.load_session_action.isEnabled()
        _wait(qapp, lambda: lp.fit_controls.fit_button.isEnabled())
        assert lp.image_source.isEnabled()
        assert lp.measure_panel.isEnabled()
        assert window.load_session_action.isEnabled()

    def test_folder_load_refused_while_fitting(self, qapp, gui, make_sequence):
        window, controller = gui
        folder_a = make_sequence("A", 30)
        folder_b = make_sequence("B", 5)
        _load(controller, folder_a)
        controller.auto_ctrl.on_auto_fit()
        _load(controller, folder_b)
        assert controller._get_state().folder_path == str(folder_a)
        _wait(qapp, lambda: not controller.auto_ctrl.is_busy())

    def test_results_from_a_stale_worker_are_ignored(self, gui, make_sequence):
        _, controller = gui
        _load(controller, make_sequence("A", 5))
        stale = object()
        controller.auto_ctrl._on_worker_frame_done(
            stale, 1, 42.0, np.zeros((3, 2)), None, 10.0, 10.0, 0)
        assert controller._get_state().radius[1] == -1.0

    def test_final_autosave_and_sources_after_fit_all(self, qapp, gui, make_sequence):
        window, controller = gui
        folder = make_sequence("A", 12)
        _load(controller, folder)
        _use_owner_params(controller)
        _run_fit_all(qapp, window, controller)
        assert (folder / ".bubbletrack_autosave.mat").exists()
        state = controller._get_state()
        assert (state.radius > 0).all()
        assert (state.fit_source == FitSource.AUTO).all()

    def test_failed_frames_are_reported(self, qapp, gui, make_sequence):
        window, controller = gui
        folder = make_sequence("A", 10)
        (folder / "frame_005.tiff").write_bytes(b"corrupt")
        _load(controller, folder)
        _run_fit_all(qapp, window, controller)
        assert "1 frame" in _status(window).lower()

    def test_stop_reports_stopped_not_done(self, gui, make_sequence):
        window, controller = gui
        _load(controller, make_sequence("A", 5))
        controller.auto_ctrl._stop_requested = True
        controller.auto_ctrl._finish_run(stopped=True)
        assert "stopped" in _status(window).lower()


# -- provenance: manual fixes are protected ---------------------------------------------

class TestManualProtection:
    def _manual_fix_frame3(self, controller):
        _use_owner_params(controller)
        controller.file_ctrl.on_frame_changed(3)
        controller.manual_ctrl.on_manual_select()
        _click_circle(controller, r=33.0)
        controller.manual_ctrl.on_manual_done()
        state = controller._get_state()
        assert state.fit_source[3] == FitSource.MANUAL
        return float(state.radius[3])

    def test_fit_all_keeps_manual_fixes_by_default(self, qapp, gui, make_sequence,
                                                   monkeypatch):
        window, controller = gui
        _load(controller, make_sequence("A", 8))
        manual_r = self._manual_fix_frame3(controller)
        monkeypatch.setattr(controller.auto_ctrl, "_ask_manual_policy", lambda n: "keep")
        _run_fit_all(qapp, window, controller)
        state = controller._get_state()
        assert state.radius[3] == manual_r
        assert state.fit_source[3] == FitSource.MANUAL
        assert state.fit_source[4] == FitSource.AUTO

    def test_fit_all_can_overwrite_manual_fixes(self, qapp, gui, make_sequence,
                                                monkeypatch):
        window, controller = gui
        _load(controller, make_sequence("A", 8))
        manual_r = self._manual_fix_frame3(controller)
        monkeypatch.setattr(controller.auto_ctrl, "_ask_manual_policy",
                            lambda n: "overwrite")
        _run_fit_all(qapp, window, controller)
        state = controller._get_state()
        assert state.radius[3] != manual_r
        assert state.fit_source[3] == FitSource.AUTO

    def test_cancel_does_not_start(self, gui, make_sequence, monkeypatch):
        _, controller = gui
        _load(controller, make_sequence("A", 8))
        self._manual_fix_frame3(controller)
        monkeypatch.setattr(controller.auto_ctrl, "_ask_manual_policy", lambda n: "cancel")
        controller.auto_ctrl.on_auto_fit()
        assert controller.auto_ctrl._worker is None



# -- closing the window ------------------------------------------------------------------

class TestClose:
    def test_close_while_fitting_can_be_cancelled(self, qapp, gui, make_sequence,
                                                  monkeypatch):
        window, controller = gui
        _load(controller, make_sequence("A", 30))
        controller.auto_ctrl.on_auto_fit()
        monkeypatch.setattr(controller, "_ask_yes_no", lambda title, text: False)
        assert controller._confirm_close() is False
        _wait(qapp, lambda: not controller.auto_ctrl.is_busy())

    def test_close_saves_last_session_when_results_exist(self, gui, make_sequence,
                                                         isolated_config):
        _, controller = gui
        _load(controller, make_sequence("A", 4))
        controller._get_state().radius[1] = 22.0
        assert controller._confirm_close() is True
        saved = isolated_config / "recovery" / "autosave.brt"  # the one recovery file
        assert saved.exists()
        assert load_session(str(saved))["radius"][1] == 22.0
