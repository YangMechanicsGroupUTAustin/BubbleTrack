"""Random sequences of user actions on the real window (offscreen, synthetic data).

Every exception raised by an action, every exception escaping a Qt slot
(sys.excepthook) and every ERROR log record (the event bus swallows handler
errors) fails the run, with the action trail that led to it.  A smoke test of
how actions interact (load, fit, refit, chart ranges, Review, undo, sessions,
export, reference and seed, method switches); 30 seeds x 150 actions passed
when it was written (2026-11-18), this keeps a quick subset.
"""

import logging
import sys
import time
import traceback

import numpy as np
import pytest

from bubbletrack.controller import worker as worker_mod
from bubbletrack.model.state import update_state
from bubbletrack.ui.review_panel import ReviewPanel


def _wait(qapp, predicate, timeout_s=60.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        qapp.processEvents()
        if predicate():
            return True
        time.sleep(0.005)
    return False


@pytest.mark.parametrize("seed", list(range(4)))
def test_monkey(qapp, gui, make_sequence, tmp_path, monkeypatch, caplog, request, seed):
    import faulthandler

    from PyQt6.QtWidgets import QMessageBox

    from bubbletrack.ui.export_dialog import ExportDialog

    faulthandler.dump_traceback_later(240, exit=True)  # a hang prints the stacks and ends the run
    request.addfinalizer(faulthandler.cancel_dump_traceback_later)  # whatever fails below
    caplog.set_level(logging.WARNING)
    monkeypatch.setattr(worker_mod.BatchWorker, "inline", True)  # threads, not processes
    # No modal dialog may wait for a click offscreen: answer at once
    monkeypatch.setattr(QMessageBox, "exec", lambda self: 0)
    for name in ("question", "warning", "information", "critical"):
        monkeypatch.setattr(QMessageBox, name, staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes))
    monkeypatch.setattr(ExportDialog, "confirm_overwrite", lambda self, paths: True)
    window, ctl = gui
    rng = np.random.default_rng(seed)
    slot_errors = []
    monkeypatch.setattr(sys, "excepthook",
                        lambda t, v, tb: slot_errors.append("".join(traceback.format_exception(t, v, tb))))
    folders = [make_sequence(f"a{seed}", 12, r_min=15, r_max=40), make_sequence(f"b{seed}", 7, r_min=25, r_max=30)]
    ctl.retrack.timer.setInterval(30)
    trail = []

    def n_frames():
        return len(ctl._get_state().images or ())

    def frame():
        return int(rng.integers(0, max(n_frames(), 1)))

    def busy():
        return ctl.auto_ctrl.is_busy()

    def load():
        ctl.file_ctrl.on_folder_selected(str(folders[int(rng.integers(0, 2))]))

    def goto():
        if n_frames():
            ctl.file_ctrl.on_frame_changed(frame())

    def fit_all():
        if n_frames() and not busy():
            ctl.auto_ctrl.on_auto_fit()
            _wait(qapp, lambda: not busy())

    def fit_range():
        if n_frames() and not busy():
            a, b = sorted((frame(), frame()))
            window.radius_chart.refit_range_requested.emit(a, b)
            _wait(qapp, lambda: not busy())

    def mode():
        modes = ["edge", "edge", "intensity", "gradient", "area"]
        ctl._set_state(update_state(ctl._get_state(), detection_mode=modes[int(rng.integers(0, 5))]))

    def chart_range():
        if n_frames():
            a, b = sorted((frame(), frame()))
            sig = [window.radius_chart.lock_range_requested, window.radius_chart.unlock_range_requested,
                   window.radius_chart.no_bubble_range_requested][int(rng.integers(0, 3))]
            sig.emit(a, b)

    def undo():
        ctl._on_undo()

    def review():
        if n_frames():
            ctl.review.perform(ReviewPanel.ACTIONS[int(rng.integers(0, len(ReviewPanel.ACTIONS)))])

    def fix_cancel():
        if ctl.review.fix.active:
            ctl.review.fix.cancel()

    def session():
        if n_frames():
            path = tmp_path / f"s{seed}.brt"
            if ctl.save_session_file(str(path)):
                ctl.load_session_file(str(path))

    def export():
        if n_frames() and not busy():
            dialog = ctl.export_ctrl.make_dialog()
            dialog.set_units("pixel")
            dialog.set_path(str(tmp_path / f"r{seed}_{len(trail)}.csv"))
            ctl.export_ctrl.export_from(dialog)

    def reference():
        if n_frames():
            goto()
            ctl.pretune_ctrl.edge_setup.use_current()

    def seed_click():
        if n_frames():
            ctl.pretune_ctrl.edge_setup.on_point_clicked(float(rng.uniform(0, 160)), float(rng.uniform(0, 120)))

    def non_round():
        ctl._set_state(update_state(ctl._get_state(), bubble_non_round=bool(rng.integers(0, 2))))

    def clear():
        if not busy():
            ctl._on_clear_results()

    def settle():
        _wait(qapp, lambda: not ctl.retrack.is_running() and not ctl.retrack.timer.isActive(), 10)

    actions = [load, goto, goto, fit_all, fit_range, mode, chart_range, chart_range, undo, undo, review,
               review, review, fix_cancel, session, export, reference, seed_click, non_round, clear, settle]
    load()
    for _ in range(80):
        act = actions[int(rng.integers(0, len(actions)))]
        trail.append(act.__name__)
        try:
            act()
        except Exception:  # noqa: BLE001
            pytest.fail(f"{act.__name__} raised after {trail}:\n{traceback.format_exc()}")
        qapp.processEvents()
        assert not slot_errors, f"slot error after {trail}:\n{slot_errors[0]}"
        bad = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert not bad, (f"error log after {trail}: {bad[0].getMessage()}\n"
                         f"{bad[0].exc_text or ''}")
    settle()
    assert not slot_errors, slot_errors[:1]
