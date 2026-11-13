"""ML assist through the real MainWindow + AppController offscreen: the Auto
box's state line and button, the hint after a failing Auto run, Train and
apply, Fit All while it is on, the switch and sessions.  The teaching panel
itself: ``test_ml_teach_gui``."""

from __future__ import annotations

import time

import pytest

pytest.importorskip("sklearn")

import _ml_synth as S  # noqa: E402
from bubbletrack.controller import ml_controller  # noqa: E402
from bubbletrack.controller import worker as worker_module  # noqa: E402
from bubbletrack.event_bus import RESULTS_CHANGED  # noqa: E402
from bubbletrack.model import ml_assist  # noqa: E402
from bubbletrack.model.provenance import FitSource  # noqa: E402
from bubbletrack.model.state import update_state  # noqa: E402

LAST = len(S.RADII) - 1


def _wait(qapp, predicate, timeout_s=120.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        qapp.processEvents()
        if predicate():
            return
        time.sleep(0.01)
    raise TimeoutError("condition not reached")


@pytest.fixture
def app(gui, tmp_path, monkeypatch, qapp):
    monkeypatch.setattr(worker_module.BatchWorker, "inline", True)
    window, controller = gui
    monkeypatch.setattr(controller.auto_ctrl, "_ask_manual_policy", lambda _n: "keep")
    controller.file_ctrl.on_folder_selected(str(S.write_recording(tmp_path / "rec")))
    controller._state = update_state(controller._get_state(), detection_mode="edge", reference_frame=0)
    pt = window.left_panel.measure_panel
    pt.set_filter_params({**pt.get_filter_params(), "detection_mode": "edge"})
    pt.filters_changed.emit()
    return window, controller


def _fit_all(qapp, controller):
    controller.auto_ctrl.fit_range(0, LAST)
    _wait(qapp, lambda: not controller.auto_ctrl.is_busy())


def _verify(controller, frames=(3, 7)):
    """The user fixed these frames by hand (their true circles)."""
    state = controller._get_state()
    for f in frames:
        rc, cc, r = S.circle(f)
        state.radius[f] = r
        state.circle_fit_par[f] = [rc, cc]
        state.circle_xy[f] = None
        state.fit_source[f] = FitSource.MANUAL
    controller.bus.emit(RESULTS_CHANGED)


def _box(window):
    return window.left_panel.measure_panel.edge_box


def _teach(controller):
    return controller.ml.teach


# --------------------------------------------------------------------------- #
#  The Auto box
# --------------------------------------------------------------------------- #

def test_auto_box_shows_state_switch_and_button(app):
    window, controller = app
    box = _box(window)
    assert box.ml_button.text() == "Teach…" and box.ml_button.isEnabled()
    assert box.ml_switch.text() == "Off · not taught yet" and not box.ml_switch.isEnabled()
    _verify(controller, (3,))
    assert box.ml_switch.text() == "Off · 1 example" and not box.ml_switch.isEnabled()
    assert "confirm two different frames" in box.ml_switch.toolTip()
    _verify(controller, (3, 7))                # two different frames: taught
    assert box.ml_switch.text() == "Off · 2 examples" and box.ml_switch.isEnabled()


def test_without_scikit_learn_the_button_is_disabled_with_a_reason(qapp, isolated_config, monkeypatch):
    from bubbletrack.controller.controller import AppController
    from bubbletrack.ui.main_window import MainWindow

    monkeypatch.setattr(ml_controller, "ml_available", lambda: False)
    window = MainWindow()
    controller = AppController(window)
    try:
        box = _box(window)
        assert not box.ml_button.isEnabled() and "scikit-learn" in box.ml_button.toolTip()
        controller.ml.open_teaching()
        assert not controller.ml.teach.active
    finally:
        window.close()


def test_hint_after_an_auto_run_where_many_frames_fail(app, qapp, monkeypatch):
    window, controller = app
    _fit_all(qapp, controller)                 # 2 of the bubble's 9 frames have no result
    hint = controller.ml.hint
    assert hint.isHidden()                     # fewer than HINT_MIN_FRAMES: quiet
    monkeypatch.setattr(ml_assist, "HINT_MIN_FRAMES", 2)
    _fit_all(qapp, controller)
    assert not hint.isHidden() and hint.text() == "22% of the bubble's frames have no result."
    assert window.header.status_text().endswith("· ML assist can help")
    hint.open_clicked.emit()
    assert _teach(controller).active and hint.isHidden()


# --------------------------------------------------------------------------- #
#  Apply, Fit All, the switch, sessions
# --------------------------------------------------------------------------- #

def test_train_and_apply_measures_frames_to_check_and_lists_ml_outlines(app, qapp):
    window, controller = app
    _fit_all(qapp, controller)
    _verify(controller)
    state = controller._get_state()
    kept = {f: state.radius[f] for f in (3, 7)}
    good = {f: state.radius[f] for f in range(LAST + 1)
            if f not in controller.ml.scope() and f not in kept}         # Auto's unflagged frames
    assert 6 in controller.ml.scope()                                   # a hard frame
    controller.ml.open_teaching()
    controller.ml.teach.panel.train_clicked.emit()
    _wait(qapp, lambda: not controller.auto_ctrl.is_busy())
    state = controller._get_state()
    assert not _teach(controller).active
    assert {f: state.radius[f] for f in kept} == kept                      # never touched
    assert all(state.fit_source[f] == FitSource.MANUAL for f in kept)
    assert {f: state.radius[f] for f in good} == good                      # nor good results
    assert state.fit_source[6] == FitSource.ML and "ml_outline" in state.fit_flags[6]
    assert all(state.fit_suggestion[f] is None for f in range(LAST + 1) if state.radius[f] > 0)
    assert state.radius[6] == pytest.approx(S.RADII[6], rel=0.1)
    assert any("Outline from ML assist" in text for text in controller.review.panel.texts())
    assert state.ml_assist and _box(window).ml_switch.isChecked()
    assert _box(window).ml_switch.text().startswith("On · ")
    assert window.header.status_text().startswith("ML assist measured ")
    assert "ML assist:" in window.header.status_details()  # its counts, in the tooltip


def test_fit_all_uses_ml_assist_while_on_and_plain_auto_when_off(app, qapp):
    window, controller = app
    _verify(controller)
    _box(window).ml_switch.setChecked(True)
    assert controller._get_state().ml_assist
    _fit_all(qapp, controller)
    state = controller._get_state()
    assert state.fit_source[6] == FitSource.ML
    assert any("ml_guided" in state.fit_flags[f] for f in (1, 2, 4))
    _box(window).ml_switch.setChecked(False)
    assert not controller._get_state().ml_assist
    _fit_all(qapp, controller)
    state = controller._get_state()
    assert FitSource.ML not in set(state.fit_source.tolist())
    assert not any(flags & {"ml_guided", "ml_outline"} for flags in state.fit_flags)


def test_session_keeps_ml_assist_and_a_new_source_clears_it(app, qapp, tmp_path):
    window, controller = app
    _verify(controller)
    controller.ml.open_teaching()
    _teach(controller).panel.set_painting(True)
    window.original_panel.stroke_drawn.emit([(75.0, 55.0), (85.0, 65.0)])
    _box(window).ml_switch.setChecked(True)
    examples = controller._get_state().ml_examples
    path = tmp_path / "s.brt"
    assert controller.save_session_file(str(path))
    controller.file_ctrl.on_folder_selected(str(S.write_recording(tmp_path / "other")))
    state = controller._get_state()
    assert state.ml_examples is None and not state.ml_assist and not _teach(controller).active
    assert controller.load_session_file(str(path))
    state = controller._get_state()
    assert state.ml_assist and state.ml_examples.strokes == examples.strokes
    assert _box(window).ml_switch.isChecked() and _box(window).ml_switch.text().startswith("On")
