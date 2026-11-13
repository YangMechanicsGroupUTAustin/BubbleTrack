"""ML assist through the real MainWindow + AppController offscreen: the Auto
box's state line and button, the hint after a failing Auto run, the teaching
mode (strokes, keys, wheel, undo, live preview, "Correct"), Apply, Fit All
while it is on, the switch, the run lock and sessions."""

from __future__ import annotations

import time

import pytest

pytest.importorskip("sklearn")

import _ml_synth as S  # noqa: E402
from bubbletrack.controller import ml_controller  # noqa: E402
from bubbletrack.controller import worker as worker_module  # noqa: E402
from bubbletrack.event_bus import RESULTS_CHANGED  # noqa: E402
from bubbletrack.model.provenance import FitSource  # noqa: E402
from bubbletrack.model.state import update_state  # noqa: E402
from bubbletrack.ui.image_compare import CompareMode  # noqa: E402

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
    pt = window.left_panel.pretune_tab
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
    return window.left_panel.pretune_tab.edge_box


def _teach(controller):
    return controller.ml.teach


def _wait_model(qapp, controller):
    teach = _teach(controller)
    _wait(qapp, lambda: teach._model is not None and teach._model_version == teach._version
          and controller._get_state().image_no in teach._previews, 120)


# --------------------------------------------------------------------------- #
#  The Auto box
# --------------------------------------------------------------------------- #

def test_auto_box_shows_state_switch_and_button(app):
    window, controller = app
    box = _box(window)
    assert box.ml_button.text() == "ML assist…" and box.ml_button.isEnabled()
    assert box.ml_state_label.text() == "Off" and not box.ml_switch.isEnabled()
    _verify(controller, (3,))
    assert box.ml_state_label.text() == "Off (1 example)" and box.ml_switch.isEnabled()


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


def test_hint_after_an_auto_run_where_many_frames_fail(app, qapp):
    window, controller = app
    _fit_all(qapp, controller)
    hint = controller.ml.hint
    assert not hint.isHidden() and "ML assist can help" in hint.text()
    assert "try ML assist" in window.header.status_text()
    hint.open_clicked.emit()
    assert _teach(controller).active and hint.isHidden()


# --------------------------------------------------------------------------- #
#  Teaching mode
# --------------------------------------------------------------------------- #

def test_teaching_mode_paints_with_keys_wheel_and_undo(app):
    window, controller = app
    controller.ml.open_teaching()
    teach, panel = _teach(controller), window.original_panel
    assert teach.active and not controller.ml.dock.isHidden()
    assert panel.mode == "paint" and window.compare_mode is CompareMode.SIDE_BY_SIDE
    assert len(teach._suggested) == 2 and 0 not in teach._suggested       # never the reference
    idx = controller._get_state().image_no
    assert idx in teach._suggested
    panel.stroke_drawn.emit([(75.0, 55.0), (85.0, 65.0)])                  # bubble brush
    teach._keys[1].activated.emit()                                         # G: background
    assert teach.tool() == "background"
    panel.stroke_drawn.emit([(10.0, 10.0), (40.0, 10.0)])
    strokes = controller._get_state().ml_examples.strokes[idx]
    assert len(strokes.bubble) > 10 and len(strokes.background) > 10
    assert teach.panel.quality_text() == "0 verified frames, 2 strokes — ready"
    before = teach.radius
    panel.brush_scrolled.emit(1)
    assert teach.radius > before and teach.panel.brush_size() == teach.radius
    controller._on_undo()                                                   # Ctrl+Z: the stroke
    assert not controller._get_state().ml_examples.strokes[idx].background
    teach._keys[2].activated.emit()                                         # E: eraser
    panel.stroke_drawn.emit([(75.0, 55.0), (85.0, 65.0)])
    assert idx not in controller._get_state().ml_examples.strokes
    assert teach.panel.quality_text().endswith("needs at least one bubble and one background example")


def test_strokes_outside_the_roi_are_refused(app):
    window, controller = app
    controller._state = update_state(controller._get_state(), gridx=(20, 100), gridy=(20, 140))
    controller.ml.open_teaching()
    window.original_panel.stroke_drawn.emit([(3.0, 3.0), (8.0, 3.0)])
    assert controller._get_state().ml_examples is None
    assert "inside the ROI" in window.header.status_text()


def test_live_preview_after_verified_frames(app, qapp):
    window, controller = app
    _verify(controller)
    controller.ml.open_teaching()
    teach = _teach(controller)
    assert teach.panel.quality_text() == "2 verified frames, 0 strokes — ready"
    _wait_model(qapp, controller)
    assert window.binary_panel._title_label.text() == "ML assist: P(bubble)"
    assert teach.panel.preview_text().startswith(f"Frame {controller._get_state().image_no + 1}: ")
    _wait(qapp, lambda: all(f in teach._previews for f in teach._suggested))
    for f in teach._suggested:
        assert "px" in teach.panel.card_caption(f) or "no result" in teach.panel.card_caption(f)
    controller.file_ctrl.on_frame_changed(6)                                # a hard frame
    _wait(qapp, lambda: 6 in teach._previews)
    assert "ML outline" in teach.panel.preview_text()


def test_correct_keeps_mls_result_locked_and_undoable(app, qapp):
    window, controller = app
    _verify(controller)
    controller.ml.open_teaching()
    teach = _teach(controller)
    _wait(qapp, lambda: all(f in teach._previews for f in teach._suggested), 120)
    frame = teach._suggested[0]
    teach.panel.accept_requested.emit(frame)
    state = controller._get_state()
    assert state.fit_locked[frame] and state.radius[frame] > 0
    assert frame in teach._examples.verified and "✓" in teach.panel.card_caption(frame)
    controller._on_undo()                                                   # no strokes: the edit
    assert not controller._get_state().fit_locked[frame]


def test_run_lock_stops_painting(app):
    window, controller = app
    controller.ml.open_teaching()
    controller.bus.emit("run_state_changed", True)
    assert window.original_panel.mode == "normal" and not _teach(controller).panel._apply_btn.isEnabled()
    controller.bus.emit("run_state_changed", False)
    assert window.original_panel.mode == "paint"


# --------------------------------------------------------------------------- #
#  Apply, Fit All, the switch, sessions
# --------------------------------------------------------------------------- #

def test_apply_measures_unconfirmed_frames_and_lists_ml_outlines(app, qapp):
    window, controller = app
    _fit_all(qapp, controller)
    _verify(controller)
    kept = {f: controller._get_state().radius[f] for f in (3, 7)}
    controller.ml.open_teaching()
    _wait_model(qapp, controller)
    controller.ml.teach.panel.apply_clicked.emit()
    assert not _teach(controller).active and controller.auto_ctrl.is_busy()
    _wait(qapp, lambda: not controller.auto_ctrl.is_busy())
    state = controller._get_state()
    assert {f: state.radius[f] for f in kept} == kept                      # never touched
    assert all(state.fit_source[f] == FitSource.MANUAL for f in kept)
    assert state.fit_source[6] == FitSource.ML and "ml_outline" in state.fit_flags[6]
    assert state.radius[6] == pytest.approx(S.RADII[6], rel=0.1)
    assert any("ml_guided" in state.fit_flags[f] for f in (1, 2, 4))
    assert any(text.endswith("ML outline") for text in controller.review.panel.texts())
    assert state.ml_assist and _box(window).ml_switch.isChecked()
    assert _box(window).ml_state_label.text().startswith("On (")
    assert "ML assist:" in window.header.status_text()


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
    assert _box(window).ml_switch.isChecked() and _box(window).ml_state_label.text().startswith("On")
