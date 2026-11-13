"""ML assist's teaching panel through the real MainWindow + AppController offscreen.

* One owner of the image's mouse at a time: teaching and Review's Fix (the
  circle tool) never run together, whichever starts first and from wherever
  (Review's card button, the F key, the chart's "Fix frame N...", the Auto
  box's "Teach...", the Review hint).
* The flow: candidate frames with Looks right / Looks wrong, two different
  confirmed frames, no training before Train and apply, which measures only
  the frames that need review, then hands over to Review.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

pytest.importorskip("sklearn")

import _ml_synth as S  # noqa: E402
from PyQt6.QtCore import QEvent, QPointF, Qt  # noqa: E402
from PyQt6.QtGui import QMouseEvent  # noqa: E402

from bubbletrack.controller import ml_controller  # noqa: E402
from bubbletrack.controller import worker as worker_module  # noqa: E402
from bubbletrack.event_bus import RESULTS_CHANGED  # noqa: E402
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
    """The synthetic ML recording, fitted with plain Auto, in a shown window (the
    docks tab as for the user; Review's keys act while its dock is visible)."""
    monkeypatch.setattr(worker_module.BatchWorker, "inline", True)
    window, controller = gui
    window.resize(1440, 900)
    window.show()
    qapp.processEvents()
    monkeypatch.setattr(controller.auto_ctrl, "_ask_manual_policy", lambda _n: "keep")
    controller.file_ctrl.on_folder_selected(str(S.write_recording(tmp_path / "rec")))
    controller._state = update_state(controller._get_state(), detection_mode="edge", reference_frame=0)
    pt = window.left_panel.measure_panel
    pt.set_filter_params({**pt.get_filter_params(), "detection_mode": "edge"})
    pt.filters_changed.emit()
    controller.auto_ctrl.fit_range(0, LAST)
    _wait(qapp, lambda: not controller.auto_ctrl.is_busy())
    return window, controller


@pytest.fixture
def trainings(monkeypatch):
    """Counts every model training (the panel must never train by itself)."""
    from bubbletrack.model.teach import model as teach_model

    calls = []
    real = teach_model.train_model

    def counted(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(teach_model, "train_model", counted)
    monkeypatch.setattr("bubbletrack.model.teach.train_model", counted)
    return calls


def _pump(qapp, seconds):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        qapp.processEvents()
        time.sleep(0.01)


def _teach(controller):
    return controller.ml.teach


def _panel(controller):
    return controller.ml.teach.panel


def _mouse_drag(window, points):
    """A real left-button drag on the image view (scene (x, y) points)."""
    view = window.original_panel._view

    def send(kind, xy):
        p = view.mapFromScene(QPointF(*xy))
        held = Qt.MouseButton.NoButton if kind == QEvent.Type.MouseButtonRelease else Qt.MouseButton.LeftButton
        event = QMouseEvent(kind, QPointF(p), QPointF(p), Qt.MouseButton.LeftButton, held,
                            Qt.KeyboardModifier.NoModifier)
        {QEvent.Type.MouseButtonPress: view.mousePressEvent,
         QEvent.Type.MouseMove: view.mouseMoveEvent,
         QEvent.Type.MouseButtonRelease: view.mouseReleaseEvent}[kind](event)

    send(QEvent.Type.MouseButtonPress, points[0])
    for xy in points[1:]:
        send(QEvent.Type.MouseMove, xy)
    send(QEvent.Type.MouseButtonRelease, points[-1])


def _gestures(window):
    seen = []
    window.original_panel.circle_drawn.connect(lambda *_a: seen.append("circle"))
    window.original_panel.stroke_drawn.connect(lambda *_a: seen.append("stroke"))
    return seen


def _stroke_points():
    rc, cc = S.CENTRE
    return [(cc - 6 + k, rc) for k in range(0, 13, 2)]


def _start_fix_on_a_card_frame(controller):
    """F in Review on a frame the teaching panel will show (so opening it
    does not move to another frame, which used to end the fix by chance)."""
    controller.ml.open_teaching()
    frame = _teach(controller).slots[0]
    controller.ml.teach.leave()
    controller.file_ctrl.on_frame_changed(frame)
    controller.review.perform("fix")
    assert controller.review.fix.active and controller.circle_ctrl.active
    return frame


# --------------------------------------------------------------------------- #
#  Bug 1: the Fix circle tool and teaching never share the image's mouse
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("opened_from", ["review hint", "auto box"])
def test_opening_teaching_ends_a_running_fix(app, opened_from):
    window, controller = app
    _start_fix_on_a_card_frame(controller)
    if opened_from == "review hint":
        controller.ml.hint.open_clicked.emit()
    else:
        window.left_panel.measure_panel.edge_box.ml_assist_clicked.emit()
    teach = _teach(controller)
    assert teach.active
    assert not controller.review.fix.active and not controller.circle_ctrl.active
    assert window.original_panel.mode == "normal"          # the panel's painting is closed
    teach.panel.set_painting(True)
    assert window.original_panel.mode == "paint"
    seen = _gestures(window)
    _mouse_drag(window, _stroke_points())                   # the owner's Bubble-brush stroke
    assert seen == ["stroke"]


def test_opening_teaching_drops_a_refused_circle_and_its_prompt(app):
    window, controller = app
    frame = _start_fix_on_a_card_frame(controller)
    rc, cc = S.CENTRE
    controller.circle_ctrl.on_circle_drawn(cc + 40.0, rc + 30.0, 6.0, True)   # beside the bubble
    strip = controller.review.fix.strip
    assert strip.prompt_visible() and "Nothing was stored" in strip.prompt_text()
    controller.ml.hint.open_clicked.emit()
    assert not strip.prompt_visible() and controller.circle_ctrl._pending is None
    assert controller._get_state().image_no == frame
    assert "Fix of frame" in window.header.status_text()


@pytest.mark.parametrize("how", ["card button", "F key", "chart menu"])
def test_fix_while_teaching_leaves_teaching_first(app, how):
    window, controller = app
    controller.ml.open_teaching()
    teach, review = _teach(controller), controller.review
    teach.panel.set_painting(True)
    frame = controller._get_state().image_no
    if how == "card button":
        review.panel.action_requested.emit("fix")
    elif how == "F key":
        review.shortcuts.shortcuts["fix"].activated.emit()
    else:
        window.radius_chart.fix_requested.emit(frame)
    assert not teach.active and controller.ml.dock.isHidden()
    assert review.fix.active and window.original_panel.mode == "circle"
    assert not any(k.isEnabled() for k in teach.paint.keys)          # B / G / E are free again
    assert "ML assist is closed" in window.header.status_text()
    seen = _gestures(window)
    _mouse_drag(window, _stroke_points())
    assert seen == ["circle"]                                    # the fix owns the mouse now


def test_after_a_fix_teaching_opens_again_with_its_work(app):
    window, controller = app
    controller.ml.open_teaching()
    teach = _teach(controller)
    first = teach.slots[0]
    teach.looks_right(first)
    review = controller.review
    review.perform("fix")
    review.fix.cancel()
    assert window.original_panel.mode == "normal"
    controller.ml.open_teaching()
    assert teach.active and first in teach.slots
    assert teach.panel.card_confirmed(first)


def test_painting_comes_back_after_another_tool_borrowed_the_image(app):
    window, controller = app
    controller.ml.open_teaching()
    _teach(controller).panel.set_painting(True)
    panel = window.original_panel
    panel.set_mode("point")                                       # e.g. "Click bubble"
    panel.set_mode("normal")                                      # ... clicked
    assert panel.mode == "paint"


# --------------------------------------------------------------------------- #
#  The flow: candidates, Looks right / wrong, two different frames
# --------------------------------------------------------------------------- #

def test_opening_shows_two_different_candidates_and_step_one(app):
    window, controller = app
    controller.ml.hint.open_clicked.emit()
    teach, panel = _teach(controller), _panel(controller)
    a, b = teach.slots
    assert a is not None and b is not None and a != b and 0 not in (a, b)   # never the reference
    assert controller._get_state().image_no == a
    assert panel.step_text() == "Step 1 of 2: confirm 2 frames where the outline is right — 0 of 2 done"
    assert panel.next_text().startswith("Judge each frame: Looks right ✓")
    assert panel.question_text() == "Is the red outline on the bubble wall?"
    assert not panel.train_button.isEnabled()
    assert panel.train_button.toolTip() == "Confirm 2 different frames first (0 of 2 done)"
    assert not panel.painting_open() and window.original_panel.mode == "normal"


def test_looks_wrong_replaces_the_candidate_and_never_offers_it_again(app):
    window, controller = app
    controller.ml.open_teaching()
    teach, panel = _teach(controller), _panel(controller)
    seen = set(teach.slots)
    for _ in range(3):
        wrong = teach.slots[0]
        panel.reject_requested.emit(wrong)
        new = teach.slots[0]
        if new is None:
            break
        assert new not in seen and new != teach.slots[1]
        seen.add(new)
    assert teach.rejected and not (teach.rejected & set(s for s in teach.slots if s is not None))
    state = controller._get_state()
    assert not any(state.fit_locked[f] for f in teach.rejected)  # Looks wrong stores nothing


def test_running_out_of_candidates_points_to_painting(app):
    window, controller = app
    controller.ml.open_teaching()
    teach, panel = _teach(controller), _panel(controller)
    for _ in range(LAST + 1):
        frames = [s for s in teach.slots if s is not None]
        if not frames:
            break
        panel.reject_requested.emit(frames[0])
    assert teach.slots == (None, None)
    assert "Paint examples by hand" in panel.next_text() and panel.next_is_warning()


def test_looks_right_confirms_without_training_and_two_different_frames_enable_train(app, qapp, trainings):
    window, controller = app
    controller.ml.open_teaching()
    teach, panel = _teach(controller), _panel(controller)
    a, b = teach.slots
    qapp.processEvents()                                         # the fit's quiet turn ends
    panel.confirm_requested.emit(a)
    state = controller._get_state()
    assert state.fit_locked[a] and state.radius[a] > 0           # accepted: an example now
    assert not controller.retrack.timer.isActive()               # and no re-track follows
    assert panel.card_confirmed(a) and teach.slots[0] == a
    assert panel.step_text().endswith("— 1 of 2 done")
    assert panel.next_text() == f"One more: judge the other frame, which differs from frame {a + 1}."
    assert not panel.train_button.isEnabled()
    panel.confirm_requested.emit(b)
    assert panel.step_text() == "Step 2 of 2: Train and apply — 2 different frames confirmed"
    assert panel.train_button.isEnabled() and panel.next_text().startswith("Press Train and apply")
    controller._on_undo()                                        # Ctrl+Z: the last confirmation
    assert not controller._get_state().fit_locked[b] and not panel.card_confirmed(b)
    assert not panel.train_button.isEnabled()
    _pump(qapp, 0.6)
    assert trainings == []                                       # nothing trained meanwhile


def test_frames_verified_in_review_count_and_show_as_confirmed(app):
    window, controller = app
    state = controller._get_state()
    for f in (1, 4):
        state.fit_locked[f] = state.radius[f] > 0 or state.fit_locked[f]
        state.radius[f], state.circle_fit_par[f] = S.circle(f)[2], list(S.circle(f)[:2])
        state.fit_source[f] = FitSource.MANUAL
    controller.bus.emit(RESULTS_CHANGED)
    controller.ml.open_teaching()
    teach, panel = _teach(controller), _panel(controller)
    assert set(teach.slots) == {1, 4} and all(panel.card_confirmed(f) for f in (1, 4))
    assert panel.step_text().startswith("Step 2 of 2") and panel.train_button.isEnabled()


# --------------------------------------------------------------------------- #
#  Painting by hand: folded away, keys only while open, never trains
# --------------------------------------------------------------------------- #

def test_painting_section_is_folded_and_its_keys_live_only_while_open(app, qapp, trainings):
    window, controller = app
    controller.ml.open_teaching()
    teach, panel = _teach(controller), _panel(controller)
    assert not panel.painting_open() and not any(k.isEnabled() for k in teach.paint.keys)
    window.original_panel.stroke_drawn.emit(_stroke_points())    # ignored while folded
    assert controller._get_state().ml_examples is None
    panel.set_painting(True)
    assert all(k.isEnabled() for k in teach.paint.keys) and window.original_panel.mode == "paint"
    teach.paint.keys[1].activated.emit()                              # G: background
    assert teach.paint.tool() == "background"
    window.original_panel.stroke_drawn.emit([(5.0, 5.0), (30.0, 5.0)])
    teach.paint.keys[0].activated.emit()                              # B: bubble
    window.original_panel.stroke_drawn.emit(_stroke_points())
    idx = controller._get_state().image_no
    strokes = controller._get_state().ml_examples.strokes[idx]
    assert strokes.bubble and strokes.background
    assert "Painted: 2 strokes on 1 frame" in panel.paint_summary()
    assert idx in teach.progress().frames                        # a painted example frame
    controller._on_undo()                                        # Ctrl+Z: the last stroke
    assert not controller._get_state().ml_examples.strokes[idx].bubble
    panel.set_painting(False)
    assert not any(k.isEnabled() for k in teach.paint.keys) and window.original_panel.mode == "normal"
    _pump(qapp, 0.6)
    assert trainings == []


def test_a_painted_frames_card_follows_its_strokes(app):
    from bubbletrack.model.teach import BACKGROUND, BUBBLE, MLExamples

    window, controller = app
    controller.ml.open_teaching()
    teach = _teach(controller)
    first = MLExamples().paint(9, BUBBLE, [(60, 80), (60, 81)]).paint(9, BACKGROUND, [(10, 10)])
    moved = MLExamples().paint(9, BUBBLE, [(62, 80), (62, 81)]).paint(9, BACKGROUND, [(10, 10)])
    strokes = [first.strokes[9], moved.strokes[9]]     # as many pixels, other places
    pictures = [teach._picture(9, None, s) for s in strokes]
    assert pictures[0] is not pictures[1]


def test_strokes_outside_the_roi_are_refused(app):
    window, controller = app
    controller._state = update_state(controller._get_state(), gridx=(20, 100), gridy=(20, 140))
    controller.ml.open_teaching()
    _panel(controller).set_painting(True)
    window.original_panel.stroke_drawn.emit([(3.0, 3.0), (8.0, 3.0)])
    assert controller._get_state().ml_examples is None
    assert "inside the ROI" in window.header.status_text()


def test_run_lock_stops_painting(app):
    window, controller = app
    controller.ml.open_teaching()
    panel = _panel(controller)
    panel.set_painting(True)
    controller.bus.emit("run_state_changed", True)
    assert window.original_panel.mode == "normal" and not panel.train_button.isEnabled()
    assert panel.next_text() == "Wait for the running fit to finish."
    controller.bus.emit("run_state_changed", False)
    assert window.original_panel.mode == "paint"


# --------------------------------------------------------------------------- #
#  Train and apply
# --------------------------------------------------------------------------- #

def _confirm_two(controller):
    teach = _teach(controller)
    for _ in range(2):
        frame = next(f for f in teach.slots if f is not None and not teach.panel.card_confirmed(f))
        teach.panel.confirm_requested.emit(frame)
    assert teach.progress().ready


def _snapshot(state, frames):
    return {f: (float(state.radius[f]), tuple(np.round(state.circle_fit_par[f], 6)),
                int(state.fit_source[f]), frozenset(state.fit_flags[f]), bool(state.fit_locked[f]))
            for f in frames}


def test_train_and_apply_measures_only_the_frames_that_need_review(app, qapp, monkeypatch):
    window, controller = app
    controller.ml.open_teaching()
    _confirm_two(controller)
    state = controller._get_state()
    scope = controller.ml.scope()
    listed = controller.review.queued_frames()
    no_result = {f for f in range(LAST + 1) if not state.radius[f] > 0}
    user = {f for f in range(LAST + 1) if state.fit_locked[f]}
    assert set(scope) == (listed | no_result) - user
    untouched = sorted(set(range(LAST + 1)) - set(scope))
    before = _snapshot(state, untouched)
    calls = []
    real = controller.auto_ctrl.run_ml_assist
    monkeypatch.setattr(controller.auto_ctrl, "run_ml_assist",
                        lambda job, **kw: calls.append(kw["frames"]) or real(job, **kw))
    _panel(controller).train_button.click()
    _wait(qapp, lambda: not controller.auto_ctrl.is_busy())
    assert calls == [scope]
    assert _snapshot(controller._get_state(), untouched) == before   # good results never touched
    assert not _teach(controller).active and controller.ml.dock.isHidden()
    assert not controller.review.dock.isHidden()
    text = window.header.status_text()
    assert text.startswith("ML assist measured ") and f"of {len(scope)} frames to check" in text
    assert controller.ml.hint.text() == text and not controller.ml.hint.isHidden()
    assert controller._get_state().ml_assist


def test_train_and_apply_replaces_a_pending_retrack(app, qapp):
    window, controller = app
    controller.ml.open_teaching()
    _confirm_two(controller)
    qapp.processEvents()                                         # the fit's quiet turn ends
    controller.bus.emit(RESULTS_CHANGED)                         # an edit elsewhere: countdown
    assert controller.retrack.timer.isActive()
    _panel(controller).train_button.click()
    assert not controller.retrack.timer.isActive()
    _wait(qapp, lambda: not controller.auto_ctrl.is_busy())
    assert window.header.status_text().startswith("ML assist measured ")


def test_a_training_that_fails_says_what_to_do(app, monkeypatch):
    window, controller = app
    controller.ml.open_teaching()
    _confirm_two(controller)

    def fails(_job):
        raise ValueError("no background pixels")

    monkeypatch.setattr(ml_controller, "train_job", fails)
    panel = _panel(controller)
    panel.train_button.click()
    assert not controller.auto_ctrl.is_busy() and _teach(controller).active
    assert panel.next_is_warning() and "could not learn from these frames" in panel.next_text()
    assert "no background pixels" in window.header.status_text()
    assert panel.train_button.isEnabled()                        # it can be tried again


def test_train_and_apply_waits_for_two_different_frames(app):
    window, controller = app
    controller.ml.open_teaching()
    teach = _teach(controller)
    teach.panel.confirm_requested.emit(teach.slots[0])
    controller.ml.apply()
    assert not controller.auto_ctrl.is_busy() and teach.active
    assert "Confirm 2 different frames first" in window.header.status_text()


def test_train_and_apply_shows_progress_in_the_panel(app, qapp):
    window, controller = app                     # the run is on its own thread
    controller.ml.open_teaching()
    _confirm_two(controller)
    panel = _panel(controller)
    panel.train_button.click()
    assert controller.auto_ctrl.is_busy() and _teach(controller).active
    assert panel.next_text().startswith("Measuring ") and not panel.train_button.isEnabled()
    assert not panel._progress.isHidden() and not panel._stop_btn.isHidden()
    _wait(qapp, lambda: not controller.auto_ctrl.is_busy())
    assert not _teach(controller).active


def test_without_scikit_learn_the_teaching_mode_does_not_open(qapp, isolated_config, monkeypatch):
    from bubbletrack.controller.controller import AppController
    from bubbletrack.ui.main_window import MainWindow

    monkeypatch.setattr(ml_controller, "ml_available", lambda: False)
    window = MainWindow()
    controller = AppController(window)
    try:
        controller.ml.open_teaching()
        assert not controller.ml.teach.active
    finally:
        window.close()
