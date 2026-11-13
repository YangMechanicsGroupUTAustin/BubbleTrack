"""Accept in Review on frames with Auto's guess, offscreen.

A (Accept) stores the guess on screen only (one Ctrl+Z) and moves to the
row's next frame.  Shift+A (Accept all) stores every result and guess of the
row, and asks once when some guesses were never on screen.  Accept never
means "no bubble": empty frames stay for Fix or No bubble."""

from __future__ import annotations

import numpy as np
import pytest
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication, QMessageBox

from bubbletrack.controller.circle_controller import circle_outline
from bubbletrack.controller.review_queue import ReviewItem
from bubbletrack.model.cases.labels import app_result, frame_label
from bubbletrack.model.cases.schema import BY_ACCEPTED, LABEL_BUBBLE, LABEL_UNLABELLED
from bubbletrack.model.provenance import FitSource
from bubbletrack.model.state import update_state
from bubbletrack.model.suggestion import make_suggestion, shown_suggestion
from bubbletrack.ui import chart_style as cs
from bubbletrack.ui import review_confirm
from bubbletrack.ui.review_rows import tooltip_html
from conftest import write_sequence

SHAPE, CENTRE, R = (60, 90), (30.0, 45.0), 15.0


def _app(gui, tmp_path, n: int):
    window, controller = gui
    folder = write_sequence(tmp_path / f"seq{n}", [R] * n, shape=SHAPE, center=CENTRE)
    controller.file_ctrl.on_folder_selected(str(folder))
    controller._state = update_state(controller._get_state(), detection_mode="edge",
                                     reference_frame=0)
    return window, controller


@pytest.fixture
def app(gui, tmp_path):
    return _app(gui, tmp_path, 12)


def _state(controller):
    return controller._get_state()


def _guess(state, frames, conf=0.2):
    """Rejected frames that show a guess (as Fit All leaves them)."""
    for f in frames:
        state.fit_confidence[f] = conf
        state.fit_flags[f] = frozenset({"circle_fallback"})
        state.fit_suggestion[f] = make_suggestion(
            R + 0.1 * f, *CENTRE, circle_outline(*CENTRE, R), conf,
            frozenset({"circle_fallback"}), {"r_circle": R + 0.5})


def _empty(state, frames, conf=0.1):
    """Rejected frames without a guess."""
    state.fit_confidence[list(frames)] = conf


def _show(controller, frame: int):
    """Display *frame* the way the user does; the Review row follows."""
    controller.review.refresh()
    controller.file_ctrl.on_frame_changed(frame)
    assert controller.review.panel.current_frame() == frame


def _accept(controller):
    controller.review.perform("accept")


def _status(window) -> str:
    return window.header._status_label.text()


def _promoted(state) -> list[int]:
    return [int(i) for i in np.flatnonzero(state.radius > 0)]


@pytest.fixture
def answers(monkeypatch):
    """Answer Accept all's question with the queued replies; records the
    questions and their small print (empty frames that stay)."""
    asked: list[str] = []
    replies: list[bool] = []
    empties: list[int] = []

    def ask(_parent, question, *, empty=0):
        asked.append(question)
        empties.append(empty)
        return replies.pop(0)
    monkeypatch.setattr(review_confirm, "ask_accept_all", ask)
    return asked, replies, empties


# -- Accept: the guess on screen ------------------------------------------------------------

def test_a_stores_only_the_guess_on_screen_then_shows_the_next_frame(app):
    window, controller = app
    state = _state(controller)
    _empty(state, [2, 4, 6, 7, 9])
    _guess(state, [3, 5, 8])
    _show(controller, 5)
    assert controller.review.panel.current_segment().frames == (2, 3, 4, 5, 6, 7, 8, 9)
    assert controller.review.panel.card.buttons["accept"].text() == "Accept the guess"
    _accept(controller)
    assert _promoted(state) == [5] and np.flatnonzero(state.fit_locked).tolist() == [5]
    assert "Frame 6: Auto's guess accepted" in _status(window)
    assert _state(controller).image_no == 6  # the row's next frame
    _accept(controller)  # 6 has nothing to accept: a hint, nothing else
    assert _promoted(state) == [5] and not state.fit_locked[6]
    assert _state(controller).image_no == 6
    assert controller.review.panel.feedback.text() == \
        "Nothing to accept here — F to fix, N if there is no bubble"
    assert not controller.review.panel.feedback.offers_undo()


def test_accept_stores_the_guess_as_a_locked_auto_result(app):
    window, controller = app
    state = _state(controller)
    _guess(state, [3])
    s = state.fit_suggestion[3]
    _show(controller, 3)
    _accept(controller)
    assert state.radius[3] == s.radius and tuple(state.circle_fit_par[3]) == (s.rc, s.cc)
    np.testing.assert_array_equal(state.circle_xy[3], s.edge_xy)
    assert state.fit_confidence[3] == 0.2 and state.fit_flags[3] == s.flags
    assert state.fit_metrics["r_circle"][3] == R + 0.5
    assert state.fit_source[3] == FitSource.AUTO and state.fit_locked[3]
    assert shown_suggestion(state, 3) is None and 3 not in controller.review.panel.frames()
    # user-confirmed everywhere: diamond on the chart, an accepted hard-case label
    chart = window.radius_chart
    pos = int(np.flatnonzero(chart._frames == 4)[0])
    assert cs.marker_symbols(chart._sources, chart._locked, len(chart._frames))[pos] == "d"
    label = frame_label(app_result(state, 3))
    assert (label.status, label.by) == (LABEL_BUBBLE, BY_ACCEPTED)


def test_ctrl_z_undoes_one_accepted_guess(app):
    window, controller = app
    state = _state(controller)
    _guess(state, [3, 5])
    first, second = state.fit_suggestion[3], state.fit_suggestion[5]
    _show(controller, 3)
    _accept(controller)  # 3, then the view moves to 5
    _accept(controller)  # 5
    controller._on_undo()
    assert _promoted(state) == [3] and shown_suggestion(state, 5) is second
    assert "Undid the accept of Auto's guess on frame 6" in _status(window)
    controller._on_undo()
    assert _promoted(state) == [] and not state.fit_locked.any()
    assert shown_suggestion(state, 3) is first and state.fit_source[3] == FitSource.NONE
    assert {3, 5} <= set(controller.review.panel.frames())


def test_a_never_says_no_bubble(app):
    """An empty frame may be a bubble Auto missed: no false "no bubble" label."""
    _, controller = app
    state = _state(controller)
    _empty(state, [3])
    _show(controller, 3)
    _accept(controller)
    assert not state.fit_locked[3] and 3 in controller.review.panel.frames()
    assert frame_label(app_result(state, 3)).status == LABEL_UNLABELLED


# -- Accept all: every result and guess of the row ---------------------------------------

def test_accept_all_stores_every_guess_once_all_were_seen(app, answers):
    window, controller = app
    asked, _, _ = answers
    state = _state(controller)
    _empty(state, [2, 4])
    _guess(state, [3, 5])
    for f in (3, 5):
        controller.file_ctrl.on_frame_changed(f)  # the user looked at both guesses
    _show(controller, 2)
    controller.review.perform("accept_all")
    assert asked == []  # nothing unseen: no question
    assert _promoted(state) == [3, 5] and state.fit_locked[[3, 5]].all()
    assert not state.fit_locked[[2, 4]].any()  # empty: left for Fix or No bubble
    assert ("2 of frames 3–6 accepted and locked (2 of Auto's guesses stored; "
            "2 empty frames left to fix or mark)") in _status(window)
    assert controller.review.panel.frames() == [2, 4]
    assert _state(controller).image_no == 2  # the row's frames left to settle
    controller._on_undo()  # one Ctrl+Z undoes it all
    assert _promoted(state) == [] and not state.fit_locked.any()
    assert all(shown_suggestion(state, f) is not None for f in (3, 5))


def test_accept_all_asks_about_unseen_guesses_and_cancel_stores_nothing(app, answers):
    _, controller = app
    asked, replies, small_print = answers
    state = _state(controller)
    _empty(state, [2, 4, 6, 7, 9])
    _guess(state, [3, 5, 8])
    _show(controller, 3)
    replies.append(False)
    controller.review.perform("accept_all")
    assert asked == ["Accept 3 frames? 2 of them are Auto's guesses you have not looked at."]
    assert small_print == [5]  # frames 2, 4, 6, 7, 9 stay for Fix or No bubble
    assert _promoted(state) == [] and not state.fit_locked.any()
    assert _state(controller).image_no == 3  # cancelled: stay
    replies.append(True)
    controller.review.perform("accept_all")
    assert _promoted(state) == [3, 5, 8]
    assert np.flatnonzero(state.fit_locked).tolist() == [3, 5, 8]


def test_accept_all_on_a_row_without_anything_to_accept_says_so(app, answers):
    _, controller = app
    asked, _, _ = answers
    state = _state(controller)
    _empty(state, [2, 3, 4])
    _show(controller, 2)
    assert controller.review.panel.card.buttons["accept_all"].isHidden()
    controller.review.perform("accept_all")  # Shift+A anyway
    assert asked == [] and not state.fit_locked.any()
    assert controller.review.panel.feedback.text().startswith("Nothing to accept in this row")


def test_the_real_question_box(app, qapp):
    """The dialog says what it will do; Cancel is one click away."""
    window, controller = app
    state = _state(controller)
    _guess(state, [3, 4, 5])
    _show(controller, 3)
    seen = {}

    def answer():
        box = QApplication.activeModalWidget()
        if not isinstance(box, QMessageBox):
            QTimer.singleShot(20, answer)
            return
        seen["text"], seen["detail"] = box.text(), box.informativeText()
        seen["buttons"] = sorted(b.text() for b in box.buttons())
        box.button(QMessageBox.StandardButton.Cancel).click()
    QTimer.singleShot(0, answer)
    controller.review.perform("accept_all")
    assert seen["text"] == "Accept 3 frames? 2 of them are Auto's guesses you have not looked at."
    assert "Ctrl+Z" in seen["detail"] and "empty" not in seen["detail"]
    assert "Accept all" in seen["buttons"] and len(seen["buttons"]) == 2
    assert _promoted(state) == []


# -- texts -----------------------------------------------------------------------------

def test_row_tooltip_and_card_say_what_accept_does(app):
    _, controller = app
    state = _state(controller)
    _empty(state, [2])
    _guess(state, [3, 4])
    _show(controller, 3)
    panel = controller.review.panel
    tip = tooltip_html(panel.current_segment(), state.images)
    assert ("Auto&#x27;s guess on 2 of 3 frames (dashed): Accept (A) stores the one on screen, "
            "Accept all (Shift+A) every one") in tip
    assert "Not a bubble? No bubble (N); Shift+N for the whole row" in tip
    assert panel.texts()[0].endswith("No result — Auto has a guess · 3 frames")
    assert panel.card.advice.text().startswith("No result, but Auto has a guess")
    assert panel.card.recommended() == "accept"


# -- the hard case: one long row (lic_hard 05: 229 frames, 213 guesses) ----------------------

def test_one_press_on_a_229_frame_row_promotes_exactly_one_frame(gui, tmp_path, answers):
    window, controller = _app(gui, tmp_path, 231)
    asked, replies, _ = answers
    state = _state(controller)
    bare = list(range(16, 231, 14))  # 16 of the row's frames have no guess, as on lic_hard 05
    _guess(state, [f for f in range(2, 231) if f not in bare])
    _empty(state, bare)
    _show(controller, 5)
    segment = controller.review.panel.current_segment()
    assert segment.count == 229
    assert "Auto&#x27;s guess on 213 of 229 frames" in tooltip_html(segment, state.images)
    _accept(controller)
    assert _promoted(state) == [5] and int(state.fit_locked.sum()) == 1
    assert _state(controller).image_no == 6
    assert controller.review.panel.current_segment().count == 228
    assert controller.review.panel.card.buttons["accept_all"].text() == "Accept 212 of 228"
    # Accept all on the rest: one question, then 212 guesses stored at once
    replies.append(True)
    controller.review.perform("accept_all")
    assert asked == ["Accept 212 frames? 211 of them are Auto's guesses you have not looked at."]
    assert len(_promoted(state)) == 213 and not state.fit_locked[bare].any()
    assert controller.review.panel.frames() == bare  # the 16 empty frames are left
    assert _state(controller).image_no == bare[0]


# -- other rows with guesses (the tracker's "Auto lost the bubble") --------------------------

def test_lost_track_rows_follow_the_same_rule(app):
    _, controller = app
    state = _state(controller)
    _empty(state, [2, 4, 6])
    _guess(state, [3, 5])
    controller.review.set_extra([ReviewItem(f, ("lost",), 100) for f in range(2, 7)])
    _show(controller, 3)
    assert controller.review.panel.current_segment().label == "Auto lost the bubble"
    _accept(controller)
    assert _promoted(state) == [3] and _state(controller).image_no == 4
    _show(controller, 5)
    _accept(controller)  # only the frame on screen
    assert np.flatnonzero(state.fit_locked).tolist() == [3, 5]
    _show(controller, 2)
    _accept(controller)  # nothing to accept on an empty frame
    assert not state.fit_locked[2]
