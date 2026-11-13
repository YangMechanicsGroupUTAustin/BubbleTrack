"""Accept (A) in Review on frames with Auto's guess, offscreen: a guess becomes a
result only once the user has seen it.  A on a frame showing a guess stores
that guess only (one Ctrl+Z) and shows the row's next guess; A on a frame
without one accepts the row but leaves every guess alone."""

from __future__ import annotations

import numpy as np
import pytest

from bubbletrack.controller.circle_controller import circle_outline
from bubbletrack.controller.review_queue import ReviewItem
from bubbletrack.model.cases.labels import app_result, frame_label
from bubbletrack.model.cases.schema import BY_ACCEPTED, LABEL_BUBBLE, LABEL_NO_BUBBLE
from bubbletrack.model.provenance import FitSource
from bubbletrack.model.state import update_state
from bubbletrack.model.suggestion import make_suggestion, shown_suggestion
from bubbletrack.ui import chart_style as cs
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


# -- a guess on screen ---------------------------------------------------------------------

def test_a_stores_only_the_guess_on_screen_then_shows_the_next(app):
    window, controller = app
    state = _state(controller)
    _empty(state, [2, 4, 6, 7, 9])
    _guess(state, [3, 5, 8])
    _show(controller, 5)
    assert controller.review.panel.current_segment().frames == (2, 3, 4, 5, 6, 7, 8, 9)
    _accept(controller)
    assert _promoted(state) == [5] and np.flatnonzero(state.fit_locked).tolist() == [5]
    assert "Frame 6: Auto's guess accepted" in _status(window)
    assert _state(controller).image_no == 8  # the row's next guess
    _accept(controller)
    assert _promoted(state) == [5, 8]
    assert _state(controller).image_no == 3  # wraps to the row's first
    _accept(controller)
    assert _promoted(state) == [3, 5, 8]
    assert _state(controller).image_no == 4  # no guess left: the next listed frame, as before
    assert not state.fit_locked[[2, 4, 6, 7, 9]].any()


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


# -- no guess on screen ----------------------------------------------------------------------

def test_a_without_a_guess_accepts_the_row_but_skips_every_guess(app):
    window, controller = app
    state = _state(controller)
    _empty(state, [2, 4, 6, 7, 9])
    _guess(state, [3, 5, 8])
    _show(controller, 2)
    detail = controller.review.panel._detail.text()
    assert detail.endswith("A skips the 3 frames with a guess")
    _accept(controller)
    assert _promoted(state) == []  # nobody has looked at the guesses
    assert np.flatnonzero(state.fit_locked).tolist() == [2, 4, 6, 7, 9]  # "no bubble"
    assert all(shown_suggestion(state, f) is not None for f in (3, 5, 8))
    assert "3 frame(s) with Auto's guess left to look at" in _status(window)
    assert _state(controller).image_no == 3  # the guesses still need their own look
    assert controller.review.panel.current_segment().frames == (3, 5)  # 8 is a row of its own now


def test_a_without_any_guess_still_means_no_bubble(app):
    _, controller = app
    state = _state(controller)
    _empty(state, [3])
    _show(controller, 3)
    _accept(controller)
    assert state.radius[3] == -1.0 and state.fit_locked[3]
    assert frame_label(app_result(state, 3)).status == LABEL_NO_BUBBLE


# -- texts ---------------------------------------------------------------------------------

def test_row_tooltip_and_detail_say_what_a_does(app):
    _, controller = app
    state = _state(controller)
    _empty(state, [2])
    _guess(state, [3, 4])
    _show(controller, 3)
    panel = controller.review.panel
    tip = tooltip_html(panel.current_segment(), state.images)
    assert "Auto&#x27;s guess on 2 of 3 frames (dashed): Accept (A) stores the one on screen" in tip
    assert "Not a bubble? Shift+drag on the chart &gt; Mark as no bubble" in tip
    assert panel._detail.text().endswith("A accepts the guess shown; 1 more guess to look at")


# -- the hard case: one long row (lic_hard 05: 229 frames, 213 guesses) ----------------------

def test_one_press_on_a_229_frame_row_promotes_exactly_one_frame(gui, tmp_path):
    window, controller = _app(gui, tmp_path, 231)
    state = _state(controller)
    bare = list(range(16, 231, 14))  # 16 of the row's frames have no guess, as on lic_hard 05
    _guess(state, [f for f in range(2, 231) if f not in bare])
    _empty(state, bare)
    _show(controller, 5)
    segment = controller.review.panel.current_segment()
    assert segment.count == 229
    assert "Auto&#x27;s guess on 213 of 229 frames" in tooltip_html(segment, state.images)
    assert controller.review.panel._detail.text().endswith(
        "A accepts the guess shown; 212 more guesses to look at")
    _accept(controller)
    assert _promoted(state) == [5] and int(state.fit_locked.sum()) == 1
    assert _state(controller).image_no == 6
    assert controller.review.panel.current_segment().count == 228


# -- other rows with guesses (the tracker's "Lost track") --------------------------------------

def test_lost_track_rows_follow_the_same_rule(app):
    _, controller = app
    state = _state(controller)
    _empty(state, [2, 4, 6])
    _guess(state, [3, 5])
    controller.review.set_extra([ReviewItem(f, ("lost",), 100) for f in range(2, 7)])
    _show(controller, 3)
    assert controller.review.panel.current_segment().label == "Lost track"
    _accept(controller)
    assert _promoted(state) == [3] and _state(controller).image_no == 5
    _show(controller, 2)
    _accept(controller)  # the row without its guess: 2, 4, 6
    assert np.flatnonzero(state.fit_locked).tolist() == [2, 3, 4, 6]
    assert shown_suggestion(state, 5) is not None
