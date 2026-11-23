"""After Review is done, the frame card stays under the done page.

Once a fit exists and nothing is left to check, the done page keeps its one
line and the card of the frame on screen stays below it: no button filled,
no row actions, the buttons where they sat during Review.  Fix (the button,
as F) still fixes a frame Auto measured without a flag.  This partly
reverses A5 ("the done page stands alone"): the owner's choice.
"""

from __future__ import annotations

import numpy as np
import pytest

from _review_dock import QSS, button_rects, settle, size_dock
from bubbletrack.model.provenance import FitSource
from bubbletrack.model.state import update_state

N_FRAMES = 8


@pytest.fixture
def styled(qapp):
    qapp.setStyleSheet(QSS.read_text(encoding="utf-8"))
    yield
    qapp.setStyleSheet("")


@pytest.fixture
def app(gui, make_sequence):
    """Eight frames fitted, none flagged (intensity mode: Fix clicks edge points)."""
    window, controller = gui
    controller.file_ctrl.on_folder_selected(str(make_sequence("A", N_FRAMES)))
    controller._state = update_state(controller._get_state(), detection_mode="intensity")
    state = controller._get_state()
    state.radius[:] = np.linspace(30.0, 40.0, N_FRAMES)
    state.circle_fit_par[:] = [[125.0, 200.0]] * N_FRAMES
    state.fit_source[:] = FitSource.AUTO
    state.fit_confidence[:] = 0.9
    controller.review.refresh()
    return window, controller, controller.review


def _points(controller, n=8, radius=33.0):
    for a in np.linspace(0, 2 * np.pi, n, endpoint=False):
        controller.manual_ctrl.on_point_clicked(200 + radius * np.cos(a), 125 + radius * np.sin(a))


def _done(rc) -> bool:
    panel = rc.panel
    return panel.count() == 0 and panel._stack.currentWidget() is panel.done_page


def test_the_card_stays_under_the_done_page_with_nothing_filled(app):
    _, controller, rc = app
    controller.file_ctrl.on_frame_changed(4)
    panel, card = rc.panel, rc.panel.card
    assert _done(rc) and not panel.done_page.isHidden()
    assert panel._summary.isHidden()  # the done page's line is the one summary
    assert card.isVisibleTo(panel)
    assert card.title.text().startswith("Frame 5")
    assert card.advice.text() == "Not flagged: nothing to do here."
    assert card.recommended() is None
    assert all(not card.buttons[a].isHidden() for a in ("accept", "fix", "no_bubble"))
    assert card.buttons["accept_all"].isHidden() and card.buttons["no_bubble_all"].isHidden()
    assert card.more.isHidden()  # no row: no row actions


def test_an_accepted_frame_says_so_and_offers_fix(app):
    _, controller, rc = app
    controller._get_state().fit_locked[3] = True
    rc.refresh()
    controller.file_ctrl.on_frame_changed(3)
    card = rc.panel.card
    assert _done(rc) and card.advice.text() == "Accepted: nothing to do here."
    assert card.recommended() is None and card.buttons["accept"].isHidden()
    assert not card.buttons["fix"].isHidden()


def test_nothing_is_filled_on_the_done_page_whatever_the_advice(app):
    """A skipped or unflagged empty frame may get advice; done, nothing is filled."""
    _, controller, rc = app
    controller.file_ctrl.on_frame_changed(4)
    rc.panel.set_advisor(lambda _f, _i: rc._advice(_f, _i)._replace(action="fix"))
    assert _done(rc) and rc.panel.card.recommended() is None


def test_before_a_fit_the_done_page_stands_alone(gui, make_sequence):
    _, controller = gui
    controller.file_ctrl.on_folder_selected(str(make_sequence("B", 4)))
    controller.file_ctrl.on_frame_changed(1)
    rc = controller.review
    assert rc.panel._stack.currentWidget() is rc.panel.done_page
    assert rc.panel.card.isHidden()


def test_fix_on_an_unlisted_frame_stores_it_and_review_stays_done(app):
    _, controller, rc = app
    controller.file_ctrl.on_frame_changed(4)
    rc.panel.card.buttons["fix"].click()  # the button, as F
    assert rc.fix.active and rc.fix.frame == 4
    assert rc.panel.card.buttons["fix"].isHidden()  # the strip is where to act now
    _points(controller)
    rc.fix.strip._done_btn.click()
    state = controller._get_state()
    assert not rc.fix.active
    assert state.fit_source[4] == FitSource.MANUAL and state.fit_locked[4]
    assert state.radius[4] == pytest.approx(33.0, abs=0.5)
    assert rc.panel.feedback.text().startswith("✓ Frame 5 fixed: R = 33.0 px")
    assert _done(rc) and rc.panel.card.isVisibleTo(rc.panel)
    assert rc.panel.card.advice.text() == "Fixed by you: nothing to do here."


def test_the_buttons_sit_where_they_sat_during_review(qapp, styled, app):
    window, controller, rc = app
    size_dock(qapp, window, rc, 1180, 720, 300)
    state = controller._get_state()
    state.fit_confidence[[2, 3]] = 0.1  # one row to check
    rc.refresh()
    rc.panel.step(1)
    settle(qapp)
    assert not _done(rc) and rc.panel.current_segment() is not None
    during = button_rects(rc)
    state.fit_confidence[[2, 3]] = 0.9  # checked: nothing left
    rc.refresh()
    controller.file_ctrl.on_frame_changed(2)
    settle(qapp)
    assert _done(rc) and rc.panel.card.isVisible()
    assert button_rects(rc) == during
