"""First-time-user review fixes, through the real window offscreen.

C1/C2: frames before the bubble are one grey row whose action is "No bubble
on all"; an empty frame is "No bubble" only when it looks like the reference
frame outside the bubble's frames.  M4: the header counts checks apart from
information, and the selected row stays in view.  Polish 9/10: skipped
frames are not "checked"; the row dots are explained.
"""

from __future__ import annotations

import numpy as np
import pytest

from bubbletrack.controller.key_frames import to_check_frames
from bubbletrack.model import ml_assist
from bubbletrack.model.provenance import FitSource
from bubbletrack.model.state import update_state
from bubbletrack.model.suggestion import make_suggestion, shown_suggestion
from conftest import write_sequence

SHAPE, CENTRE, R = (80, 120), (40.0, 60.0), 18.0
N = 14
BUBBLE = range(3, 10)  # frames 4-10 show the bubble; 1-3 and 11-14 are empty


@pytest.fixture
def app(gui, tmp_path):
    """Frames 1-3 empty (pre-laser), 4-10 a bubble, 11-14 empty again; Auto
    measured 5-10 only (frame 4: no result although the bubble is there)."""
    window, controller = gui
    radii = [R if i in BUBBLE else 0.0 for i in range(N)]
    folder = write_sequence(tmp_path / "seq", radii, shape=SHAPE, center=CENTRE)
    controller.file_ctrl.on_folder_selected(str(folder))
    controller._state = update_state(controller._get_state(), detection_mode="edge",
                                     reference_frame=0)
    state = controller._get_state()
    for i in range(N):
        measured = i in BUBBLE and i != 3
        state.radius[i] = R if measured else -1.0
        state.circle_fit_par[i] = list(CENTRE)
        state.fit_source[i] = FitSource.AUTO if measured else FitSource.NONE
        state.fit_confidence[i] = 0.9 if measured else 0.05
    # Auto's weak guesses on the pre-laser frames (debris)
    for i in (0, 2):
        state.fit_suggestion[i] = make_suggestion(4.0, 10.0, 10.0, np.zeros((0, 2)), 0.05)
    controller.review.refresh()
    return window, controller, controller.review


def _row(rc, category):
    return next(s for s in rc.panel.segments() if s.category == category)


class TestBeforeTheBubble:
    def test_one_grey_row_not_a_red_false_alarm(self, app):
        _, controller, rc = app
        before = _row(rc, "before")
        assert before.frames == (0, 1, 2) and before.tier == "low"
        assert rc.panel.texts()[-1] == "Frames 1–3 · Before the bubble · 3 frames"
        assert rc.panel.segments()[0].category != "before"  # not the top row
        state = controller._get_state()
        assert all(shown_suggestion(state, f) is None for f in (0, 2))  # weak: not offered

    def test_the_card_recommends_no_bubble_on_the_whole_row(self, app):
        window, controller, rc = app
        controller.file_ctrl.on_frame_changed(0)  # Auto's weak guess is not offered
        card = rc.panel.card
        assert card.advice.text().startswith("Before the bubble")
        assert card.recommended() == "no_bubble_all"
        assert card.buttons["no_bubble_all"].text() == "No bubble on all 3"
        assert card.buttons["accept"].isHidden() and card.buttons["accept_all"].isHidden()
        assert "guess" not in window.header.status_text().lower()  # no invitation to accept
        rc.perform("no_bubble_all")
        state = controller._get_state()
        assert state.fit_locked[:3].all() and (state.radius[:3] < 0).all()
        assert "before" not in [s.category for s in rc.panel.segments()]

    def test_the_bubbles_first_frame_without_result_is_a_real_check(self, app):
        _, controller, rc = app
        controller.file_ctrl.on_frame_changed(3)  # the bubble is there, Auto missed it
        segment = rc.panel.current_segment()
        assert segment.category == "no_result" and segment.tier == "high"
        assert rc.panel.card.recommended() == "fix"
        assert rc.panel.card.advice.text() == (
            "No result here. Is there a bubble? Fix it (F), or No bubble (N) if the frame "
            "is empty.")


class TestEmptyFramesAfterTheBubble:
    def test_no_bubble_only_where_the_frame_looks_like_the_reference(self, app):
        """C1: never 'probably no bubble' on a frame that shows one."""
        _, controller, rc = app
        controller.file_ctrl.on_frame_changed(12)  # after the bubble, empty
        assert rc.panel.card.recommended() == "no_bubble"
        assert "looks like the reference frame" in rc.panel.card.advice.text()
        state = controller._get_state()
        state.radius[5] = -1.0  # a frame inside the bubble's frames
        state.fit_source[5] = FitSource.NONE
        rc.refresh()
        controller.file_ctrl.on_frame_changed(5)
        assert rc.panel.card.recommended() == "fix"

    def test_a_soft_bubble_after_the_last_result_means_fix(self, app):
        _, controller, rc = app
        state = controller._get_state()
        state.radius[7:10] = -1.0  # Auto lost the bubble's last frames (they show it)
        state.fit_source[7:10] = FitSource.NONE
        state.fit_confidence[7:10] = 0.05
        rc.refresh()
        controller.file_ctrl.on_frame_changed(9)
        assert rc.panel.card.recommended() == "fix"


class TestHeaderAndList:
    def test_checks_and_information_are_counted_apart(self, app):
        _, controller, rc = app
        rc.panel.reset_progress()
        # the red row (frame 4) and the after-bubble frames 11-14 are checks;
        # the before row is for information
        assert rc.panel.summary() == "5 frames to check · 3 for information"
        assert not rc.panel.tier_key.isHidden()
        assert "likely wrong" in rc.panel.tier_key.text()
        assert "for information" in rc.panel.tier_key.text()

    def test_status_line_and_key_frames_count_what_review_counts(self, app):
        _, controller, rc = app
        # "to check" means red and amber rows everywhere: the status line after
        # a fit says 5 like Review's header, and the key-frame strip does not
        # call the grey before-the-bubble frames "to check".
        assert controller.auto_ctrl.review_count() == 5
        firsts = to_check_frames(rc.panel.segments())
        assert 3 in firsts and 0 not in firsts

    def test_skipped_frames_are_not_counted_as_checked(self, app):
        _, controller, rc = app
        rc.panel.reset_progress()
        controller.file_ctrl.on_frame_changed(3)
        rc.perform("skip")
        controller.file_ctrl.on_frame_changed(10)
        rc.perform("no_bubble_all")
        assert rc.panel.summary() == "4 checked · 1 skipped · 3 for information"
        controller.file_ctrl.on_frame_changed(0)
        rc.perform("no_bubble_all")
        assert rc.panel.done_page.text() == (
            "All frames to check are done (1 skipped).\nNext: export R(t).")

    def test_the_selected_row_stays_in_view(self, qapp, gui, tmp_path):
        window, controller = gui
        folder = write_sequence(tmp_path / "long", [R] * 60, shape=SHAPE, center=CENTRE)
        controller.file_ctrl.on_folder_selected(str(folder))
        window.resize(1280, 800)
        window.show()
        qapp.processEvents()
        state = controller._get_state()
        state.radius[:] = R
        state.fit_source[:] = FitSource.AUTO
        state.fit_confidence[:] = 0.9
        state.fit_confidence[::3] = 0.1  # 20 separate rows
        rc = controller.review
        rc.refresh()
        assert len(rc.panel.segments()) == 20
        controller.file_ctrl.on_frame_changed(57)  # the last row, selected silently
        for _ in range(3):
            qapp.processEvents()
        lv = rc.panel._list
        rect = lv.visualItemRect(lv.currentItem())
        assert lv.currentRow() == 19
        assert 0 <= rect.top() and rect.bottom() <= lv.viewport().height()


class TestMlAssistLeads:
    def test_when_most_of_the_bubble_needs_a_check(self, app, monkeypatch):
        window, controller, rc = app
        ml = controller.ml
        monkeypatch.setattr(ml, "available", True)
        ml.hint.hide()
        state = controller._get_state()
        state.fit_confidence[4:10] = 0.2  # every measured frame doubtful
        rc.refresh()
        finished = {"stopped": False, "edge": True, "frames": tuple(range(N))}
        controller.bus.emit("run_finished", finished)
        assert ml.hint.isHidden()  # 6 frames: fewer than ML assist's own minimum
        monkeypatch.setattr(ml_assist, "HINT_MIN_FRAMES", 5)
        controller.bus.emit("run_finished", finished)
        assert not ml.hint.isHidden()
        assert ml.hint.text().startswith("100% of the bubble's frames need a check.")

    def test_not_when_few_frames_need_a_check(self, app, monkeypatch):
        _, controller, rc = app
        ml = controller.ml
        monkeypatch.setattr(ml, "available", True)
        ml.hint.hide()
        controller.bus.emit("run_finished", {"stopped": False, "edge": True,
                                             "frames": tuple(range(N))})
        assert ml.hint.isHidden()
