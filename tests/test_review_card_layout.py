"""The Review card's buttons never move under the pointer.

The owner clicks Accept many times in a row: whatever the card, the row or
the feedback line shows, Accept, Fix and No bubble (and the row's buttons)
must stay exactly where they are at a given dock size.
"""

from __future__ import annotations

import numpy as np
import pytest
from PyQt6.QtCore import QRect, Qt
from PyQt6.QtGui import QFontMetrics

from _review_dock import QSS, button_rects, settle, size_dock
from bubbletrack.controller.review_queue import ReviewItem
from bubbletrack.controller.review_segments import build_segments
from bubbletrack.model.provenance import FitSource
from bubbletrack.ui.lines_label import ELLIPSIS, fit_lines
from bubbletrack.ui.review_card import FEEDBACK_LINES, ActionButton, CardAdvice, FeedbackLine

N_FRAMES = 40
ROW = range(8, 20)  # a row of 12 frames
SINGLES = (24, 28, 32, 36)  # rows of one frame
OUTSIDE = 2  # a frame in no row
ADVICE = {
    1: "Auto is not sure here.",
    4: ("The image edge cuts it and the wall you can see is not round, so the circle may "
        "be off: accept it if it follows the wall. If speckle hides the wall, fix it with a "
        "circle, or mark No bubble when the frame is empty and nothing is there at all."),
}
LONG = ("Frame 45: Auto's guess accepted (R = 123.4 px) and locked. That row is done. "
        "The next row opens on its first frame, and Ctrl+Z undoes all of this.")
DOCK_SIZES = ((1180, 720, 280), (1500, 960, 430))  # window width, height; dock width


@pytest.fixture
def styled(qapp):
    """The app's style sheet: its two button looks have different borders."""
    qapp.setStyleSheet(QSS.read_text(encoding="utf-8"))
    yield
    qapp.setStyleSheet("")


@pytest.fixture
def review(styled, gui, make_sequence):
    window, controller = gui
    controller.file_ctrl.on_folder_selected(str(make_sequence("A", N_FRAMES)))
    state = controller._get_state()
    state.radius[:] = np.linspace(30.0, 40.0, N_FRAMES)
    state.circle_fit_par[:] = [[125.0, 200.0]] * N_FRAMES
    state.fit_source[:] = FitSource.AUTO
    state.fit_confidence[:] = 0.9
    rc = controller.review
    rc.refresh()
    return window, controller, rc


class CardDriver:
    """Puts the real Review dock into one state of the matrix at a time."""

    def __init__(self, qapp, window, rc, images) -> None:
        self.qapp, self.window, self.rc, self.images = qapp, window, rc, images
        self.advice, self.accept_n = CardAdvice(ADVICE[1], "accept"), len(ROW)
        rc.panel.set_advisor(lambda _frame, _item: self.advice)
        rc.panel.set_row_counter(lambda _segment: self.accept_n)

    def size(self, width: int, height: int, dock_width: int) -> None:
        size_dock(self.qapp, self.window, self.rc, width, height, dock_width)

    def settle(self) -> None:
        settle(self.qapp)

    def frame(self, *, lines=1, action="accept", row="multi", can_accept=True,
              accept_n=len(ROW), singles=1) -> None:
        self.advice = CardAdvice(ADVICE[lines], action, can_accept=can_accept)
        self.accept_n = accept_n
        items = [ReviewItem(f, ("low_confidence",), 50, 0.2) for f in ROW]
        items += [ReviewItem(f, ("low_confidence",), 50, 0.2) for f in SINGLES[:singles]]
        panel = self.rc.panel
        panel.set_segments(build_segments(items), self.images, fitted=True)
        panel.select_frame({"multi": ROW[2], "single": SINGLES[0], "none": OUTSIDE}[row],
                           keep_row=False)
        panel.refresh_card()
        self.settle()

    def rects(self) -> dict[str, QRect]:
        return button_rects(self.rc)


def _states(drive: CardDriver):
    """(name, set-up) for every state that used to move the buttons."""
    panel = drive.rc.panel
    yield "advice 1 line", lambda: drive.frame(lines=1)
    yield "advice 4 lines", lambda: drive.frame(lines=4)
    yield "no row", lambda: drive.frame(row="none")
    yield "row of one frame", lambda: drive.frame(row="single")
    yield "row, nothing for Accept all", lambda: drive.frame(accept_n=0)
    yield "row before the bubble", lambda: drive.frame(action="no_bubble_all")
    for action in ("fix", "no_bubble", "", "accept_all"):
        yield f"filled: {action or 'none'}", lambda a=action: drive.frame(action=a)
    yield "empty frame", lambda: drive.frame(action="fix", can_accept=False)
    yield "empty frame, no row", lambda: drive.frame(action="fix", can_accept=False, row="none")
    for name, show in (("feedback + Undo", lambda: panel.show_feedback("Frame 9 accepted.")),
                       ("long feedback + Undo", lambda: panel.show_feedback(LONG)),
                       ("feedback, no Undo", lambda: panel.show_feedback("Done.", undo=False)),
                       ("hint", lambda: panel.show_hint(LONG)),
                       ("no feedback", panel.clear_feedback)):
        yield name, lambda s=show: (drive.frame(), s(), drive.settle())
    for singles in (0, len(SINGLES)):
        yield f"list of {1 + singles} rows", lambda k=singles: drive.frame(singles=k)


def test_the_buttons_stay_put_in_every_state(qapp, review):
    window, controller, rc = review
    drive = CardDriver(qapp, window, rc, controller._get_state().images)
    by_size = []
    for width, height, dock_width in DOCK_SIZES:
        drive.size(width, height, dock_width)
        seen: dict[str, dict[str, QRect]] = {}
        for name, set_up in _states(drive):
            set_up()
            seen[name] = drive.rects()
        reference = seen["advice 1 line"]
        for name, rects in seen.items():
            assert rects == reference, f"{name} at dock width {rc.dock.width()}"
        by_size.append(reference)
    assert by_size[0] != by_size[1]  # resizing the dock may move them; content may not


def test_hidden_accept_keeps_its_slot_and_fix_never_takes_it(qapp, review):
    window, controller, rc = review
    drive = CardDriver(qapp, window, rc, controller._get_state().images)
    drive.size(*DOCK_SIZES[0])
    drive.frame()
    before = drive.rects()
    drive.frame(action="fix", can_accept=False)
    buttons = rc.panel.card.buttons
    assert buttons["accept"].isHidden()  # nothing to accept here
    assert drive.rects()["fix"] == before["fix"] != before["accept"]


def test_the_buttons_step_aside_during_a_fix_and_come_back_in_place(qapp, review):
    window, controller, rc = review
    drive = CardDriver(qapp, window, rc, controller._get_state().images)
    drive.size(*DOCK_SIZES[0])
    drive.frame()
    before = drive.rects()
    card = rc.panel.card
    card.set_fixing(True)
    drive.settle()
    assert all(b.isHidden() for b in card.buttons.values()) and card.more.isHidden()
    card.set_fixing(False)
    drive.settle()
    assert drive.rects() == before


def test_both_looks_of_a_button_are_the_same_size(qapp, styled):
    filled, outlined = ActionButton("accept"), ActionButton("fix")
    filled.set_primary(True)
    for button in (filled, outlined):
        button.setText("Accept this frame")
    assert filled.sizeHint().height() == outlined.sizeHint().height()


def test_feedback_line_is_one_height_and_cuts_a_long_text(qapp, styled):
    line = FeedbackLine()
    line.resize(240, 80)
    heights = set()
    for show in (lambda: line.show_text("Frame 9 accepted."), lambda: line.show_text(LONG),
                 lambda: line.show_hint(LONG), lambda: line.show_text("x", undo=False),
                 line.clear):
        show()
        qapp.processEvents()
        heights.add(line.sizeHint().height())
    assert len(heights) == 1
    assert line.sizePolicy().retainSizeWhenHidden()  # empty, it still holds its room
    line.show_text(LONG)
    qapp.processEvents()
    shown = line._text.text()
    assert shown.endswith(ELLIPSIS) and line._text.toolTip() == f"✓ {LONG}"
    assert line.text() == f"✓ {LONG}"  # the whole text for whoever asks
    line.close()


def test_fit_lines_keeps_a_short_text_and_cuts_a_long_one_to_its_lines(qapp):
    from PyQt6.QtWidgets import QLabel

    metrics = QFontMetrics(QLabel().font())
    assert fit_lines("Short.", metrics, 200, FEEDBACK_LINES) == "Short."
    cut = fit_lines(LONG * 3, metrics, 200, FEEDBACK_LINES)
    assert cut.endswith(ELLIPSIS) and LONG.startswith(cut[:-1].rstrip()[:40])
    wrap = Qt.TextFlag.TextWordWrap.value
    most = "\n".join("X" * FEEDBACK_LINES)
    lines = metrics.boundingRect(QRect(0, 0, 200, 10_000), wrap, most).height()
    assert metrics.boundingRect(QRect(0, 0, 200, 10_000), wrap, cut).height() <= lines
