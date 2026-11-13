"""Plain words in the lines users read first: the run summary and the header.

"Done: 253 of 256 frames measured · 15 to check in Review", green unless
something failed; routine notes go to the line's tooltip.
"""

from __future__ import annotations

import time

from bubbletrack.controller.run_summary import OK_COLOUR, WARN_COLOUR, RunOutcome, summary_line
from bubbletrack.ui.header_bar import HeaderBar
from bubbletrack.ui.wording import confidence_text, plural


def test_done_is_green_with_the_review_count():
    text, colour, details = summary_line(RunOutcome(
        total=256, measured=253, to_check=15, details=("tracking re-measured 1 frame",)))
    assert text == "Done: 253 of 256 frames measured · 15 to check in Review"
    assert colour == OK_COLOUR and details == "tracking re-measured 1 frame"


def test_nothing_to_check_says_nothing_about_review():
    text, colour, _ = summary_line(RunOutcome(total=71, measured=71))
    assert text == "Done: 71 of 71 frames measured" and colour == OK_COLOUR


def test_problems_are_named_and_amber():
    text, colour, _ = summary_line(RunOutcome(
        total=10, measured=8, problems=("1 frame could not be read",)))
    assert text == "Done: 8 of 10 frames measured — 1 frame could not be read"
    assert colour == WARN_COLOUR


def test_stopped_and_ml_assist():
    assert summary_line(RunOutcome(total=256, measured=30, stopped=True, frames_done=34,
                                   to_check=4))[:2] == (
        "Stopped after 34 frames · 4 to check in Review", WARN_COLOUR)
    assert summary_line(RunOutcome(total=1, measured=1, ml=True))[0] == (
        "Done with ML assist: 1 of 1 frame measured")


def test_plurals_and_confidence_wording():
    assert plural(1, "frame") == "1 frame" and plural(2, "frame") == "2 frames"
    assert plural(3, "fix", "fixes") == "3 fixes"
    assert confidence_text(0.871) == "confidence 0.87" and confidence_text(float("nan")) == ""


def test_header_keeps_details_for_the_same_message(qapp):
    header = HeaderBar()
    header.set_status("Done: 5 of 5 frames measured", OK_COLOUR, "tracking re-measured 1 frame")
    assert header.status_details() == "tracking re-measured 1 frame"
    header.set_status("Frame 3", OK_COLOUR)
    assert header.status_details() == ""
    header.set_status("Done: 5 of 5 frames measured", OK_COLOUR)  # put back (e.g. by Review)
    assert header.status_details() == "tracking re-measured 1 frame"


def test_fit_summary_in_the_app(qapp, gui, make_sequence):
    window, controller = gui
    controller.file_ctrl.on_folder_selected(str(make_sequence("A", 6)))
    controller.auto_ctrl.on_auto_fit()
    deadline = time.monotonic() + 120
    while controller.auto_ctrl.is_busy() and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    text, colour = window.header.status()
    assert text.startswith("Done: ") and " of 6 frames measured" in text
    assert "(s)" not in text and "unreliable" not in text
    assert colour == OK_COLOUR
