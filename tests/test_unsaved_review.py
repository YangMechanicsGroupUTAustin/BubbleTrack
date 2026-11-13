"""Loading another recording asks before it replaces an unsaved review.

"Save the review of <name> first?" [Save session…] [Discard] [Cancel], only
when the user decided something (accepted, fixed, no bubble, ML strokes) that
is not in a session file.
"""

from __future__ import annotations

import pytest

from bubbletrack.controller import session_controller, unsaved_review
from bubbletrack.controller.unsaved_review import (
    CANCEL, DISCARD, SAVE, describe, review_counts, review_signature, unsaved_review_dialog,
)
from bubbletrack.model.provenance import FitSource


@pytest.fixture
def answers(monkeypatch):
    """Answer the question with ``answers.next`` and record what was asked."""

    class Answers(list):
        next = CANCEL

    record = Answers()

    def ask(_parent, name, detail):
        record.append((name, detail))
        return record.next

    monkeypatch.setattr(session_controller, "ask_save_review", ask)
    return record


def _decide(controller) -> None:
    """Two accepted frames, one fix by hand, one 'no bubble'."""
    state = controller._get_state()
    state.radius[:] = 30.0
    state.fit_locked[[0, 1, 3]] = True
    state.radius[3] = -1.0
    state.fit_source[2] = FitSource.MANUAL


def _load(controller, folder) -> None:
    controller.file_ctrl.on_folder_selected(str(folder))


def test_counts_and_wording(gui, make_sequence):
    _window, controller = gui
    _load(controller, make_sequence("A", 5))
    assert review_signature(controller._get_state()) == ()
    _decide(controller)
    counts = review_counts(controller._get_state())
    assert counts == {"fixed": 1, "accepted": 2, "no_bubble": 1, "painted": 0}
    assert describe(counts) == ("2 frames accepted, 1 frame fixed by hand and "
                                "1 frame marked “no bubble”")


def test_plain_fit_results_are_not_asked_about(gui, make_sequence, answers):
    _window, controller = gui
    _load(controller, make_sequence("A", 5))
    controller._get_state().radius[:] = 30.0
    _load(controller, make_sequence("B", 4))
    assert answers == [] and len(controller._get_state().images) == 4


def test_cancel_keeps_the_recording_and_its_review(gui, make_sequence, answers):
    window, controller = gui
    first = make_sequence("A", 5)
    _load(controller, first)
    _decide(controller)
    _load(controller, make_sequence("B", 4))
    assert answers == [("A", "2 frames accepted, 1 frame fixed by hand and "
                             "1 frame marked “no bubble”")]
    state = controller._get_state()
    assert state.folder_path == str(first) and state.fit_locked[0]
    assert "still open" in window.header.status_text()


def test_discard_loads_the_other_recording(gui, make_sequence, answers):
    _window, controller = gui
    _load(controller, make_sequence("A", 5))
    _decide(controller)
    answers.next = DISCARD
    _load(controller, make_sequence("B", 4))
    assert len(controller._get_state().images) == 4


def test_save_session_then_load(gui, make_sequence, answers, tmp_path):
    _window, controller = gui
    _load(controller, make_sequence("A", 5))
    _decide(controller)
    answers.next = SAVE
    path = tmp_path / "a.brt"
    controller.session_ctrl.choose_save_path = lambda: str(path)
    _load(controller, make_sequence("B", 4))
    assert path.exists() and len(controller._get_state().images) == 4


def test_cancelled_save_dialog_keeps_the_recording(gui, make_sequence, answers):
    _window, controller = gui
    _load(controller, make_sequence("A", 5))
    _decide(controller)
    answers.next = SAVE
    controller.session_ctrl.choose_save_path = lambda: ""
    _load(controller, make_sequence("B", 4))
    assert len(controller._get_state().images) == 5


def test_a_saved_or_loaded_review_is_not_asked_about(gui, make_sequence, answers, tmp_path):
    _window, controller = gui
    _load(controller, make_sequence("A", 5))
    _decide(controller)
    assert controller.save_session_file(str(tmp_path / "a.brt"))
    _load(controller, make_sequence("B", 4))
    assert answers == []
    assert controller.load_session_file(str(tmp_path / "a.brt"))
    assert controller._get_state().fit_locked[0] and not controller.session_ctrl.has_unsaved_review()
    controller._get_state().fit_locked[4] = True  # a new decision after loading
    assert controller.session_ctrl.has_unsaved_review()


def test_loading_a_session_asks_too(gui, make_sequence, answers, tmp_path):
    _window, controller = gui
    _load(controller, make_sequence("A", 5))
    assert controller.save_session_file(str(tmp_path / "a.brt"))
    _decide(controller)
    assert not controller.load_session_file(str(tmp_path / "a.brt"))
    assert len(answers) == 1 and controller._get_state().fit_locked[0]


def test_a_video_is_asked_about_too(gui, make_sequence, answers, monkeypatch):
    _window, controller = gui
    _load(controller, make_sequence("A", 5))
    _decide(controller)

    class Reader:
        total_frames, fps = 3, 0.0

        def close(self):
            Reader.closed = True

    monkeypatch.setattr("bubbletrack.controller.file_controller.open_recording", lambda p: Reader())
    controller.file_ctrl.on_video_selected("clip.avi")
    assert answers and Reader.closed and len(controller._get_state().images) == 5


def test_the_question(qapp):
    box, buttons = unsaved_review_dialog(None, "gelatin", "2 frames accepted")
    assert box.text() == "Save the review of gelatin first?"
    assert [buttons[a].text() for a in (SAVE, DISCARD)] == ["Save session…", "Discard"]
    assert box.defaultButton() is buttons[SAVE] and box.escapeButton() is buttons[CANCEL]


def test_recording_name_is_the_video_or_the_folder(gui, make_sequence):
    _window, controller = gui
    folder = make_sequence("gelatin_run", 3)
    _load(controller, folder)
    assert unsaved_review.recording_name(controller._get_state()) == "gelatin_run"
