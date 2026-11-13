"""The user's review decisions, and the question before another recording replaces them.

A review is what the user decided frame by frame: accepted (locked) frames,
fixes by hand, "no bubble" marks and ML assist's painted strokes.  Plain fit
results are not a review (Fit makes them again).  The recovery autosave keeps
only one recording, so loading another one asks first when the decisions are
not in a session file.
"""

from __future__ import annotations

import os

import numpy as np
from PyQt6.QtWidgets import QMessageBox

from bubbletrack.model.provenance import is_user_confirmed
from bubbletrack.model.state import AppState
from bubbletrack.model.teach import MLExamples
from bubbletrack.ui.wording import plural

SAVE, DISCARD, CANCEL = "save", "discard", "cancel"


def _locked(state: AppState) -> np.ndarray:
    n = 0 if state.radius is None else len(state.radius)
    return np.zeros(n, bool) if state.fit_locked is None else np.asarray(state.fit_locked, bool)


def _by_hand(state: AppState, i: int) -> bool:
    return state.fit_source is not None and is_user_confirmed(state.fit_source[i])


def review_signature(state: AppState) -> tuple:
    """What the user decided, comparable between two moments (empty: nothing)."""
    if state.radius is None:
        return ()
    locked = _locked(state)
    decided = tuple(
        (i, round(float(state.radius[i]), 3), bool(locked[i]), _by_hand(state, i))
        for i in range(len(state.radius)) if locked[i] or _by_hand(state, i))
    strokes = tuple((f, s.bubble, s.background) for f, s in _painted(state))
    return decided + ((("strokes", strokes),) if strokes else ())


def _painted(state: AppState) -> list:
    """(frame, strokes) of the frames with ML assist strokes, by frame."""
    examples = state.ml_examples
    if not isinstance(examples, MLExamples):
        return []
    return sorted(((f, s) for f, s in examples.strokes.items() if not s.is_empty),
                  key=lambda item: item[0])


def review_counts(state: AppState) -> dict[str, int]:
    """{'fixed', 'accepted', 'no_bubble', 'painted'}: frames per kind of decision."""
    counts = dict.fromkeys(("fixed", "accepted", "no_bubble", "painted"), 0)
    if state.radius is None:
        return counts
    locked = _locked(state)
    for i in range(len(state.radius)):
        if _by_hand(state, i):
            counts["fixed"] += 1
        elif locked[i]:
            counts["accepted" if state.radius[i] > 0 else "no_bubble"] += 1
    counts["painted"] = len(_painted(state))
    return counts


def describe(counts: dict[str, int]) -> str:
    """'2 frames accepted, 1 fixed by hand and strokes on 1 frame'."""
    parts = []
    if counts["accepted"]:
        parts.append(f"{plural(counts['accepted'], 'frame')} accepted")
    if counts["fixed"]:
        parts.append(f"{plural(counts['fixed'], 'frame')} fixed by hand")
    if counts["no_bubble"]:
        parts.append(f"{plural(counts['no_bubble'], 'frame')} marked “no bubble”")
    if counts["painted"]:
        parts.append(f"ML assist strokes on {plural(counts['painted'], 'frame')}")
    if not parts:
        return ""
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


def recording_name(state: AppState) -> str:
    """The recording as the user knows it: the video file or the folder name."""
    if state.video_path:
        return os.path.basename(state.video_path)
    return os.path.basename(os.path.normpath(state.folder_path)) if state.folder_path else ""


def unsaved_review_dialog(parent, name: str, detail: str) -> tuple[QMessageBox, dict]:
    """The question box and its buttons by answer (Save session… is the default)."""
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Question)
    box.setWindowTitle("Unsaved review")
    box.setText(f"Save the review of {name} first?")
    box.setInformativeText(f"{detail} — not saved in a session file. "
                           "Loading another recording replaces them.")
    buttons = {
        SAVE: box.addButton("Save session…", QMessageBox.ButtonRole.AcceptRole),
        DISCARD: box.addButton("Discard", QMessageBox.ButtonRole.DestructiveRole),
        CANCEL: box.addButton(QMessageBox.StandardButton.Cancel),
    }
    box.setDefaultButton(buttons[SAVE])
    box.setEscapeButton(buttons[CANCEL])
    return box, buttons


def ask_save_review(parent, name: str, detail: str) -> str:
    """'Save the review of <name> first?' -> SAVE, DISCARD or CANCEL."""
    box, buttons = unsaved_review_dialog(parent, name, detail)
    box.exec()
    clicked = box.clickedButton()
    return next((answer for answer, button in buttons.items() if button is clicked), CANCEL)
