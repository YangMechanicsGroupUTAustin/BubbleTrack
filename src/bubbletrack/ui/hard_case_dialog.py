"""Export Hard Case: which frames, what is wrong, who reports it.

::

    Save frames the app gets wrong as one .zip ... send it to the maintainer.
    Frames      [ 121 ] to [ 180 ]
                60 frames + 6 neighbours + reference frame 2 · up to 12 MB
                4 fixed by you (labels)
    What is wrong here?
    [                                              ]
    Your name   [ Ann                ]  (optional)
                                     [Cancel] [Save...]
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from PyQt6.QtWidgets import (
    QDialog, QDialogButtonBox, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit,
    QSpinBox, QVBoxLayout, QWidget,
)

INTRO = ("Saves these frames (cropped to the ROI), the settings and your fixes as one "
         ".zip file. Send it to the maintainer any way you like (Box, Teams, USB).")
NOTE_PLACEHOLDER = "e.g. the radius jumps when the bubble touches the wall"
MAX_NOTE_CHARS = 4000


@dataclass(frozen=True)
class HardCaseInput:
    start: int  # 0-based
    end: int  # 0-based, inclusive
    note: str
    reporter: str


class HardCaseDialog(QDialog):
    """Asks for the range, the note and the reporter; ``values()`` after ``exec()``.

    *describe(start, end)* (0-based) returns the summary shown under the
    range, e.g. how many frames, how large the file and how many labels.
    """

    def __init__(self, parent: QWidget | None, *, n_frames: int, start: int, end: int,
                 reporter: str = "", describe: Callable[[int, int], str] = lambda s, e: "") -> None:
        super().__init__(parent)
        self.setWindowTitle("Export Hard Case")
        self.setMinimumWidth(440)
        self._describe = describe
        layout = QVBoxLayout(self)
        intro = QLabel(INTRO)
        intro.setWordWrap(True)
        layout.addWidget(intro)

        form = QFormLayout()
        self._start, self._end = self._spin(n_frames, start), self._spin(n_frames, end)
        row = QHBoxLayout()
        row.addWidget(self._start)
        row.addWidget(QLabel("to"))
        row.addWidget(self._end)
        row.addStretch(1)
        form.addRow("Frames", row)
        self._summary = QLabel()
        self._summary.setObjectName("hardCaseSummary")
        self._summary.setWordWrap(True)
        self._summary.setStyleSheet("color:#a1a1aa; font-size:11px;")
        form.addRow("", self._summary)
        layout.addLayout(form)

        layout.addWidget(QLabel("What is wrong here?"))
        self._note = QPlainTextEdit()
        self._note.setPlaceholderText(NOTE_PLACEHOLDER)
        self._note.setFixedHeight(90)
        layout.addWidget(self._note)

        name_row = QFormLayout()
        self._reporter = QLineEdit(reporter)
        self._reporter.setPlaceholderText("optional")
        self._reporter.setMaxLength(80)
        name_row.addRow("Your name", self._reporter)
        layout.addLayout(name_row)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save
                                   | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Save).setText("Save...")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        for spin in (self._start, self._end):
            spin.valueChanged.connect(self._update_summary)
        self._update_summary()
        self._note.setFocus()

    @staticmethod
    def _spin(n_frames: int, value: int) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(1, max(n_frames, 1))
        spin.setValue(min(max(value + 1, 1), max(n_frames, 1)))  # shown 1-based
        return spin

    def _range(self) -> tuple[int, int]:
        a, b = self._start.value() - 1, self._end.value() - 1
        return min(a, b), max(a, b)

    def _update_summary(self) -> None:
        self._summary.setText(self._describe(*self._range()))

    def set_note(self, text: str) -> None:
        self._note.setPlainText(text)

    def values(self) -> HardCaseInput:
        start, end = self._range()
        note = self._note.toPlainText().strip()[:MAX_NOTE_CHARS]
        return HardCaseInput(start, end, note, " ".join(self._reporter.text().split()))
