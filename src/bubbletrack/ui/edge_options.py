"""The Auto (reference frame) box: reference, bubble position, ML assist; Advanced.

One compact line each: a label, the current value and a small action button
(a second button only while it has something to undo).  The ML assist line
appears once it is relevant (after a fit, or when examples exist).

The wall is always placed at the 50% crossing of the steep edge; the former
"Outer extent" choice is no longer offered (``edge_definition`` stays
"steep").  The bubble colour (dark by default) is in Advanced.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QPushButton, QWidget,
)

from bubbletrack.ui.widgets import CollapsibleSection

STEEP_EDGE = "steep"  # the only wall position the app offers
_POLARITIES = (
    ("Dark bubble", "dark"),
    ("Bright bubble", "bright"),
)

REFERENCE_TIP = (
    "A frame without a bubble, used to remove the static background\n"
    "(illumination, debris, speckle). Auto picks the last normal frame\n"
    "before the flash or the bubble.")
USE_FRAME_TIP = ("Use the frame on screen as the reference.\n"
                 "It must show no bubble: you are warned if it seems to hold one.")
AUTO_REFERENCE_TIP = "Let Auto choose the reference frame again."
POSITION_TIP = (
    "Optional: click once on the bubble so the right object is measured\n"
    "when several dark objects are present. Auto finds it by itself otherwise.")
PICK_TIP = "Then click once on the bubble in the image."
CLEAR_POSITION_TIP = "Forget the clicked position: Auto finds the bubble by itself."
ML_ASSIST_TIP = (
    "ML assist helps Auto on hard recordings (blood, speckle, low contrast).\n"
    "Teach…: confirm two different frames where the outline is right (frames\n"
    "you verified in Review count too), then Train and apply: it measures the\n"
    "frames to check; ML's own outlines are listed in Review for you to check.")
ML_SWITCH_TIP = ("On: Fit measures with ML assist.\n"
                 "Off: plain Auto (fit again to return to Auto's results).")
ML_SWITCH_LOCKED_TIP = ("Not available yet: teach ML assist first (Teach…):\n"
                        "confirm two different frames where the outline is right.")
AUTO_POSITION_TEXT = "auto · click to set"
NAME_WIDTH = 86  # the names column, so the values line up
UNDO_BUTTON_WIDTH = 26


def _combo(items: tuple[tuple[str, str], ...]) -> QComboBox:
    box = QComboBox()
    for label, value in items:
        box.addItem(label, value)
    return box


def _select(box: QComboBox, value: str) -> None:
    index = box.findData(value)
    if index >= 0:
        box.blockSignals(True)
        box.setCurrentIndex(index)
        box.blockSignals(False)


def _button(text: str, tip: str, slot) -> QPushButton:
    button = QPushButton(text)
    button.setObjectName("secondaryBtn")
    button.setToolTip(tip)
    button.clicked.connect(slot)
    return button


def _undo_button(name: str, tip: str, slot) -> QPushButton:
    """A small '×' that undoes a choice; *name* is what it is called (tooltip, a11y)."""
    button = _button("×", f"{name}: {tip[:1].lower()}{tip[1:]}", slot)
    button.setAccessibleName(name)
    button.setFixedWidth(UNDO_BUTTON_WIDTH)
    return button


def _name(text: str, tip: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("dimText")
    label.setToolTip(tip)
    return label


def _value(tip: str) -> QLabel:
    label = QLabel()
    label.setToolTip(tip)
    label.setWordWrap(True)  # a long fallback text wraps instead of widening the panel
    return label


class EdgeOptionsBox(QGroupBox):
    """Reference frame, bubble position, ML assist; Advanced: bubble colour, tracking."""

    options_changed = pyqtSignal()
    use_current_reference_clicked = pyqtSignal()
    auto_reference_clicked = pyqtSignal()
    pick_seed_clicked = pyqtSignal()
    clear_seed_clicked = pyqtSignal()
    ml_assist_clicked = pyqtSignal()  # "Teach…" (teaching mode)
    ml_switch_toggled = pyqtSignal(bool)  # ML assist on / off for this recording

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Auto (reference frame)", parent)
        grid = QGridLayout(self)
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(6)
        grid.setColumnMinimumWidth(0, NAME_WIDTH)
        grid.setColumnStretch(1, 1)

        self._reference_label = _value(REFERENCE_TIP)
        self.use_frame_button = _button("Use this frame", USE_FRAME_TIP,
                                        self.use_current_reference_clicked)
        self.auto_reference_button = _undo_button("Auto", AUTO_REFERENCE_TIP,
                                                  self.auto_reference_clicked)
        self._add_line(grid, 0, _name("Reference", REFERENCE_TIP), self._reference_label,
                       self.use_frame_button, self.auto_reference_button)

        self._seed_label = _value(POSITION_TIP)
        self.pick_button = _button("Click bubble", PICK_TIP, self.pick_seed_clicked)
        self.clear_seed_button = _undo_button("Clear", CLEAR_POSITION_TIP,
                                              self.clear_seed_clicked)
        self._add_line(grid, 1, _name("Bubble position", POSITION_TIP), self._seed_label,
                       self.pick_button, self.clear_seed_button)

        self._ml_widgets = self._build_ml_assist()
        self._add_line(grid, 2, *self._ml_widgets)
        self.advanced = self._build_advanced()
        grid.addWidget(self.advanced, 3, 0, 1, 2)
        self.set_reference_text("Auto", user_set=False)
        self.set_seed_text(AUTO_POSITION_TEXT, user_set=False)

    @staticmethod
    def _add_line(grid: QGridLayout, row: int, name: QLabel, value: QWidget,
                  *buttons: QPushButton) -> None:
        """Name | value and its buttons: each line's value gets the room its
        own buttons leave (a shared buttons column would squeeze every value)."""
        grid.addWidget(name, row, 0, Qt.AlignmentFlag.AlignVCenter)
        line = QHBoxLayout()
        line.setSpacing(4)
        line.addWidget(value, 1)
        for button in buttons:
            line.addWidget(button)
        grid.addLayout(line, row, 1)

    def _build_advanced(self) -> CollapsibleSection:
        """Collapsed: options that rarely need changing."""
        advanced = CollapsibleSection("Advanced", collapsed=True)
        self._polarity = _combo(_POLARITIES)
        self._polarity.setToolTip(
            "Dark bubble: the usual backlit shadowgraph.\n"
            "Bright bubble: the bubble appears brighter than its surroundings."
        )
        self._polarity.currentIndexChanged.connect(self.options_changed)
        polarity = QWidget()
        row = QHBoxLayout(polarity)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(QLabel("Bubble:"))
        row.addWidget(self._polarity, 1)
        advanced.add_widget(polarity)
        self._track_check = QCheckBox("Track across frames after Fit")
        self._track_check.setChecked(True)
        self._track_check.setToolTip(
            "Follow the bubble from frame to frame once every frame is fitted:\n"
            "a frame whose result does not fit the track is measured again near\n"
            "the predicted circle, and doubtful frames join the Review queue.\n"
            "Manual and locked frames are kept and anchor the track.\n"
            "Turn off to keep pure single-frame results.")
        advanced.add_widget(self._track_check)
        return advanced

    def _build_ml_assist(self) -> tuple[QLabel, QCheckBox, QPushButton]:
        """'ML assist  [ ] Off · 2 examples  [Teach…]', hidden until relevant."""
        self.ml_switch = QCheckBox("Off")
        self.ml_switch.setToolTip(ML_SWITCH_LOCKED_TIP)
        self.ml_switch.setEnabled(False)
        self.ml_switch.toggled.connect(self.ml_switch_toggled)
        self.ml_button = _button("Teach…", ML_ASSIST_TIP, self.ml_assist_clicked)
        widgets = (_name("ML assist", ML_ASSIST_TIP), self.ml_switch, self.ml_button)
        for widget in widgets:
            widget.hide()
        return widgets

    def ml_line_visible(self) -> bool:
        """The ML assist line is shown (relevant: after a fit, or with examples)."""
        return not self.ml_switch.isHidden()

    def set_ml_state(self, text: str, on: bool, can_switch: bool, colour: str = "",
                     *, relevant: bool = True) -> None:
        """Show the state and the switch without emitting ml_switch_toggled.

        *text* is the state line ('Off', 'On · 2 examples'); the line is shown
        only when *relevant* (after a fit, or when examples exist).
        """
        self.ml_switch.blockSignals(True)
        self.ml_switch.setChecked(on)
        self.ml_switch.blockSignals(False)
        self.ml_switch.setText(text)
        self.ml_switch.setStyleSheet(f"color:{colour};" if colour else "")
        self.ml_switch.setEnabled(can_switch)
        self.ml_switch.setToolTip(ML_SWITCH_TIP if can_switch else ML_SWITCH_LOCKED_TIP)
        for widget in self._ml_widgets:
            widget.setVisible(bool(relevant))

    def set_ml_available(self, available: bool, tip: str = "") -> None:
        """Without scikit-learn the button is disabled and says why."""
        self.ml_button.setEnabled(available)
        self.ml_button.setToolTip(ML_ASSIST_TIP if available else tip)

    # -- values ---------------------------------------------------------- #

    def edge_definition(self) -> str:
        """Where the wall is placed: always the 50% crossing of the steep edge."""
        return STEEP_EDGE

    def polarity(self) -> str:
        return self._polarity.currentData()

    def tracking_enabled(self) -> bool:
        return self._track_check.isChecked()

    def set_options(self, edge_definition: str, polarity: str) -> None:
        """Show *polarity* without emitting options_changed.

        *edge_definition* is accepted for older sessions and presets but not
        offered: the wall is always the steep edge (see :meth:`edge_definition`).
        """
        _select(self._polarity, polarity)

    def set_reference_text(self, text: str, *, user_set: bool = False) -> None:
        """The reference line; 'Auto' (back to automatic) only after 'Use this frame'."""
        self._reference_label.setText(text)
        self.auto_reference_button.setVisible(user_set)

    def reference_text(self) -> str:
        return self._reference_label.text()

    def set_seed_text(self, text: str, *, user_set: bool = False) -> None:
        """The bubble-position line; 'Clear' only while a position is set."""
        self._seed_label.setText(text)
        self.clear_seed_button.setVisible(user_set)

    def seed_text(self) -> str:
        return self._seed_label.text()
