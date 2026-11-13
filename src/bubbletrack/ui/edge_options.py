"""The Auto (reference frame) box: reference, seed, ML assist; Advanced options.

The wall is always placed at the 50% crossing of the steep edge; the former
"Outer extent" choice is no longer offered (``edge_definition`` stays
"steep").  The bubble colour (dark by default) is in Advanced.
"""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QPushButton, QWidget,
)

from bubbletrack.ui.widgets import CollapsibleSection

STEEP_EDGE = "steep"  # the only wall position the app offers
_POLARITIES = (
    ("Dark bubble", "dark"),
    ("Bright bubble", "bright"),
)


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


ML_ASSIST_TIP = (
    "ML assist helps Auto on hard recordings (blood, speckle, low contrast).\n"
    "Teach it with a few frames: the frames you verified count automatically\n"
    "(manual fits, circle fixes, accepted frames); a few brush strokes help more.\n"
    "Apply re-measures every frame you have not confirmed; ML's own outlines\n"
    "always go to the Review queue.")
ML_SWITCH_TIP = ("On: Fit measures with ML assist.\n"
                 "Off: plain Auto (re-fit to return to Auto's results).")


class EdgeOptionsBox(QGroupBox):
    """Reference frame, seed point, ML assist; Advanced: bubble colour, tracking."""

    options_changed = pyqtSignal()
    use_current_reference_clicked = pyqtSignal()
    auto_reference_clicked = pyqtSignal()
    pick_seed_clicked = pyqtSignal()
    clear_seed_clicked = pyqtSignal()
    ml_assist_clicked = pyqtSignal()  # "ML assist..." (teaching mode)
    ml_switch_toggled = pyqtSignal(bool)  # ML assist on / off for this recording

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Auto (reference frame)", parent)
        grid = QGridLayout(self)
        grid.setSpacing(6)

        self._reference_label = QLabel("Auto")
        self._reference_label.setToolTip(
            "A frame without a bubble, used to remove static background\n"
            "(illumination, debris, speckle). Auto picks one of the first frames."
        )
        ref_buttons = QHBoxLayout()
        use_current = QPushButton("Use current")
        use_current.setObjectName("secondaryBtn")
        use_current.setToolTip("Use the displayed frame as the bubble-free reference.")
        use_current.clicked.connect(self.use_current_reference_clicked)
        auto = QPushButton("Auto")
        auto.setObjectName("secondaryBtn")
        auto.clicked.connect(self.auto_reference_clicked)
        ref_buttons.addWidget(use_current)
        ref_buttons.addWidget(auto)
        grid.addWidget(QLabel("Reference:"), 0, 0)
        grid.addWidget(self._reference_label, 0, 1)
        grid.addLayout(ref_buttons, 1, 1, 1, 2)

        self._seed_label = QLabel("None")
        self._seed_label.setToolTip(
            "Optional: click once on the bubble so the right object is chosen\n"
            "when several dark objects are present."
        )
        seed_buttons = QHBoxLayout()
        pick = QPushButton("Click bubble")
        pick.setObjectName("secondaryBtn")
        pick.clicked.connect(self.pick_seed_clicked)
        clear = QPushButton("Clear")
        clear.setObjectName("secondaryBtn")
        clear.clicked.connect(self.clear_seed_clicked)
        seed_buttons.addWidget(pick)
        seed_buttons.addWidget(clear)
        grid.addWidget(QLabel("Seed:"), 2, 0)
        grid.addWidget(self._seed_label, 2, 1)
        grid.addLayout(seed_buttons, 3, 1, 1, 2)
        self._build_ml_assist(grid, 4)
        self.advanced = self._build_advanced()
        grid.addWidget(self.advanced, 6, 0, 1, 3)

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

    def _build_ml_assist(self, grid: QGridLayout, row: int) -> None:
        """'ML assist:' state line, the switch and the 'ML assist...' button."""
        self.ml_state_label = QLabel("Off")
        self.ml_state_label.setToolTip(ML_ASSIST_TIP)
        self.ml_button = QPushButton("ML assist…")
        self.ml_button.setObjectName("secondaryBtn")
        self.ml_button.setToolTip(ML_ASSIST_TIP)
        self.ml_button.clicked.connect(self.ml_assist_clicked)
        self.ml_switch = QCheckBox("On")
        self.ml_switch.setToolTip(ML_SWITCH_TIP)
        self.ml_switch.setEnabled(False)
        self.ml_switch.toggled.connect(self.ml_switch_toggled)
        buttons = QHBoxLayout()
        buttons.addWidget(self.ml_button)
        buttons.addWidget(self.ml_switch)
        grid.addWidget(QLabel("ML assist:"), row, 0)
        grid.addWidget(self.ml_state_label, row, 1, 1, 2)
        grid.addLayout(buttons, row + 1, 1, 1, 2)

    def set_ml_state(self, text: str, on: bool, can_switch: bool, colour: str = "") -> None:
        """Show the state line and the switch without emitting ml_switch_toggled."""
        self.ml_state_label.setText(text)
        self.ml_state_label.setStyleSheet(f"color:{colour};" if colour else "")
        self.ml_switch.blockSignals(True)
        self.ml_switch.setChecked(on)
        self.ml_switch.blockSignals(False)
        self.ml_switch.setEnabled(can_switch)

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

    def set_reference_text(self, text: str) -> None:
        self._reference_label.setText(text)

    def set_seed_text(self, text: str) -> None:
        self._seed_label.setText(text)
