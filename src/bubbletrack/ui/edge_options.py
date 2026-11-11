"""Options for the Edge detection engine (shown only in Edge mode)."""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QPushButton, QWidget,
)

_DEFINITIONS = (
    ("Steep edge (50%)", "steep"),
    ("Outer extent", "outer"),
)
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


class EdgeOptionsBox(QGroupBox):
    """Edge definition, bubble polarity, reference frame and seed point."""

    options_changed = pyqtSignal()
    use_current_reference_clicked = pyqtSignal()
    auto_reference_clicked = pyqtSignal()
    pick_seed_clicked = pyqtSignal()
    clear_seed_clicked = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Edge engine", parent)
        grid = QGridLayout(self)
        grid.setSpacing(6)

        self._definition = _combo(_DEFINITIONS)
        self._definition.setToolTip(
            "Where on the dark-to-bright transition the bubble wall is placed.\n"
            "Steep edge (50%): halfway up the steep part of the edge. Independent\n"
            "of focus and optics; recommended and comparable across setups.\n"
            "Outer extent: where brightness has recovered to ~90% of the\n"
            "background. Moves outward with blur/defocus."
        )
        self._definition.currentIndexChanged.connect(self.options_changed)
        grid.addWidget(QLabel("Edge:"), 0, 0)
        grid.addWidget(self._definition, 0, 1, 1, 2)

        self._polarity = _combo(_POLARITIES)
        self._polarity.setToolTip(
            "Dark bubble: the usual backlit shadowgraph.\n"
            "Bright bubble: the bubble appears brighter than its surroundings."
        )
        self._polarity.currentIndexChanged.connect(self.options_changed)
        grid.addWidget(QLabel("Bubble:"), 1, 0)
        grid.addWidget(self._polarity, 1, 1, 1, 2)

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
        grid.addWidget(QLabel("Reference:"), 2, 0)
        grid.addWidget(self._reference_label, 2, 1)
        grid.addLayout(ref_buttons, 3, 1, 1, 2)

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
        grid.addWidget(QLabel("Seed:"), 4, 0)
        grid.addWidget(self._seed_label, 4, 1)
        grid.addLayout(seed_buttons, 5, 1, 1, 2)

    # -- values ---------------------------------------------------------- #

    def edge_definition(self) -> str:
        return self._definition.currentData()

    def polarity(self) -> str:
        return self._polarity.currentData()

    def set_options(self, edge_definition: str, polarity: str) -> None:
        """Set values without emitting options_changed."""
        _select(self._definition, edge_definition)
        _select(self._polarity, polarity)

    def set_reference_text(self, text: str) -> None:
        self._reference_label.setText(text)

    def set_seed_text(self, text: str) -> None:
        self._seed_label.setText(text)
