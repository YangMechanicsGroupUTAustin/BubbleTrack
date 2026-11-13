"""Manual mode tab: two fix tools, "Drag a circle" (default) and "Click edge points"."""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QButtonGroup, QHBoxLayout, QLabel, QProgressBar, QPushButton, QSpinBox,
    QStackedWidget, QVBoxLayout, QWidget,
)

from bubbletrack.ui.circle_tool import CircleToolPage
from bubbletrack.ui.widgets import InfoBox

TOOL_CIRCLE, TOOL_POINTS = "circle", "points"
_TOOLS = (  # (name, label, tooltip when available)
    (TOOL_CIRCLE, "Drag a circle",
     "Press on the bubble centre and drag to its wall: Auto measures the exact wall"),
    (TOOL_POINTS, "Click edge points", "Click at least 3 points on the bubble edge"),
)
CIRCLE_UNAVAILABLE_TIP = (
    "Drag a circle uses the Auto (reference frame) engine.\n"
    "Choose \"Auto (reference frame)\" as the detection mode in Pre-tune to use it."
)


class ManualTab(QWidget):
    """Fix single frames by hand: draw a circle, or click edge points."""

    select_clicked = pyqtSignal()
    done_clicked = pyqtSignal()
    clear_clicked = pyqtSignal()
    frame_selected = pyqtSignal(int)  # 0-indexed frame number
    tool_changed = pyqtSignal(str)  # TOOL_CIRCLE | TOOL_POINTS (the tool in use)
    use_circle_clicked = pyqtSignal()
    cancel_circle_clicked = pyqtSignal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        layout.addLayout(self._build_frame_row())
        layout.addLayout(self._build_tool_selector())

        self.circle_page = CircleToolPage()
        self.circle_page.use_circle_clicked.connect(self.use_circle_clicked)
        self.circle_page.cancel_clicked.connect(self.cancel_circle_clicked)
        self._pages = QStackedWidget()
        self._pages.addWidget(self.circle_page)
        self._pages.addWidget(self._build_points_page())
        layout.addWidget(self._pages)
        layout.addStretch()

        self._preferred = TOOL_CIRCLE  # the user's choice
        self._circle_available = True
        self._tool = ""
        self._apply_tool()

    # -- construction ------------------------------------------------------ #

    def _build_frame_row(self) -> QHBoxLayout:
        frame_row = QHBoxLayout()
        frame_row.addWidget(QLabel("Frame:"))
        self._frame_spin = QSpinBox()
        self._frame_spin.setMinimum(1)
        self._frame_spin.setValue(1)
        self._frame_spin.setMinimumWidth(70)
        self._frame_spin.valueChanged.connect(
            lambda v: self.frame_selected.emit(v - 1)
        )
        frame_row.addWidget(self._frame_spin)
        self._frame_total_label = QLabel("/ —")
        frame_row.addWidget(self._frame_total_label)
        frame_row.addStretch()
        return frame_row

    def _build_tool_selector(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(0)
        self._tool_group = QButtonGroup(self)
        self._tool_group.setExclusive(True)
        self._tool_buttons: dict[str, QPushButton] = {}
        for i, (name, label, tip) in enumerate(_TOOLS):
            btn = QPushButton(label)
            btn.setObjectName("segmentBtn")
            btn.setCheckable(True)
            btn.setToolTip(tip)
            btn.setProperty("segment", "first" if i == 0 else "last")
            btn.clicked.connect(lambda _checked, n=name: self.set_tool(n))
            self._tool_group.addButton(btn)
            self._tool_buttons[name] = btn
            row.addWidget(btn, 1)
        return row

    def _build_points_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        layout.addWidget(InfoBox(
            "Click on the bubble edge to select points.\n"
            "At least 3 points are needed for circle fitting.\n"
            "Press Done or Enter when finished.",
            colour="rgba(99, 102, 241, 0.15)", border="rgba(99, 102, 241, 0.20)",
        ))

        counter_row = QHBoxLayout()
        self._count_label = QLabel("Points: 0")
        self._count_label.setStyleSheet("font-weight:bold; font-size:14px;")
        counter_row.addWidget(self._count_label)
        counter_row.addStretch()
        self._status = QLabel("Need ≥ 3")
        self._status.setObjectName("dimText")
        counter_row.addWidget(self._status)
        layout.addLayout(counter_row)

        self._progress = QProgressBar()
        self._progress.setRange(0, 3)
        self._progress.setValue(0)
        self._progress.setTextVisible(False)
        layout.addWidget(self._progress)

        btn_row = QHBoxLayout()
        self._select_btn = QPushButton("Select Points")
        self._select_btn.clicked.connect(self.select_clicked)
        btn_row.addWidget(self._select_btn)
        self._done_btn = QPushButton("Done")
        self._done_btn.setEnabled(False)
        self._done_btn.clicked.connect(self.done_clicked)
        btn_row.addWidget(self._done_btn)
        self._clear_btn = QPushButton("Clear")
        self._clear_btn.setObjectName("secondaryBtn")
        self._clear_btn.clicked.connect(self.clear_clicked)
        btn_row.addWidget(self._clear_btn)
        layout.addLayout(btn_row)
        return page

    # -- tools ------------------------------------------------------------- #

    def tool(self) -> str:
        """The tool in use (points while the circle tool is unavailable)."""
        return self._tool

    def set_tool(self, name: str) -> None:
        """Choose a tool; emits tool_changed if the tool in use changes."""
        if name not in self._tool_buttons:
            raise ValueError(f"unknown manual tool {name!r}")
        self._preferred = name
        self._apply_tool()

    def set_circle_available(self, available: bool) -> None:
        """Drag a circle needs the Auto engine; otherwise it is disabled."""
        self._circle_available = bool(available)
        btn = self._tool_buttons[TOOL_CIRCLE]
        btn.setEnabled(self._circle_available)
        btn.setToolTip(_TOOLS[0][2] if self._circle_available else CIRCLE_UNAVAILABLE_TIP)
        self._apply_tool()

    def circle_available(self) -> bool:
        return self._circle_available

    def _apply_tool(self) -> None:
        tool = self._preferred
        if tool == TOOL_CIRCLE and not self._circle_available:
            tool = TOOL_POINTS
        self._tool_buttons[tool].setChecked(True)
        self._pages.setCurrentIndex(0 if tool == TOOL_CIRCLE else 1)
        if tool != self._tool:
            self._tool = tool
            self.tool_changed.emit(tool)

    # -- circle prompt ----------------------------------------------------- #

    def show_circle_prompt(self, text: str, offer_manual: bool = True) -> None:
        self.circle_page.show_prompt(text, offer_manual)

    def hide_circle_prompt(self) -> None:
        self.circle_page.hide_prompt()

    # -- point counter ----------------------------------------------------- #

    def set_point_count(self, n: int):
        self._count_label.setText(f"Points: {n}")
        self._progress.setValue(min(n, 3))
        self._done_btn.setEnabled(n >= 3)
        if n >= 3:
            self._status.setText("Ready to fit")
            self._status.setStyleSheet("color:#10b981; font-size:11px;")
        else:
            self._status.setText(f"Need {3 - n} more")
            self._status.setStyleSheet("color:#71717a; font-size:11px;")

    def reset(self):
        self.set_point_count(0)

    def set_frame_range(self, total: int):
        """Set the max frame number without emitting a frame change.

        Shrinking the maximum clamps the value, which would otherwise emit
        frame_selected for a frame of the previous source.
        """
        self._frame_spin.blockSignals(True)
        self._frame_spin.setMaximum(total)
        self._frame_spin.blockSignals(False)
        self._frame_total_label.setText(f"/ {total}")

    def set_frame_value(self, idx: int):
        """Set frame spinbox to 1-indexed value without emitting signal."""
        self._frame_spin.blockSignals(True)
        self._frame_spin.setValue(idx + 1)
        self._frame_spin.blockSignals(False)
