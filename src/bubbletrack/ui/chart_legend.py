"""Bottom row of the R(t) chart: marker shapes, the score ramp, the chart's buttons.

One compact line, so the plot keeps the chart's height.  The keys may be
clipped on narrow windows rather than widen them; only the buttons set a
minimum width.
"""

from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QSizePolicy, QWidget

from bubbletrack.ui import chart_style as cs

BUTTON_HEIGHT = 24
_KEY_STYLE = "background: transparent; color:#a1a1aa; font-size:11px;"
# The app's segment buttons are 30 px tall (min-height + padding); these are compact.
_BUTTON_STYLE = "QPushButton#segmentBtn { min-height: 18px; padding: 1px 12px; }"


@dataclass(frozen=True)
class LegendRow:
    """The row's layout and the widgets the chart updates or connects."""

    layout: QHBoxLayout
    shapes: QLabel
    score_name: QLabel
    select_btn: QPushButton
    reset_btn: QPushButton


def _chart_button(text: str, tip: str) -> QPushButton:
    btn = QPushButton(text)
    btn.setObjectName("segmentBtn")
    btn.setProperty("segment", "single")
    btn.setStyleSheet(_BUTTON_STYLE)
    btn.setFixedHeight(BUTTON_HEIGHT)
    btn.setToolTip(tip)
    return btn


def build_legend_row() -> LegendRow:
    """Shape key, colour ramp, then "Select range" and "Reset Zoom"."""
    row = QHBoxLayout()
    row.setContentsMargins(8, 0, 0, 0)
    row.setSpacing(10)
    shapes = QLabel(cs.shape_legend_html())
    score_name = QLabel("fit quality: poor")
    bar = QLabel()
    bar.setFixedSize(90, 9)
    bar.setStyleSheet(f"background: {cs.ramp_gradient_css()}; border-radius: 4px;")
    good = QLabel("good")
    for lbl in (shapes, score_name, good):
        lbl.setStyleSheet(_KEY_STYLE)
    shapes.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
    keys = QWidget()
    keys.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
    keys_row = QHBoxLayout(keys)
    keys_row.setContentsMargins(0, 0, 0, 0)
    keys_row.setSpacing(10)
    keys_row.addWidget(shapes, 1)
    for widget in (score_name, bar, good):
        keys_row.addWidget(widget)
    row.addWidget(keys, 1)
    row.addSpacing(12)
    select_btn = _chart_button(
        "Select range",
        "Drag across the chart to select frames (Shift+drag works any time).\n"
        "Right-click the selection to refit, lock or mark those frames as no bubble.",
    )
    select_btn.setCheckable(True)
    reset_btn = _chart_button(
        "Reset Zoom",
        "Show all frames (double-click the chart does the same).\n"
        "Wheel: zoom frames around the cursor; drag: pan; wheel on the radius axis: zoom radius.",
    )
    row.addWidget(select_btn)
    row.addWidget(reset_btn)
    return LegendRow(row, shapes, score_name, select_btn, reset_btn)
