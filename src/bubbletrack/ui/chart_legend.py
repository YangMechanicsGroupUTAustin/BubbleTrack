"""Bottom row of the R(t) chart: marker keys, the colour scale, the chart's buttons.

One compact line, so the plot keeps the chart's height.  On a narrow chart
the least needed keys hide whole (their text moves to the row's tooltip)
rather than being cut mid-word; only the buttons set a minimum width.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from PyQt6.QtWidgets import (
    QHBoxLayout, QLabel, QLayout, QPushButton, QSizePolicy, QWidget,
)

from bubbletrack.ui import chart_style as cs

BUTTON_HEIGHT = 24
_KEY_STYLE = "background: transparent; color:#a1a1aa; font-size:11px;"
# The app's segment buttons are 30 px tall (min-height + padding); these are compact.
_BUTTON_STYLE = "QPushButton#segmentBtn { min-height: 18px; padding: 1px 12px; }"
RAMP_WIDTH = 64
# What the marker colours mean: Auto's confidence, or (classic methods) the fit quality.
CONFIDENCE, FIT_QUALITY = "confidence", "fit quality"
_RAMP_TIPS = {
    CONFIDENCE: ("Marker colour: Auto's confidence in the fit, from 0 (red, likely wrong)\n"
                 "to 1 (green). Review lists the frames with low confidence."),
    FIT_QUALITY: ("Marker colour: how well the edge points fit a circle, from 0 (red)\n"
                  "to 1 (green). Classic methods have no confidence of their own."),
}
# Which key hides first when the row is too narrow: the highest number; the
# colour scale goes last.
_DROP_COMPUTED, _DROP_MANUAL, _DROP_RAMP = 3, 2, 1
RAMP_KEY = "colour scale"


def keys_that_fit(widths: Sequence[int], drop_order: Sequence[int], room: int,
                  spacing: int) -> list[bool]:
    """Which keys to show in *room* pixels: the highest *drop_order* hides first,
    0 never hides (it may then not fit)."""
    shown = [True] * len(widths)

    def used() -> int:
        n = sum(shown)
        return sum(w for w, s in zip(widths, shown) if s) + spacing * max(n - 1, 0)

    for k in sorted((i for i, d in enumerate(drop_order) if d > 0), key=lambda i: -drop_order[i]):
        if used() <= room:
            break
        shown[k] = False
    return shown


def ramp_tip(name: str) -> str:
    return _RAMP_TIPS.get(name, f"Marker colour: {name}, from 0 (red) to 1 (green).")


class KeyRow(QWidget):
    """Keys in one line, left to right; keys that do not fit hide whole."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(10)
        # The keys never set a minimum width: they hide instead.
        self._layout.setSizeConstraint(QLayout.SizeConstraint.SetNoConstraint)
        self._keys: list[tuple[QWidget, int, str]] = []
        self._stretches = 0

    def spacing(self) -> int:
        return self._layout.spacing()

    def add_key(self, widget: QWidget, drop_order: int = 0, text: str = "") -> None:
        """Append *widget*; *text* names it in the row's tooltip while it is hidden."""
        self._layout.addWidget(widget)
        self._keys.append((widget, drop_order, text))
        self._fit()

    def add_stretch(self) -> None:
        self._layout.addStretch(1)
        self._stretches += 1

    def keys(self) -> list[QWidget]:
        return [widget for widget, _, _ in self._keys]

    def hidden_texts(self) -> list[str]:
        """Texts of the hidden keys, the first to hide first."""
        hidden = [(d, t) for w, d, t in self._keys if w.isHidden()]
        return [t for _, t in sorted(hidden, key=lambda dt: -dt[0])]

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().resizeEvent(event)
        self._fit()

    def _fit(self) -> None:
        room = self.width() - self.spacing() * self._stretches
        shown = keys_that_fit([w.sizeHint().width() for w, _, _ in self._keys],
                              [d for _, d, _ in self._keys], room, self.spacing())
        for (widget, _, _), show in zip(self._keys, shown):
            if widget.isHidden() == show:
                widget.setVisible(show)
        hidden = self.hidden_texts()
        self.setToolTip("Not shown (narrow window): " + "; ".join(hidden) if hidden else "")


@dataclass(frozen=True)
class LegendRow:
    """The row's layout and the widgets the chart updates or connects."""

    layout: QHBoxLayout
    keys: KeyRow
    score_name: QLabel  # "confidence 0", left of the colour ramp
    score_high: QLabel  # "1", right of it
    select_btn: QPushButton
    reset_btn: QPushButton


def _key_label(html: str, tip: str = "") -> QLabel:
    label = QLabel(html)
    label.setStyleSheet(_KEY_STYLE)
    label.setToolTip(tip)
    return label


def _chart_button(text: str, tip: str) -> QPushButton:
    btn = QPushButton(text)
    btn.setObjectName("segmentBtn")
    btn.setProperty("segment", "single")
    btn.setStyleSheet(_BUTTON_STYLE)
    btn.setFixedHeight(BUTTON_HEIGHT)
    btn.setToolTip(tip)
    return btn


def _ramp() -> tuple[QWidget, QLabel, QLabel]:
    """"confidence 0 [red -> green] 1": the markers' colour scale."""
    box = QWidget()
    row = QHBoxLayout(box)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(6)
    low, high = _key_label(f"{CONFIDENCE} 0", ramp_tip(CONFIDENCE)), _key_label("1")
    bar = QLabel()
    bar.setFixedSize(RAMP_WIDTH, 9)
    bar.setStyleSheet(f"background: {cs.ramp_gradient_css()}; border-radius: 4px;")
    for widget in (low, bar, high):
        row.addWidget(widget)
    return box, low, high


def _marker_keys(keys: KeyRow) -> None:
    (computed_glyph, computed), (manual_glyph, manual) = cs.SHAPE_LEGEND
    glyph = '<span style="color:#e4e4e7;">{}</span> {}'
    keys.add_key(_key_label(glyph.format(computed_glyph, computed),
                            "Round markers: results Auto (or a refit) computed"),
                 _DROP_COMPUTED, f"{computed_glyph} {computed}")
    keys.add_key(_key_label(glyph.format(manual_glyph, manual),
                            "Diamonds: results you fixed, accepted or locked"),
                 _DROP_MANUAL, f"{manual_glyph} {manual}")


def build_legend_row() -> LegendRow:
    """Marker keys, the colour scale, then "Select range" and "Reset Zoom"."""
    row = QHBoxLayout()
    row.setContentsMargins(8, 0, 0, 0)
    row.setSpacing(10)
    keys = KeyRow()
    _marker_keys(keys)
    keys.add_stretch()
    ramp, low, high = _ramp()
    keys.add_key(ramp, _DROP_RAMP, RAMP_KEY)
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
    return LegendRow(row, keys, low, high, select_btn, reset_btn)
