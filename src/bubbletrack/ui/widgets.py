"""Reusable UI primitives: SliderInput, ToggleSwitch, CollapsibleSection, InfoBox."""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal, QSize, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import QPainter, QColor
from PyQt6.QtWidgets import (
    QCheckBox, QDoubleSpinBox, QHBoxLayout, QLabel, QSlider, QSpinBox,
    QVBoxLayout, QWidget, QPushButton, QFrame, QSizePolicy,
)


def _to_ticks(value: float, scale: int) -> int:
    """Convert a value to integer slider ticks, rounding (not truncating).

    Truncation turned a saved threshold of 0.58 (57.999... ticks) into 57.
    """
    return int(round(value * scale))


# ------------------------------------------------------------------ #
# SliderInput  (label + spinbox + slider, synchronised)
# ------------------------------------------------------------------ #

class SliderInput(QWidget):
    """Horizontal: ``Label  [SpinBox]  ====o====``."""

    value_changed = pyqtSignal(float)

    def __init__(
        self,
        label: str,
        min_val: float = 0,
        max_val: float = 100,
        default: float = 50,
        step: float = 1,
        decimals: int = 0,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self._decimals = decimals
        self._scale = 10 ** decimals  # slider works in ints

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)

        self._label = QLabel(label)
        self._label.setFixedWidth(80)
        layout.addWidget(self._label)

        if decimals > 0:
            self._spin = QDoubleSpinBox()
            self._spin.setDecimals(decimals)
        else:
            self._spin = QSpinBox()
        self._spin.setRange(_to_ticks(min_val, self._scale), _to_ticks(max_val, self._scale))
        self._spin.setSingleStep(step if decimals > 0 else int(step))
        self._spin.setValue(_to_ticks(default, self._scale))
        self._spin.setFixedWidth(72)
        layout.addWidget(self._spin)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(_to_ticks(min_val, self._scale), _to_ticks(max_val, self._scale))
        self._slider.setSingleStep(max(1, _to_ticks(step, self._scale)))
        self._slider.setPageStep(max(1, _to_ticks(step, self._scale)))
        self._slider.setValue(_to_ticks(default, self._scale))
        layout.addWidget(self._slider, 1)

        # Sync
        self._slider.valueChanged.connect(self._slider_moved)
        self._spin.valueChanged.connect(self._spin_changed)

    # -- internal sync --
    def _slider_moved(self, v: int):
        self._spin.blockSignals(True)
        self._spin.setValue(v)
        self._spin.blockSignals(False)
        self.value_changed.emit(v / self._scale)

    def _spin_changed(self, v):
        iv = _to_ticks(float(v), self._scale) if self._decimals > 0 else int(v)
        self._slider.blockSignals(True)
        self._slider.setValue(iv)
        self._slider.blockSignals(False)
        self.value_changed.emit(float(v) if self._decimals > 0 else v)

    # -- public API --
    def value(self) -> float:
        return self._spin.value() / self._scale if self._decimals > 0 else self._spin.value()

    def set_value(self, v: float):
        self._spin.setValue(_to_ticks(v, self._scale))

    def set_range(self, lo: float, hi: float):
        self._spin.setRange(_to_ticks(lo, self._scale), _to_ticks(hi, self._scale))
        self._slider.setRange(_to_ticks(lo, self._scale), _to_ticks(hi, self._scale))


# ------------------------------------------------------------------ #
# ToggleSliderInput  (checkbox + label + spinbox + slider)
# ------------------------------------------------------------------ #

class ToggleSliderInput(QWidget):
    """``[☐] [Label] [SpinBox] ====o====`` — emits 0 when checkbox is off."""

    value_changed = pyqtSignal(float)

    def __init__(
        self,
        label: str,
        min_val: float = 0,
        max_val: float = 100,
        default: float = 50,
        step: float = 1,
        decimals: int = 0,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self._decimals = decimals
        self._scale = 10 ** decimals
        self._default = default

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)

        self._check = QCheckBox()
        self._check.setFixedWidth(20)
        layout.addWidget(self._check)

        self._label = QLabel(label)
        self._label.setFixedWidth(68)
        layout.addWidget(self._label)

        if decimals > 0:
            self._spin = QDoubleSpinBox()
            self._spin.setDecimals(decimals)
            self._spin.setSingleStep(step)
        else:
            self._spin = QSpinBox()
            self._spin.setSingleStep(int(step))
        self._spin.setRange(_to_ticks(min_val, self._scale), _to_ticks(max_val, self._scale))
        self._spin.setValue(_to_ticks(default, self._scale))
        self._spin.setFixedWidth(72)
        self._spin.setEnabled(False)
        layout.addWidget(self._spin)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(_to_ticks(min_val, self._scale), _to_ticks(max_val, self._scale))
        self._slider.setValue(_to_ticks(default, self._scale))
        self._slider.setEnabled(False)
        layout.addWidget(self._slider, 1)

        # Sync
        self._slider.valueChanged.connect(self._slider_moved)
        self._spin.valueChanged.connect(self._spin_changed)
        self._check.toggled.connect(self._on_toggled)

    def _slider_moved(self, v: int):
        self._spin.blockSignals(True)
        self._spin.setValue(v)
        self._spin.blockSignals(False)
        self.value_changed.emit(v / self._scale)

    def _spin_changed(self, v):
        iv = _to_ticks(float(v), self._scale) if self._decimals > 0 else int(v)
        self._slider.blockSignals(True)
        self._slider.setValue(iv)
        self._slider.blockSignals(False)
        self.value_changed.emit(float(v) if self._decimals > 0 else v)

    def _on_toggled(self, checked: bool):
        self._spin.setEnabled(checked)
        self._slider.setEnabled(checked)
        self.value_changed.emit(self.value())

    def is_enabled(self) -> bool:
        return self._check.isChecked()

    def value(self) -> float:
        if not self._check.isChecked():
            return 0.0
        raw = self._spin.value()
        return raw / self._scale if self._decimals > 0 else float(raw)

    def set_value(self, v: float, enabled: bool = True):
        """Set value and checkbox state without emitting signals."""
        self._check.blockSignals(True)
        self._spin.blockSignals(True)
        self._slider.blockSignals(True)
        self._check.setChecked(enabled)
        self._spin.setEnabled(enabled)
        self._slider.setEnabled(enabled)
        iv = _to_ticks(v, self._scale)
        self._spin.setValue(iv)
        self._slider.setValue(iv)
        self._check.blockSignals(False)
        self._spin.blockSignals(False)
        self._slider.blockSignals(False)


# ------------------------------------------------------------------ #
# ToggleSwitch  (40×20 sliding toggle)
# ------------------------------------------------------------------ #

class ToggleSwitch(QCheckBox):
    """Custom painted on/off toggle."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setFixedSize(QSize(40, 20))

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        if self.isChecked():
            p.setBrush(QColor("#6366f1"))
        else:
            p.setBrush(QColor("#52525b"))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(0, 0, 40, 20, 10, 10)
        # Thumb
        p.setBrush(QColor("white"))
        cx = 26 if self.isChecked() else 10
        p.drawEllipse(cx - 7, 3, 14, 14)
        p.end()

    def hitButton(self, pos):
        return self.rect().contains(pos)


# ------------------------------------------------------------------ #
# CollapsibleSection
# ------------------------------------------------------------------ #

class CollapsibleSection(QWidget):
    """A section with a clickable header that shows/hides its content."""

    def __init__(self, title: str, parent: QWidget | None = None, collapsed: bool = False):
        super().__init__(parent)
        self._title = title
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        arrow = "▸" if collapsed else "▾"
        self._toggle = QPushButton(f"  {arrow} {title}")
        self._toggle.setObjectName("collapseToggle")
        self._toggle.setCheckable(True)
        self._toggle.setChecked(not collapsed)
        self._toggle.setToolTip("Click to expand/collapse")
        self._toggle.clicked.connect(self._on_toggle)
        layout.addWidget(self._toggle)

        self._content = QWidget()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(0, 4, 0, 4)
        layout.addWidget(self._content)

        if collapsed:
            self._content.hide()

    def content_layout(self) -> QVBoxLayout:
        return self._content_layout

    def add_widget(self, w: QWidget):
        self._content_layout.addWidget(w)

    def _on_toggle(self, checked: bool):
        self._content.setVisible(checked)
        arrow = "▾" if checked else "▸"
        self._toggle.setText(f"  {arrow} {self._title}")


# ------------------------------------------------------------------ #
# InfoBox  (coloured hint box)
# ------------------------------------------------------------------ #

class InfoBox(QFrame):
    """Coloured information / instruction box."""

    def __init__(self, text: str, colour: str = "rgba(99, 102, 241, 0.15)",
                 border: str = "rgba(99, 102, 241, 0.20)",
                 parent: QWidget | None = None):
        super().__init__(parent)
        self.setStyleSheet(
            f"background:{colour}; border:1px solid {border}; border-radius:8px; padding:8px;"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        lbl = QLabel(text)
        lbl.setWordWrap(True)
        lbl.setStyleSheet("border:none; color:#e4e4e7;")
        layout.addWidget(lbl)
