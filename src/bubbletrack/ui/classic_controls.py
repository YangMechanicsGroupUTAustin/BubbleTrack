"""Controls of the classic methods (Intensity mask, Gradient rings, Dark region area).

Shown only when Settings > "Classic detection methods" is on, or when the
loaded parameters use a classic method.  These methods will be phased out.
"""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox, QGroupBox, QHBoxLayout, QLabel, QPushButton, QSpinBox,
    QVBoxLayout, QWidget,
)

from bubbletrack.ui.widgets import CollapsibleSection, SliderInput, ToggleSliderInput

EDGE_NAMES = ("Top", "Right", "Down", "Left")

_FILTER_SLIDERS = (  # key, label, min, max, default, tooltip
    ("gaussian_sigma", "Gauss Blur", 1, 20, 3,
     "Gaussian blur applied BEFORE binarization.\n"
     "Smooths the grayscale image to suppress small noise (e.g. tracer beads).\n"
     "Sigma controls blur strength: higher = stronger smoothing."),
    ("clahe_clip", "CLAHE", 1, 40, 2,
     "Contrast Limited Adaptive Histogram Equalization.\n"
     "Applied BEFORE binarization to improve local contrast.\n"
     "Useful when lighting is uneven across the image.\n"
     "Clip limit controls enhancement strength: higher = stronger contrast."),
    ("closing_radius", "Morph Close", 1, 30, 5,
     "Morphological closing applied AFTER binarization.\n"
     "Fills small holes and gaps inside the bubble boundary.\n"
     "Radius controls the structuring element size: higher = fills larger gaps."),
    ("opening_radius", "Morph Open", 1, 20, 3,
     "Morphological opening applied AFTER binarization.\n"
     "Removes thin protrusions and spurs on the bubble edge.\n"
     "Radius controls the structuring element size: higher = removes larger spurs."),
)
_INT_FILTERS = ("closing_radius", "opening_radius")


def _secondary(text: str) -> QPushButton:
    button = QPushButton(text)
    button.setObjectName("secondaryBtn")
    return button


class ClassicControls(QWidget):
    """Threshold, removal, invert, edge flags, filters, Auto Tune and presets."""

    threshold_changed = pyqtSignal(float)
    invert_mask_changed = pyqtSignal(bool)
    removing_factor_changed = pyqtSignal(int)
    edges_changed = pyqtSignal()
    filters_changed = pyqtSignal()
    autotune_clicked = pyqtSignal()
    save_preset_clicked = pyqtSignal()
    load_preset_clicked = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        self._build_mask_controls(layout)
        layout.addWidget(self._build_edge_group())
        layout.addWidget(self._build_filters())
        self._build_autotune(layout)
        self.apply_method("intensity")

    # -- construction ------------------------------------------------------ #

    def _build_mask_controls(self, layout: QVBoxLayout) -> None:
        self._threshold = SliderInput("Threshold", 0, 100, 50, 1, 0)
        self._threshold.setToolTip(
            "Adaptive binarization sensitivity.\n"
            "Higher = more white pixels (bubble interior brighter).\n"
            "Lower = more black pixels (stricter foreground detection).")
        self._threshold.value_changed.connect(lambda v: self.threshold_changed.emit(v / 100.0))
        layout.addWidget(self._threshold)

        self._removing_factor = SliderInput("Remove", 0, 100, 90, 1, 0)
        self._removing_factor.setToolTip(
            "Remove small white speckles from the binary image.\n"
            "Higher = remove larger noise regions.\n"
            "Increase if small debris/beads remain after binarization.")
        self._removing_factor.value_changed.connect(
            lambda v: self.removing_factor_changed.emit(int(v)))
        layout.addWidget(self._removing_factor)

        # A pure binary swap after tuning Threshold and Remove, so detection
        # always sees "interior dark, background bright".
        self._invert_btn = QPushButton("Invert Mask")
        self._invert_btn.setObjectName("invertMaskBtn")
        self._invert_btn.setCheckable(True)
        self._invert_btn.setToolTip(
            "Pure binary swap: white pixels become black, black become white.\n"
            "Use AFTER tuning Threshold and Remove, when the resulting binary\n"
            "has inverted polarity (bubble interior bright, background dark).")
        self._invert_btn.toggled.connect(self.invert_mask_changed)
        layout.addWidget(self._invert_btn)

    def _build_edge_group(self) -> QGroupBox:
        group = QGroupBox("Bubble crosses edge")
        group.setToolTip(
            "Check the edges where the bubble extends beyond the ROI boundary.\n"
            "This helps the algorithm correctly detect partial bubbles\n"
            "by expanding the image in the opposite direction.")
        row = QHBoxLayout(group)
        self._edge_checks: list[QCheckBox] = []
        for name in EDGE_NAMES:
            check = QCheckBox(name)
            check.toggled.connect(lambda _on: self.edges_changed.emit())
            row.addWidget(check)
            self._edge_checks.append(check)
        self._edge_group = group
        return group

    def _build_filters(self) -> CollapsibleSection:
        section = CollapsibleSection("Advanced Filters", collapsed=True)
        self._filter_sliders: dict[str, ToggleSliderInput] = {}
        for key, label, lo, hi, default, tip in _FILTER_SLIDERS:
            slider = ToggleSliderInput(label, lo, hi, default, 1, decimals=0)
            slider.setToolTip(tip)
            slider.value_changed.connect(lambda _v: self.filters_changed.emit())
            section.add_widget(slider)
            self._filter_sliders[key] = slider

        self._gradient_boundary = SliderInput("Gradient Ring", 1, 3, 1, 1, 0)
        self._gradient_boundary.setToolTip(
            "Select which gradient boundary to fit (Gradient rings method).\n"
            "1 = first/inner gradient ring. 2 = second ring. 3 = third/outer ring.")
        self._gradient_boundary.value_changed.connect(lambda _v: self.filters_changed.emit())
        section.add_widget(self._gradient_boundary)

        self._area_regions = QSpinBox()
        self._area_regions.setRange(1, 20)
        self._area_regions.setValue(1)
        self._area_regions.setToolTip("Number of largest dark regions to include in the area.")
        self._area_regions.valueChanged.connect(lambda _v: self.filters_changed.emit())
        self._area_widget = QWidget()
        area_row = QHBoxLayout(self._area_widget)
        area_row.setContentsMargins(0, 0, 0, 0)
        area_row.addWidget(QLabel("Regions"))
        area_row.addWidget(self._area_regions)
        area_row.addStretch()
        section.add_widget(self._area_widget)

        self._temporal_refine = QCheckBox("Use Neighbor Frames")
        self._temporal_refine.setToolTip(
            "Use adjacent frames as references to repair weak gradient fits.\n"
            "Useful when a frame only shows a partial boundary.")
        self._temporal_refine.toggled.connect(lambda _on: self.filters_changed.emit())
        section.add_widget(self._temporal_refine)

        presets = QWidget()
        preset_row = QHBoxLayout(presets)
        preset_row.setContentsMargins(0, 0, 0, 0)
        save_btn, load_btn = _secondary("Save Preset"), _secondary("Load Preset")
        save_btn.clicked.connect(self.save_preset_clicked)
        load_btn.clicked.connect(self.load_preset_clicked)
        preset_row.addWidget(save_btn)
        preset_row.addWidget(load_btn)
        section.add_widget(presets)
        return section

    def _build_autotune(self, layout: QVBoxLayout) -> None:
        self._autotune_btn = _secondary("Auto Tune")
        self._autotune_btn.setToolTip(
            "Automatically find optimal Threshold and Removing Factor\n"
            "by testing multiple combinations on the current frame.")
        self._autotune_btn.clicked.connect(self.autotune_clicked)
        layout.addWidget(self._autotune_btn)
        self._quality_label = QLabel("")
        self._quality_label.setObjectName("qualityLabel")
        self._quality_label.setWordWrap(True)
        self._quality_label.hide()
        layout.addWidget(self._quality_label)

    # -- values ------------------------------------------------------------ #

    def get_threshold(self) -> float:
        return self._threshold.value() / 100.0

    def set_threshold(self, value: float) -> None:
        """Set the threshold without emitting signals."""
        self._threshold.blockSignals(True)
        self._threshold.set_value(value * 100)
        self._threshold.blockSignals(False)

    def get_invert_mask(self) -> bool:
        return self._invert_btn.isChecked()

    def set_invert_mask(self, value: bool) -> None:
        self._invert_btn.blockSignals(True)
        self._invert_btn.setChecked(bool(value))
        self._invert_btn.blockSignals(False)

    def get_removing_factor_slider(self) -> int:
        return int(self._removing_factor.value())

    def set_removing_factor(self, value: int) -> None:
        self._removing_factor.blockSignals(True)
        self._removing_factor.set_value(value)
        self._removing_factor.blockSignals(False)

    def get_edge_flags(self) -> list[bool]:
        return [check.isChecked() for check in self._edge_checks]

    def set_edge_flags(self, flags, *, emit: bool = False) -> None:
        for check, value in zip(self._edge_checks, flags):
            check.blockSignals(not emit)
            check.setChecked(bool(value))
            check.blockSignals(False)

    def filter_values(self) -> dict:
        """Filter parameters (0 = filter off) and the method-specific options."""
        values = {key: slider.value() for key, slider in self._filter_sliders.items()}
        for key in _INT_FILTERS:
            values[key] = int(values[key])
        values.update(
            area_max_regions=self._area_regions.value(),
            gradient_boundary=(self._gradient_boundary.value() - 1) / 2.0,
            temporal_refine=self._temporal_refine.isChecked(),
        )
        return values

    def set_filter_values(self, params: dict) -> None:
        """Set the filter controls from *params* without emitting signals."""
        for key, slider in self._filter_sliders.items():
            if key in params:  # set_value never emits
                value = params[key]
                slider.set_value(value if value > 0 else slider._default, enabled=value > 0)
        if "area_max_regions" in params:
            self._area_regions.blockSignals(True)
            self._area_regions.setValue(int(params["area_max_regions"]))
            self._area_regions.blockSignals(False)
        if "gradient_boundary" in params:
            value = params["gradient_boundary"]
            self._gradient_boundary.blockSignals(True)
            self._gradient_boundary.set_value(value if value > 1 else value * 2 + 1)
            self._gradient_boundary.blockSignals(False)
        if "temporal_refine" in params:
            self._temporal_refine.blockSignals(True)
            self._temporal_refine.setChecked(bool(params["temporal_refine"]))
            self._temporal_refine.blockSignals(False)

    def apply_method(self, mode: str) -> None:
        """Show the options the classic *mode* uses."""
        self._gradient_boundary.setVisible(mode == "gradient")
        self._area_widget.setVisible(mode == "area")
        self._autotune_btn.setEnabled(mode == "intensity")

    # -- Auto Tune and its readout ----------------------------------------- #

    def show_quality(self, n_pts: int, rms: float, score_pct: int) -> None:
        colour = "#10b981" if score_pct >= 70 else "#f59e0b" if score_pct >= 40 else "#ef4444"
        self._quality_label.setText(
            f'<span style="color:#a1a1aa;">Edge pts:</span> {n_pts} &nbsp; '
            f'<span style="color:#a1a1aa;">RMS:</span> {rms:.1f} px &nbsp; '
            f'<span style="color:#a1a1aa;">Confidence:</span> '
            f'<span style="color:{colour};">{score_pct}%</span>')
        self._quality_label.show()

    def show_area_result(self, area: int, count: int) -> None:
        self._quality_label.setText(f"Area: {area} px^2 &nbsp; Connected regions: {count}")
        self._quality_label.show()

    def hide_quality(self) -> None:
        self._quality_label.hide()

    def set_autotune_enabled(self, enabled: bool) -> None:
        self._autotune_btn.setEnabled(enabled)
        self._autotune_btn.setText("Auto Tune" if enabled else "Searching...")
