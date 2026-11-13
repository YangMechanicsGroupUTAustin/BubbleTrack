"""Measuring settings of the side panel: the Auto box, classic methods on request.

Auto (reference frame) is the method.  The Method selector and the classic
methods' controls appear only when Settings > "Classic detection methods" is
on, or when the loaded parameters use a classic method (so nobody is stuck
with a method whose controls are hidden).
"""

from __future__ import annotations

import json
import logging

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox, QFileDialog, QHBoxLayout, QLabel, QMessageBox, QVBoxLayout, QWidget,
)

from bubbletrack.model.config import DETECTION_MODE_LABELS, DETECTION_MODES
from bubbletrack.ui.classic_controls import ClassicControls
from bubbletrack.ui.edge_options import EdgeOptionsBox
from bubbletrack.ui.widgets import CollapsibleSection

logger = logging.getLogger(__name__)

AUTO_METHOD = "edge"
CLASSIC_METHODS = tuple(mode for mode in DETECTION_MODES if mode != AUTO_METHOD)
_METHOD_ORDER = (AUTO_METHOD, *CLASSIC_METHODS)  # Auto first: the recommended one

METHOD_TIP = (
    "Auto (reference frame): compares each frame with a bubble-free reference\n"
    "frame and places the wall at the 50% brightness crossing; no threshold\n"
    "tuning, reports a confidence per frame. Recommended.\n"
    "Classic methods (to be phased out):\n"
    "  Intensity mask: threshold + circle fit.\n"
    "  Gradient rings: choose a ring when a dark halo surrounds the bubble.\n"
    "  Dark region area: area-equivalent radius for irregular bubbles."
)

# Preset keys that set_filter_params understands (besides threshold etc.).
_FILTER_KEYS = (
    "gaussian_sigma", "clahe_clip", "closing_radius", "opening_radius", "detection_mode",
    "edge_polarity", "bubble_non_round", "area_max_regions", "gradient_boundary", "temporal_refine",
)


def is_classic(mode: str) -> bool:
    return mode in CLASSIC_METHODS


class MeasurePanel(QWidget):
    """The Auto box; with classic methods enabled, a Method selector and their controls."""

    threshold_changed = pyqtSignal(float)
    invert_mask_changed = pyqtSignal(bool)
    removing_factor_changed = pyqtSignal(int)
    edges_changed = pyqtSignal()
    filters_changed = pyqtSignal()  # method, filters or Auto options changed
    autotune_clicked = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        self.method_row = QWidget()
        row = QHBoxLayout(self.method_row)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(QLabel("Method:"))
        self._method_combo = QComboBox()
        for mode in _METHOD_ORDER:
            self._method_combo.addItem(DETECTION_MODE_LABELS[mode], mode)
        self._method_combo.setToolTip(METHOD_TIP)
        self._method_combo.currentIndexChanged.connect(self._on_method_changed)
        row.addWidget(self._method_combo, 1)
        layout.addWidget(self.method_row)

        self.edge_box = EdgeOptionsBox()
        self.edge_box.options_changed.connect(self.filters_changed)
        layout.addWidget(self.edge_box)

        self.classic = ClassicControls()
        self.classic_section = CollapsibleSection("Classic method", collapsed=True)
        self.classic_section.add_widget(self.classic)
        layout.addWidget(self.classic_section)
        self._relay_classic_signals()

        self._classic_allowed = False
        self._apply_visibility()

    def _relay_classic_signals(self) -> None:
        c = self.classic
        c.threshold_changed.connect(self.threshold_changed)
        c.invert_mask_changed.connect(self.invert_mask_changed)
        c.removing_factor_changed.connect(self.removing_factor_changed)
        c.edges_changed.connect(self.edges_changed)
        c.filters_changed.connect(self.filters_changed)
        c.autotune_clicked.connect(self.autotune_clicked)
        c.save_preset_clicked.connect(self._save_preset)
        c.load_preset_clicked.connect(self._load_preset)

    # -- method and visibility -------------------------------------------- #

    def method(self) -> str:
        return self._method_combo.currentData()

    def set_classic_methods_visible(self, visible: bool) -> None:
        """Offer the Method selector (and the classic controls of a classic method)."""
        self._classic_allowed = bool(visible)
        self._apply_visibility()

    def classic_methods_visible(self) -> bool:
        return self._classic_allowed

    def _on_method_changed(self, _index: int) -> None:
        self._apply_visibility()
        self.filters_changed.emit()

    def _apply_visibility(self) -> None:
        mode = self.method()
        self.method_row.setVisible(self._classic_allowed)
        self.edge_box.setVisible(mode == AUTO_METHOD)
        self.classic_section.setVisible(is_classic(mode))
        self.classic.apply_method(mode)

    # -- parameters (one dict for the controller, sessions and presets) --- #

    def get_filter_params(self) -> dict:
        """Method, filter and Auto parameters (0 = filter off)."""
        return {
            **self.classic.filter_values(),
            "detection_mode": self.method(),
            "edge_definition": self.edge_box.edge_definition(),
            "edge_polarity": self.edge_box.polarity(),
            "bubble_non_round": self.edge_box.non_round(),
        }

    def set_filter_params(self, params: dict) -> None:
        """Set the method, filters and Auto options without emitting signals."""
        index = self._method_combo.findData(params.get("detection_mode", AUTO_METHOD))
        self._method_combo.blockSignals(True)
        self._method_combo.setCurrentIndex(max(index, 0))
        self._method_combo.blockSignals(False)
        self.edge_box.set_options(params.get("edge_definition", "steep"),
                                  params.get("edge_polarity", "dark"),
                                  bool(params.get("bubble_non_round", False)))
        self.classic.set_filter_values(params)
        self._apply_visibility()

    # Classic values (used by the controllers and sessions).
    def get_threshold(self) -> float:
        return self.classic.get_threshold()

    def set_threshold(self, value: float) -> None:
        self.classic.set_threshold(value)

    def get_invert_mask(self) -> bool:
        return self.classic.get_invert_mask()

    def set_invert_mask(self, value: bool) -> None:
        self.classic.set_invert_mask(value)

    def get_removing_factor_slider(self) -> int:
        return self.classic.get_removing_factor_slider()

    def set_removing_factor(self, value: int) -> None:
        self.classic.set_removing_factor(value)

    def get_edge_flags(self) -> list[bool]:
        return self.classic.get_edge_flags()

    def set_edge_flags(self, flags) -> None:
        self.classic.set_edge_flags(flags)

    def show_quality(self, n_pts: int, rms: float, score_pct: int) -> None:
        self.classic.show_quality(n_pts, rms, score_pct)

    def show_area_result(self, area: int, count: int) -> None:
        self.classic.show_area_result(area, count)

    def hide_quality(self) -> None:
        self.classic.hide_quality()

    def set_autotune_enabled(self, enabled: bool) -> None:
        self.classic.set_autotune_enabled(enabled)

    # -- presets (classic methods) ----------------------------------------- #

    def preset(self) -> dict:
        return {
            "threshold": self.get_threshold(),
            "removing_factor": self.get_removing_factor_slider(),
            "edge_flags": self.get_edge_flags(),
            **self.get_filter_params(),
        }

    def apply_preset(self, preset: dict) -> None:
        """Apply a saved preset and announce the changed values."""
        if "threshold" in preset:
            self.set_threshold(float(preset["threshold"]))
            self.threshold_changed.emit(self.get_threshold())
        if "removing_factor" in preset:
            self.set_removing_factor(int(preset["removing_factor"]))
            self.removing_factor_changed.emit(self.get_removing_factor_slider())
        if "edge_flags" in preset:
            self.classic.set_edge_flags([bool(v) for v in preset["edge_flags"]])
            self.edges_changed.emit()
        filters = {key: preset[key] for key in _FILTER_KEYS if key in preset}
        self.set_filter_params({**self.get_filter_params(), **filters})
        self.filters_changed.emit()

    def _save_preset(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save Filter Preset", "", "JSON Files (*.json)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.preset(), f, indent=2)
        except OSError as exc:
            QMessageBox.warning(self, "Preset not saved", f"Could not write {path}:\n{exc}")

    def _load_preset(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load Filter Preset", "", "JSON Files (*.json)")
        if not path:
            return
        try:
            with open(path, encoding="utf-8") as f:
                preset = json.load(f)
            if not isinstance(preset, dict):
                raise ValueError("not a preset (expected a JSON object)")
            self.apply_preset(preset)
        except (OSError, TypeError, ValueError) as exc:
            logger.warning("Preset %s not loaded: %s", path, exc)
            QMessageBox.warning(self, "Preset not loaded", f"Could not load {path}:\n{exc}")
