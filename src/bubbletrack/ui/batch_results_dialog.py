"""Batch Results Browser — non-modal dialog for inspecting batch results."""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from bubbletrack.model.batch_result import (
    BatchResultStore,
    FolderResult,
    format_score,
    mean_score,
)
from bubbletrack.model.constants import QUALITY_GOOD_THRESHOLD, QUALITY_WARN_THRESHOLD
from bubbletrack.ui.radius_chart import RadiusChart

_MAX_ERROR_LINES = 8  # longer summaries are truncated with "... and N more"


def item_colours(result: FolderResult) -> tuple[str, QColor]:
    """(foreground, background tint) for a source in the list."""
    if not result.success:
        return "#71717a", QColor(113, 113, 122, 20)  # gray
    if result.n_fitted == 0:
        return "#d4d4d8", QColor(239, 68, 68, 25)  # red
    score = mean_score(result)
    if not np.isfinite(score):
        return "#d4d4d8", QColor(0, 0, 0, 0)  # not scored (area mode)
    if score >= QUALITY_GOOD_THRESHOLD:
        return "#d4d4d8", QColor(16, 185, 129, 25)  # green
    if score >= QUALITY_WARN_THRESHOLD:
        return "#d4d4d8", QColor(245, 158, 11, 25)  # amber
    return "#d4d4d8", QColor(239, 68, 68, 25)  # red


def item_text(result: FolderResult) -> str:
    kind = "video · " if result.source_kind == "video" else ""
    if not result.success and not result.has_results:
        return f"{result.folder_name}\n{kind}Failed: {result.message}"
    status = "" if result.success else " · FAILED"
    return (f"{result.folder_name}\n{kind}{result.n_fitted}/{result.n_frames} fitted · "
            f"{format_score(result)}{status}")


def error_text(store: BatchResultStore) -> str:
    """Per-source error summary ('' when nothing failed)."""
    lines = store.error_summary()
    if not lines:
        return ""
    n_failed = len(store.failures)
    head = f"{n_failed} source(s) failed:" if n_failed else "Batch problem:"
    shown = lines[:_MAX_ERROR_LINES]
    if len(lines) > len(shown):
        shown.append(f"... and {len(lines) - len(shown)} more (see the log)")
    return "\n".join([head] + [f"  • {line}" for line in shown])


class BatchResultsBrowserDialog(QDialog):
    """Non-modal dialog showing all batch source results with R-t preview.

    Signals
    -------
    load_folder_requested(object)
        Emitted with the selected ``FolderResult`` when the user clicks
        "Load into Main View".
    """

    load_folder_requested = pyqtSignal(object)

    def __init__(
        self,
        store: BatchResultStore,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._store = store
        self._selected_result: FolderResult | None = None

        self.setWindowTitle("Batch Results Browser")
        self.setMinimumSize(900, 600)
        self.resize(1000, 650)
        # Non-modal: user can interact with main window
        self.setModal(False)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

        self._build_ui()
        self._populate_folder_list()

    # ------------------------------------------------------------------ #
    #  UI construction
    # ------------------------------------------------------------------ #

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        results = self._store.results
        n_ok = sum(1 for r in results if r.success)
        total_fitted = sum(r.n_fitted for r in results)
        total_frames = sum(r.n_frames for r in results)
        self._header = QLabel(
            f"Batch Results — {n_ok}/{len(results)} sources, "
            f"{total_fitted}/{total_frames} frames fitted"
        )
        self._header.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        self._header.setStyleSheet("color: #e4e4e7; padding: 4px 0;")
        root.addWidget(self._header)

        self._error_label = QLabel(error_text(self._store))
        self._error_label.setWordWrap(True)
        self._error_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._error_label.setStyleSheet(
            "color: #fca5a5; background: rgba(239, 68, 68, 0.08); "
            "border-radius: 4px; padding: 6px;")
        self._error_label.setVisible(bool(self._error_label.text()))
        root.addWidget(self._error_label)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(3)

        # -- Left: source list ----------------------------------------- #
        self._folder_list = QListWidget()
        self._folder_list.setMinimumWidth(220)
        self._folder_list.setMaximumWidth(350)
        self._folder_list.setFont(QFont("Segoe UI", 10))
        self._folder_list.currentRowChanged.connect(self._on_folder_clicked)
        splitter.addWidget(self._folder_list)

        # -- Right: detail + chart + load button ----------------------- #
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(8, 0, 0, 0)
        right_layout.setSpacing(6)

        self._detail_label = QLabel("Select a source to preview results.")
        self._detail_label.setFont(QFont("Segoe UI", 10))
        self._detail_label.setStyleSheet("color: #a1a1aa;")
        self._detail_label.setWordWrap(True)
        right_layout.addWidget(self._detail_label)

        # Read-only RadiusChart preview (no signals connected)
        self._chart = RadiusChart(parent=right_panel)
        right_layout.addWidget(self._chart, stretch=1)

        self._load_btn = QPushButton("Load into Main View")
        self._load_btn.setObjectName("primaryBtn")
        self._load_btn.setFixedHeight(36)
        self._load_btn.setEnabled(False)
        self._load_btn.setStyleSheet(
            "QPushButton { background: #6366f1; color: white; border-radius: 6px; "
            "font-weight: bold; font-size: 13px; }"
            "QPushButton:hover { background: #818cf8; }"
            "QPushButton:disabled { background: #3f3f46; color: #71717a; }"
        )
        self._load_btn.clicked.connect(self._on_load_clicked)
        right_layout.addWidget(self._load_btn)

        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)  # source list
        splitter.setStretchFactor(1, 3)  # chart panel

        root.addWidget(splitter, stretch=1)

        close_btn = QPushButton("Close")
        close_btn.setObjectName("secondaryBtn")
        close_btn.setFixedHeight(32)
        close_btn.clicked.connect(self.close)
        bottom_row = QHBoxLayout()
        bottom_row.addStretch()
        bottom_row.addWidget(close_btn)
        root.addLayout(bottom_row)

    def _populate_folder_list(self) -> None:
        for result in self._store.results:
            item = QListWidgetItem(item_text(result))
            item.setFont(QFont("Segoe UI", 9))
            item.setToolTip(result.message)
            foreground, background = item_colours(result)
            item.setForeground(QColor(foreground))
            item.setBackground(background)
            self._folder_list.addItem(item)

    # ------------------------------------------------------------------ #
    #  Event handlers
    # ------------------------------------------------------------------ #

    def _on_folder_clicked(self, row: int) -> None:
        if row < 0 or row >= len(self._store.results):
            return
        result = self._store.results[row]
        self._selected_result = result
        self._chart.set_total_frames(result.n_frames)
        if result.has_results:
            frames = np.arange(1, len(result.radius) + 1)
            self._chart.plot_all(frames, result.radius, result.quality_scores)
        else:
            self._chart.clear()

        lines = [f"Path: {result.folder_path}",
                 f"Type: {result.source_kind}",
                 f"Fitted: {result.n_fitted}/{result.n_frames} frames",
                 f"{format_score(result)}",
                 f"Status: {result.message}"]
        lines.extend(f"Saved: {path}" for path in result.outputs)
        self._detail_label.setText("\n".join(lines))
        self._detail_label.setStyleSheet(
            "color: #e4e4e7;" if result.success else "color: #fca5a5;")
        self._load_btn.setEnabled(result.has_results)

    def _on_load_clicked(self) -> None:
        if self._selected_result is not None and self._selected_result.has_results:
            self.load_folder_requested.emit(self._selected_result)
