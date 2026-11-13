"""Export Hard Case: frames the app gets wrong, with the user's fixes, as one .zip.

Entry points: File > Export Hard Case... (the chart's brushed range if any,
else the selected Review row, else the shown frame; the dialog lets the
user change it) and "Export as hard case..." on a Review row's right-click
menu.  The model layer (``model.cases``) builds and writes the bundle; this
controller adds the dialogs and remembers which frames the user deleted,
because a deleted result is the label "no bubble here".
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QDialog, QFileDialog, QMessageBox

from bubbletrack.controller import edge_support
from bubbletrack.controller.display_mixin import current_source
from bubbletrack.event_bus import FRAMES_DELETED, RESULTS_CHANGED
from bubbletrack.model import config, edge_pipeline
from bubbletrack.model.cases.build import (
    SIZE_WARN_BYTES, CaseRequest, clip_range, default_file_name, estimate_bytes, make_case,
    plan_frames, write_case,
)
from bubbletrack.model.cases.labels import app_result, frame_label
from bubbletrack.model.cases.schema import HardCase, ROLE_CONTEXT, ROLE_REFERENCE
from bubbletrack.model.state import AppState
from bubbletrack.ui.hard_case_dialog import HardCaseDialog, HardCaseInput

logger = logging.getLogger(__name__)

_OK, _WARN, _ERR = "#10b981", "#f59e0b", "#ef4444"
_MB = 1024 * 1024


def _megabytes(n_bytes: int) -> str:
    return f"{n_bytes / _MB:.1f} MB" if n_bytes < 10 * _MB else f"{n_bytes / _MB:.0f} MB"


class HardCaseController:
    """Asks what to export and writes the bundle."""

    def __init__(self, window, app) -> None:
        self.w = window
        self._app = app
        self._deleted: set[int] = set()  # frames whose result the user deleted ...
        self._deleted_from = None  # ... in this result array (Clear / load: a new one)
        self._sample: tuple[tuple, np.ndarray] | None = None  # (source key, one raw frame)
        bus = app.bus
        bus.subscribe(FRAMES_DELETED, self._on_frames_deleted)
        bus.subscribe(RESULTS_CHANGED, lambda *_: self._prune_deleted())
        bus.subscribe("source_loaded", lambda *_: self._forget())
        bus.subscribe("run_state_changed", self._on_run_state)
        window.export_hard_case_requested.connect(self.on_export_requested)
        app.review.panel.export_case_requested.connect(
            lambda segment: self.ask_and_export(segment.start, segment.end))

    # -- deleted frames ------------------------------------------------------ #

    def deleted_frames(self) -> frozenset[int]:
        """Frames the user deleted that still have no result."""
        self._prune_deleted()
        return frozenset(self._deleted)

    def _on_frames_deleted(self, frames) -> None:
        state = self._state()
        if state.radius is not self._deleted_from:
            self._forget()
            self._deleted_from = state.radius
        self._deleted.update(int(i) for i in frames)

    def _prune_deleted(self) -> None:
        radius = self._state().radius
        if radius is not self._deleted_from:  # results were cleared or replaced
            self._forget()
            return
        self._deleted = {i for i in self._deleted if i < len(radius) and not radius[i] > 0}

    def _forget(self) -> None:
        self._deleted, self._deleted_from, self._sample = set(), None, None

    # -- entry points ---------------------------------------------------------- #

    def default_range(self) -> tuple[int, int]:
        """The chart's brushed range, else the selected Review row, else the shown frame."""
        selected = self.w.radius_chart.selected_range()
        if selected is not None:
            return selected
        segment = self._app.review.panel.current_segment()
        if segment is not None:
            return segment.start, segment.end
        frame = self._state().image_no
        return frame, frame

    def on_export_requested(self) -> None:
        self.ask_and_export(*self.default_range())

    def ask_and_export(self, start: int, end: int) -> HardCase | None:
        """Dialog, size check, file dialog, export; None if cancelled or refused."""
        if not self._can_export():
            return None
        state = self._state()
        start, end = clip_range(start, end, len(state.images))
        reporter = config.get_meta("hard_case_reporter") or ""
        dialog = HardCaseDialog(self.w, n_frames=len(state.images), start=start, end=end,
                                reporter=reporter, describe=self.describe)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None
        values = dialog.values()
        request = self.make_request(values)
        if not self._confirm_size(request):
            return None
        path = self._ask_path(request)
        if not path:
            return None
        exported = self.export(request, path)
        return exported[0] if exported is not None else None

    # -- export (no dialogs) ------------------------------------------------------ #

    def make_request(self, values: HardCaseInput) -> CaseRequest:
        state = self._state()
        edge = state.detection_mode == "edge"
        return CaseRequest(
            start=values.start, end=values.end, note=values.note, reporter=values.reporter,
            reference_index=self._reference_index() if edge else None,
            tracking=edge and self.w.left_panel.pretune_tab.edge_box.tracking_enabled(),
            max_radius=self._app._get_max_radius(), deleted=self.deleted_frames(),
        )

    def export(self, request: CaseRequest, path: str) -> tuple[HardCase, int] | None:
        """Write the bundle to *path*; (case, bytes) or None (reported) on failure."""
        try:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                case = make_case(self._state(), request, self._sample_frame().shape)
                written, size = write_case(path, case, self._read_raw)
            finally:
                QApplication.restoreOverrideCursor()
        except Exception as exc:
            logger.exception("Hard-case export to %s failed", path)
            self.w.header.set_status(f"Hard case not saved: {exc}", _ERR)
            return None
        self._remember(path, request.reporter)
        labels = sum(f.label.is_labelled for f in written.measured_frames())
        logger.info("Hard case %s: %d frames, %d labels, %d bytes",
                    path, len(written.frames), labels, size)
        self.w.header.set_status(f"Hard case saved: {os.path.basename(path)} "
                                 f"({_megabytes(size)}, {labels} labelled frame(s)); "
                                 "send it to the maintainer", _OK)
        return written, size

    def describe(self, start: int, end: int) -> str:
        """Summary for the dialog: frames, file size, labels."""
        state = self._state()
        edge = state.detection_mode == "edge"
        plan = plan_frames(start, end, len(state.images),
                           reference=self._reference_index() if edge else None)
        roles = [role for _, role in plan]
        parts = [f"{end - start + 1} frame(s)"]
        if roles.count(ROLE_CONTEXT):
            parts.append(f"{roles.count(ROLE_CONTEXT)} neighbours")
        if ROLE_REFERENCE in roles:
            parts.append(f"reference frame {next(i for i, r in plan if r == ROLE_REFERENCE) + 1}")
        size = self._estimate(len(plan))
        deleted = self.deleted_frames()
        labels = sum(frame_label(app_result(state, i), deleted=i in deleted).is_labelled
                     for i, role in plan if role != ROLE_REFERENCE)
        text = " + ".join(parts) + (f" · up to {_megabytes(size)}" if size else "")
        if labels:
            return text + f"\n{labels} frame(s) fixed by you (manual, accepted or deleted)"
        return text + ("\nNo frame here is fixed yet: fix or accept the right frames first "
                       "so the case can be scored (it is still worth sending).")

    # -- helpers -------------------------------------------------------------------- #

    def _state(self) -> AppState:
        return self._app._get_state()

    def _can_export(self) -> bool:
        if not self._state().images:
            self.w.header.set_status("Load images or a recording first", _WARN)
            return False
        if self._app.auto_ctrl.is_busy():
            self.w.header.set_status("Stop the running fit before exporting a hard case", _WARN)
            return False
        return True

    def _reference_index(self) -> int | None:
        try:
            return edge_support.reference_index(self._state())
        except Exception as exc:  # reading frames can fail; the case can do without
            logger.warning("Could not resolve the reference frame: %s", exc)
            return None

    def _read_raw(self, idx: int) -> np.ndarray:
        """Frame *idx* as the app processes it (file pixels or decoded video frame)."""
        return edge_pipeline.read_raw(current_source(self._state(), idx))

    def _sample_frame(self) -> np.ndarray:
        state = self._state()
        key = (state.folder_path, state.video_path, len(state.images))
        if self._sample is None or self._sample[0] != key:
            self._sample = (key, self._read_raw(state.image_no))
        return self._sample[1]

    def _estimate(self, n_frames: int) -> int:
        try:
            state = self._state()
            return estimate_bytes(n_frames, state.gridx, state.gridy, self._sample_frame())
        except Exception as exc:
            logger.warning("Could not estimate the hard-case size: %s", exc)
            return 0

    def _confirm_size(self, request: CaseRequest) -> bool:
        n = len(plan_frames(request.start, request.end, len(self._state().images),
                            reference=request.reference_index))
        size = self._estimate(n)
        if size <= SIZE_WARN_BYTES:
            return True
        answer = QMessageBox.question(
            self.w, "Large hard case",
            f"This hard case will be up to {_megabytes(size)} ({n} frames). Large files are "
            "slow to send; a shorter range usually shows the problem just as well.\n\n"
            "Save it anyway?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        return answer == QMessageBox.StandardButton.Yes

    def _ask_path(self, request: CaseRequest) -> str:
        name = default_file_name(self._state(), request.start, request.end)
        path, _ = QFileDialog.getSaveFileName(
            self.w, "Save Hard Case", os.path.join(self._default_dir(), name),
            "Hard case (*.zip);;All Files (*)")
        if path and not path.lower().endswith(".zip"):
            path += ".zip"
        return path

    def _default_dir(self) -> str:
        """The last export folder, else next to the data (not inside an image folder)."""
        remembered = config.get_meta("last_export_dir")
        if remembered and os.path.isdir(remembered):
            return remembered
        state = self._state()
        base = os.path.dirname(state.video_path) if state.video_path else \
            str(Path(state.folder_path).parent) if state.folder_path else ""
        return base if base and os.path.isdir(base) else os.path.expanduser("~")

    @staticmethod
    def _remember(path: str, reporter: str) -> None:
        try:
            config.set_meta(last_export_dir=os.path.dirname(os.path.abspath(path)),
                            hard_case_reporter=reporter)
        except ValueError as exc:
            logger.warning("Hard-case choices not remembered: %s", exc)

    def _on_run_state(self, running: bool) -> None:
        self.w.export_hard_case_action.setEnabled(not running)


def install_hard_cases(window, app_controller) -> HardCaseController:
    """Wire hard-case export into *window* (needs the Review dock installed first)."""
    existing = getattr(app_controller, "hard_cases", None)
    if isinstance(existing, HardCaseController):
        return existing
    return HardCaseController(window, app_controller)
